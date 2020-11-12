# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import logging
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar

import monai
from monai.handlers import CheckpointSaver, MeanDice, StatsHandler, ValidationHandler, LrScheduleHandler
from monai.transforms import (
    AddChanneld,
    CastToTyped,
    LoadNiftid,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)

from scheduler import BoundingExponentialLR
from helpers import save_args_to_file

def get_xforms(mode="train", keys=("image", "label")):
    """returns a composed transform for train/val/infer."""

    xforms = [
        LoadNiftid(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(1.25, 1.25, 5.0), mode=("bilinear", "nearest")[: len(keys)]),
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
    ]
    if mode == "train":
        xforms.extend(
            [
                SpatialPadd(keys, spatial_size=(192, 192, -1), mode="reflect"),  # ensure at least 192x192
                RandAffined(
                    keys,
                    prob=0.15,
                    rotate_range=(-0.05, 0.05),
                    scale_range=(-0.1, 0.1),
                    mode=("bilinear", "nearest"),
                    as_tensor_output=False,
                ),
                RandCropByPosNegLabeld(keys, label_key=keys[1], spatial_size=(192, 192, 16), num_samples=3),
                RandGaussianNoised(keys[0], prob=0.15, std=0.01),
                RandFlipd(keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys, spatial_axis=1, prob=0.5),
                RandFlipd(keys, spatial_axis=2, prob=0.5),
            ]
        )
        dtype = (np.float32, np.uint8)
    if mode == "val":
        dtype = (np.float32, np.uint8)
    if mode == "infer":
        dtype = (np.float32,)
    xforms.extend([CastToTyped(keys, dtype=dtype), ToTensord(keys)])
    return monai.transforms.Compose(xforms)


def get_net():
    """returns a unet model instance."""
    n_classes = 2
    net = monai.networks.nets.BasicUNet(
        dimensions=3,
        in_channels=1,
        out_channels=n_classes,
        features=(32, 32, 64, 128, 256, 32),
        dropout=0.1,
    )
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    return net


def get_inferer(_mode=None):
    """returns a sliding window inference instance."""

    patch_size = (192, 192, 16)
    sw_batch_size, overlap = 2, 0.5
    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
    )
    return inferer


class DiceCELoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        return dice + cross_entropy


def train(args):
    """run a training pipeline."""
    
    save_args_to_file(args, 'runs/')
    images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_folder, "*_seg.nii.gz")))
    logging.info(f"training: image/label ({len(images)}) folder: {args.data_folder}")

    amp = True  # auto. mixed precision
    keys = ("image", "label")
    
    #TODO
    is_one_hot = False  # whether the label has multiple channels to represent  multiple class
    
    train_frac, val_frac = 0.8, 0.2
    n_train = int(train_frac * len(images)) + 1
    n_val = min(len(images) - n_train, int(val_frac * len(images)))
    logging.info(f"training: train {n_train} val {n_val}, folder: {args.data_folder}")

    train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[:n_train], labels[:n_train])]
    val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[-n_val:], labels[-n_val:])]

    # create a training data loader
    logging.info(f"batch size {args.batch_size}")
    train_transforms = get_xforms("train", keys)
    train_ds = monai.data.CacheDataset(data=train_files, 
                                       transform=train_transforms, 
                                       cache_rate=args.cache_rate,
                                       num_workers=args.preprocessing_workers)
    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # create a validation data loader
    val_transforms = get_xforms("val", keys)
    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)
    val_loader = monai.data.DataLoader(
        val_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # create BasicUNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_net().to(device)
    
    logging.info(f"epochs {args.max_epochs}, lr {args.lr}, momentum {args.momentum}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    # create evaluator (to be used to measure model quality during training
    def pred_transform(y_pred):
        y_sigmoid = torch.sigmoid(y_pred)
        y_sigmoid = (y_sigmoid >= logit_thresh).float()
        return y_sigmoid
    
    logit_thresh = 0.5
    train_metric = MeanDice(
        include_background=False,
        device = device,
        output_transform=lambda x: (pred_transform(x["pred"]), x["label"]),
    )
    
    val_metric = MeanDice(
        include_background=False,
        device = device,
        output_transform=lambda x: (pred_transform(x["pred"]), x["label"]),
    )
    
        
    val_handlers = [
        ProgressBar(),
        CheckpointSaver(save_dir=args.model_folder, save_dict={'net': net, 
                                                               'optimizer': opt},
                        save_key_metric=True, key_metric_n_saved=3),
    ]
    evaluator = monai.engines.SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=get_inferer(),
        key_val_metric={"val_mean_dice": val_metric},
        val_handlers=val_handlers,
        amp=amp,
    )

    # evaluator as an event handler of the trainer
    train_handlers = [
        ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]),
        LrScheduleHandler(BoundingExponentialLR(opt, gamma=args.gamma), 
                          print_lr=True, 
                          name='bounding_lr_scheduler', 
                          epoch_level=True,)
    ]
    
    trainer = monai.engines.SupervisedTrainer(
        device=device,
        max_epochs=args.max_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=DiceCELoss(),
        inferer=get_inferer(),
        key_train_metric={'train_mean_dice': train_metric},
        train_handlers=train_handlers,
        amp=amp,
    )
    trainer.run()

def infer(data_folder=".", model_folder="runs", prediction_folder="output"):
    """
    run inference, the output folder will be "./output"
    """
    ckpts = sorted(glob.glob(os.path.join(model_folder, "*.pt")))
    ckpt = ckpts[-1]
    for x in ckpts:
        logging.info(f"available model file: {x}.")
    logging.info("----")
    logging.info(f"using {ckpt}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_net().to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device)['net'])
    net.eval()

    image_folder = os.path.abspath(data_folder)
    images = sorted(glob.glob(os.path.join(image_folder, "*_ct.nii.gz")))
    logging.info(f"infer: image ({len(images)}) folder: {data_folder}")
    infer_files = [{"image": img} for img in images]

    keys = ("image",)
    infer_transforms = get_xforms("infer", keys)
    infer_ds = monai.data.Dataset(data=infer_files, transform=infer_transforms)
    infer_loader = monai.data.DataLoader(
        infer_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    inferer = get_inferer()
    saver = monai.data.NiftiSaver(output_dir=prediction_folder, mode="nearest")
    with torch.no_grad():
        for infer_data in infer_loader:
            logging.info(f"segmenting {infer_data['image_meta_dict']['filename_or_obj']}")
            preds = inferer(infer_data[keys[0]].to(device), net)
            n = 1.0
            for _ in range(4):
                # test time augmentations
                _img = RandGaussianNoised(keys[0], prob=1.0, std=0.01)(infer_data)[keys[0]]
                pred = inferer(_img.to(device), net)
                preds = preds + pred
                n = n + 1.0
                for dims in [[2], [3]]:
                    flip_pred = inferer(torch.flip(_img.to(device), dims=dims), net)
                    pred = torch.flip(flip_pred, dims=dims)
                    preds = preds + pred
                    n = n + 1.0
            preds = preds / n
            preds = (preds.argmax(dim=1, keepdims=True)).float()
            saver.save_batch(preds, infer_data["image_meta_dict"])

    # copy the saved segmentations into the required folder structure for submission
    submission_dir = os.path.join(prediction_folder, "to_submit")
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)
    files = glob.glob(os.path.join(prediction_folder, "volume*", "*.nii.gz"))
    for f in files:
        new_name = os.path.basename(f)
        new_name = new_name[len("volume-covid19-A-0"):]
        new_name = new_name[: -len("_ct_seg.nii.gz")] + ".nii.gz"
        to_name = os.path.join(submission_dir, new_name)
        shutil.copy(f, to_name)
    logging.info(f"predictions copied to {submission_dir}.")

def get_args():
    parser = argparse.ArgumentParser(description="Run a basic UNet segmentation baseline.")
    parser.add_argument(
        "mode", metavar="mode", default="train", choices=("train", "infer"), type=str, help="mode of workflow"
    )
    parser.add_argument("--data_folder", default="", type=str, help="training data folder")
    parser.add_argument("--model_folder", default="runs", type=str, help="model folder")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--num_workers", default=8, type=int, help="num workers")
    parser.add_argument("--preprocessing_workers", default=8, type=int, help="preprocessing workers")
    parser.add_argument("--opt", default='adam', type=str, choices=("adam", "sgd"), help="opt")
    parser.add_argument("--cache_rate", default=0.5, type=float, help="cache rate")
    parser.add_argument("--momentum", default=0.95, type=float, help="opt momentum")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--gamma", default=0.5, type=float, help="lr scheduler gamma")
    parser.add_argument("--max_epochs", default=500, type=int, help="lr scheduler gamma")
    parser.add_argument("--seed", default=0, type=int, help="random seed")   
    parser.add_argument("--prediction_folder", default='output', type=str, help="random seed")   
    
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    """
    Usage:
        python run_net.py train --data_folder "COVID-19-20_v2/Train" # run the training pipeline
        python run_net.py infer --data_folder "COVID-19-20_v2/Validation" # run the inference pipeline
    """
    
    args = get_args()
    
    monai.config.print_config()
    monai.utils.set_determinism(seed=args.seed)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if args.mode == "train":
        data_folder = args.data_folder or os.path.join("COVID-19-20_v2", "Train")
        train(args)
    elif args.mode == "infer":
        data_folder = args.data_folder or os.path.join("COVID-19-20_v2", "Validation")
        infer(data_folder=data_folder, model_folder=args.model_folder)
    else:
        raise ValueError("Unknown mode.")
