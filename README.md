# COVID-19 Lung CT Lesion Segmentation Challenge - 2020

## Introduction

The COVID-19 pandemic has had a devastating impact on the health of individuals worldwide. The manifestation of the viral infection in the lung has been one of the earliest indicators of disease and may play an important role in the clinical management of patients. Ground glass opacities are the most common finding in COVID-19 lung computed tomography (CT) images, usually multifocal, bilateral and peripheral. However, the type, the size and distribution of the lung lesions may vary with the age of the patients and the severity or stage of the disease.

The COVID-19-20 challenge will create the platform to evaluate emerging methods for the segmentation and quantification of lung lesions caused by SARS-CoV-2 infection from CT images. The images are multi-institutional, multi-national and originate from patients of different ages, gender and with variable disease severity.

## Timeline

November 2nd, 2020 (11:59PM GMT): Launch of challenge and release of training and validation data.

__December 7th, 2020 (11:59PM GMT): Release of test data.__

__December 10th, 2020 (11:59PM GMT): Deadline for submission of test results and abstract.__

December 2020 (TBD): Ranking of results will be available.


## Installation

Follow the instruction here: https://docs.monai.io/en/latest/installation.html

```
pip install git+https://github.com/Project-MONAI/MONAI#egg=MONAI
```

## Training

To train, run this file `/train.sh`

```python
python run_net.py train \
--data_folder "COVID-19-20_v2/Train" \
--model_folder "runs" \
--batch_size 4 \
--num_workers 8 \
--preprocessing_workers 4 \
--lr 0.01 \
--momentum 0.95 \
--opt "adam" \
```