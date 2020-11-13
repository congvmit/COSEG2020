# export CUDA_VISIBLE_DEVICES=1
python run_net.py train \
--data_folder "/mnt/DATA2/congvm/COVID-19-20_v2/Train" \
--model_folder "runs" \
--batch_size 2 \
--num_workers 8 \
--preprocessing_workers 8 \
--cache_rate 1.0 \
--lr 0.01 \
--n_slice 16 \
--patch_size 256 \
--gamma 0.8 \
--min_lr 0.0001 \
--momentum 0.95 \
--opt "adam" \