export CUDA_VISIBLE_DEVICES=1
python run_net.py train \
--data_folder "/mnt/DATA2/congvm/COVID-19-20_v2/Train" \
--model_folder "runs" \
--batch_size 4 \
--num_workers 8 \
--preprocessing_workers 4 \
--lr 0.01 \
--momentum 0.95 \
--opt "adam" \