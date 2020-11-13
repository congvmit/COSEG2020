CUDA_VISIBLE_DEVICES="0,1"
python dist_run_net.py train \
--data_folder "/mnt/DATA2/congvm/COVID-19-20_v2/Train" \
--model_folder "runs" \
--batch_size 4 \
--num_workers 8 \
--preprocessing_workers 8 \
--lr 0.01 \
--momentum 0.95 \
--cache_rate 0.1 \
--opt "adam" \
--rank 1 \
--multiprocessing-distributed \


#--gpu '0,1' \