export CUDA_VISIBLE_DEVICES=1
python run_net.py infer \
--data_folder "/mnt/DATA2/congvm/COVID-19-20_v2/Train" \
--model_folder "runs" \
--prediction_folder "outputs" \