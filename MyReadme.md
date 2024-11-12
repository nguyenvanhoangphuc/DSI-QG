# Thực hiện cài đặt thư viện (update thư viện)
pip install -r 1111_requirements.txt

## Data Preparing
Simply run `bash get_data.sh`. 

# Thực hiện huấn luyện DSI Original cho tiếng Anh (MSMarco)
python3 run.py \
        --task "DSI" \
        --model_name "google/mt5-base" \
        --run_name "MSMARCO-100k-mt5-base-DSI" \
        --max_length 256 \
        --train_file data/msmarco_data/100k/msmarco_DSI_train_data.json \
        --valid_file data/msmarco_data/100k/msmarco_DSI_dev_data.json \
        --output_dir "models/MSMARCO-100k-mt5-base-DSI" \
        --learning_rate 0.0005 \
        --warmup_steps 10000 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 2 \
        --evaluation_strategy steps \
        --eval_steps 1000 \
        --max_steps 100000 \
        --save_strategy steps \
        --dataloader_num_workers 2 \
        --save_steps 1000 \
        --save_total_limit 2 \
        --load_best_model_at_end \
        --gradient_accumulation_steps 2 \
        --report_to wandb \
        --logging_steps 100 \
        --dataloader_drop_last False \
        --metric_for_best_model Recall@50 \
        --greater_is_better True

Thời gian huấn luyện: 14 tiếng 

# Thực hiện huấn luyện DSI-QG
## Step 1:

Do đó để tiết kiệm thời gian thì sử dụng pretrained mà họ đã cung cấp.