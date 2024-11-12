# Thực hiện cài đặt thư viện (update thư viện)
pip install -r 1111_requirements.txt

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
## Step 1: Run the following command to train a mT5-large cross-lingual query generation model.

python3 run.py \
        --task "docTquery" \
        --model_name "google/mt5-large" \
        --run_name "docTquery-MSMARCO" \
        --max_length 128 \
        --train_file data/msmarco_data/100k/msmarco_docTquery_train_data.json \
        --valid_file data/msmarco_data/100k/msmarco_docTquery_dev_data.json \
        --output_dir "models/msmarco_docTquery_mt5_large" \
        --learning_rate 0.0001 \
        --warmup_steps 0 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --evaluation_strategy steps \
        --eval_steps 100 \
        --max_steps 2000 \
        --save_strategy steps \
        --dataloader_num_workers 10 \
        --save_steps 100 \
        --save_total_limit 2 \
        --load_best_model_at_end \
        --gradient_accumulation_steps 4 \
        --report_to wandb \
        --logging_steps 100 \
        --dataloader_drop_last False

Do đó để tiết kiệm thời gian thì sử dụng pretrained mà họ đã cung cấp: castorini/doc2query-t5-large-msmarco

## Step 2: run the query generation for all the documents in the corpus

python3 run.py \
        --task generation \
        --model_name google/mt5-large \
        --model_path models/xorqa_docTquery_mt5_large/checkpoint-xxx \
        --per_device_eval_batch_size 32 \
        --run_name docTquery-XORQA-generation \
        --max_length 256 \
        --valid_file data/xorqa_data/100k/xorqa_corpus.tsv \
        --output_dir temp \
        --dataloader_num_workers 10 \
        --report_to wandb \
        --logging_steps 100 \
        --num_return_sequences 10

## Step 3: Train DSI-QG with query-represented corpus
python3 run.py \
        --task "DSI" \
        --model_name "google/mt5-base" \
        --run_name "XORQA-100k-mt5-base-DSI-QG" \
        --max_length 32 \
        --train_file data/xorqa_data/100k/xorqa_corpus.tsv.q10.docTquery \
        --valid_file data/xorqa_data/100k/xorqa_DSI_dev_data.json \
        --output_dir "models/XORQA-100k-mt5-base-DSI-QG" \
        --learning_rate 0.0005 \
        --warmup_steps 100000 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --evaluation_strategy steps \
        --eval_steps 1000 \
        --max_steps 1000000 \
        --save_strategy steps \
        --dataloader_num_workers 10 \
        --save_steps 1000 \
        --save_total_limit 2 \
        --load_best_model_at_end \
        --gradient_accumulation_steps 1 \
        --report_to wandb \
        --logging_steps 100 \
        --dataloader_drop_last False \
        --metric_for_best_model Hits@10 \
        --greater_is_better True \
        --remove_prompt True
