#!/bin/bash

# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=0

# Disable NCCL P2P and IB for compatibility with RTX 4000 series
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Train DSI
python3 run.py \
        --task "DSI" \
        --model_name "google/mt5-base" \
        --run_name "LEGAL-ja-mt5-base-DSI" \
        --max_length 256 \
        --train_file data/legal_data/ja_dsi_ds/train_data_dsi.json \
        --valid_file data/legal_data/ja_dsi_ds/validation_data_dsi.json \
        --output_dir "models/LEGAL-ja-mt5-base-DSI" \
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

# Thực hiện huấn luyện DSI-QG
## Step 1: Run the following command to train a mT5-large cross-lingual query generation model.

python3 run.py \
        --task "docTquery" \
        --model_name "google/mt5-large" \
        --run_name "docTquery-LEGAL" \
        --max_length 128 \
        --train_file data/legal_data/ja_dsi_ds/train_data_dsi.json \
        --valid_file data/legal_data/ja_dsi_ds/validation_data_dsi.json \
        --output_dir "models/legal_docTquery_mt5_large" \
        --learning_rate 0.0001 \
        --warmup_steps 0 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --evaluation_strategy steps \
        --eval_steps 100 \
        --max_steps 2000 \
        --save_strategy steps \
        --dataloader_num_workers 10 \
        --save_steps 100 \
        --save_total_limit 2 \
        --load_best_model_at_end \
        --gradient_accumulation_steps 2 \
        --report_to wandb \
        --logging_steps 100 \
        --dataloader_drop_last False

## Step 2: run the query generation for all the documents in the corpus

python3 run.py \
        --task generation \
        --model_name google/mt5-large \
        --model_path models/legal_docTquery_mt5_large/checkpoint-xxx \  # thay bằng checkpoint của Step1
        --per_device_eval_batch_size 2 \
        --run_name docTquery-LEGAL-generation \
        --max_length 256 \
        --valid_file data/legal_data/ja_dsi_ds/legal_corpus.tsv \
        --output_dir temp \
        --dataloader_num_workers 10 \
        --report_to wandb \
        --logging_steps 100 \
        --num_return_sequences 10

## Step 3: Train DSI-QG with query-represented corpus
python3 run.py \
        --task "DSI" \
        --model_name "google/mt5-base" \
        --run_name "LEGAL-ja-mt5-base-DSI-QG" \
        --max_length 32 \
        --train_file data/xorqa_data/ja/xorqa_corpus.tsv.q10.docTquery \
        --valid_file data/xorqa_data/ja/xorqa_DSI_dev_data.json \
        --output_dir "models/LEGAL-ja-mt5-base-DSI-QG" \
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