#!/bin/bash

# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=0

# Disable NCCL P2P and IB for compatibility with RTX 4000 series
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

python3 run.py \
        --task "docTquery" \
        --model_name "google/mt5-large" \
        --run_name "docTquery-MSMARCO" \
        --max_length 128 \
        --train_file data/msmarco_data/100k/msmarco_DSI_train_data.json \
        --valid_file data/msmarco_data/100k/msmarco_DSI_dev_data.json \
        --output_dir "models/msmarco_docTquery_mt5_large" \
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