#!/bin/bash

# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=0

# Disable NCCL P2P and IB for compatibility with RTX 4000 series
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

## Step 3: Train DSI-QG with query-represented corpus
python3 run.py \
        --task "DSI" \
        --model_name "google/mt5-base" \
        --run_name "MSMARCO-20k-mt5-base-DSI" \
        --max_length 128 \
        --train_file data/msmarco_data/20k/train_retrieval_ms_marco.json \
        --valid_file data/msmarco_data/20k/validation_retrieval_ms_marco.json \
        --output_dir "models/MSMARCO-20k-mt5-base-DSI" \
        --learning_rate 0.0005 \
        --warmup_steps 10000 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
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