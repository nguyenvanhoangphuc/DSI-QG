#!/bin/bash

# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=1

# Disable NCCL P2P and IB for compatibility with RTX 4000 series
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

python3 run.py \
        --task generation \
        --model_name google/mt5-large \
        --model_name castorini/doc2query-t5-large-msmarco \
        --per_device_eval_batch_size 8 \
        --run_name docTquery-MSMARCO-generation \
        --max_length 256 \
        --valid_file data/msmarco_data/100k/msmarco_corpus.tsv \
        --output_dir temp \
        --dataloader_num_workers 10 \
        --report_to wandb \
        --logging_steps 100 \
        --num_return_sequences 10