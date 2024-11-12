from data import IndexingTrainDataset, GenerateDataset, IndexingCollator, QueryEvalCollator
from transformers import (
    T5Tokenizer,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    TrainingArguments,
    TrainerCallback,
    MT5Tokenizer,
    MT5TokenizerFast,
    MT5ForConditionalGeneration,
    HfArgumentParser,
    set_seed,
)
from trainer import DSITrainer, DocTqueryTrainer
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional
import json
from tqdm import tqdm
set_seed(313)


@dataclass
class RunArguments:
    model_name: str = field(default=None)
    model_path: Optional[str] = field(default=None)
    max_length: Optional[int] = field(default=32)
    id_max_length: Optional[int] = field(default=20)
    remove_prompt: Optional[bool] = field(default=False)
    train_file: str = field(default=None)
    valid_file: str = field(default=None)
    task: str = field(default=None,  metadata={"help": "DSI, docTquery, generation"})
    top_k: Optional[int] = field(default=10)
    num_return_sequences: Optional[int] = field(default=10)
    q_max_length: Optional[int] = field(default=32)


def make_compute_metrics(tokenizer, valid_ids):

    def compute_metrics(eval_preds):
        recall_ks = [1, 2, 3, 5, 7, 10, 20, 50, 100, 200, 500]
        mrr_us = [10, 100]
        map_us = [10, 100]
        
        # Kết quả
        recall_results = {f"Recall@{k}": 0 for k in recall_ks}
        mrr_results = {f"MRR@{u}": 0 for u in mrr_us}
        map_results = {f"MAP@{u}": 0 for u in map_us}
        
        total_queries = len(eval_preds.predictions)
        
        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
            # Decode predicted docids and label docid
            rank_list = tokenizer.batch_decode(beams, skip_special_tokens=True)
            label_id = tokenizer.decode(label, skip_special_tokens=True)
            
            # Lọc các docid không hợp lệ và trùng lặp
            filtered_rank_list = []
            for docid in rank_list:
                if docid not in filtered_rank_list and docid in valid_ids:
                    filtered_rank_list.append(docid)
            
            # Tính Recall@k
            for k in recall_ks:
                if label_id in filtered_rank_list[:k]:
                    recall_results[f"Recall@{k}"] += 1
            
            # Tính MRR@u
            for u in mrr_us:
                top_u_list = filtered_rank_list[:u]
                if label_id in top_u_list:
                    rank = top_u_list.index(label_id) + 1  # Vị trí (tính từ 1)
                    mrr_results[f"MRR@{u}"] += 1 / rank
            
            # Tính MAP@u
            for u in map_us:
                top_u_list = filtered_rank_list[:u]
                relevant_docs = [1 if docid == label_id else 0 for docid in top_u_list]
                if sum(relevant_docs) > 0:
                    precision_at_ranks = [
                        sum(relevant_docs[:i+1]) / (i+1) for i in range(len(relevant_docs)) if relevant_docs[i] == 1
                    ]
                    map_results[f"MAP@{u}"] += np.mean(precision_at_ranks)
        
        # Chuẩn hóa giá trị (tính trung bình trên tổng số truy vấn)
        for k in recall_ks:
            recall_results[f"Recall@{k}"] /= total_queries
        for u in mrr_us:
            mrr_results[f"MRR@{u}"] /= total_queries
        for u in map_us:
            map_results[f"MAP@{u}"] /= total_queries
        
        # Gộp kết quả
        return {**recall_results, **mrr_results, **map_results}

    # def compute_metrics(eval_preds):
    #     hit_at_1 = 0
    #     hit_at_10 = 0
    #     for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
    #         rank_list = tokenizer.batch_decode(beams,
    #                                            skip_special_tokens=True)
    #         label_id = tokenizer.decode(label, skip_special_tokens=True)
    #         # filter out duplicates and invalid docids
    #         filtered_rank_list = []
    #         for docid in rank_list:
    #             if docid not in filtered_rank_list and docid in valid_ids:
    #                 filtered_rank_list.append(docid)

    #         hits = np.where(np.array(filtered_rank_list)[:10] == label_id)[0]
    #         if len(hits) != 0:
    #             hit_at_10 += 1
    #             if hits[0] == 0:
    #                 hit_at_1 += 1
    #     return {"Hits@1": hit_at_1 / len(eval_preds.predictions), "Hits@10": hit_at_10 / len(eval_preds.predictions)}
    return compute_metrics


def main():

    parser = HfArgumentParser((TrainingArguments, RunArguments))
    training_args, run_args = parser.parse_args_into_dataclasses()

    # We use wandb logger: https://wandb.ai/site.
    if training_args.local_rank == 0:  # only on main process
        # Initialize wandb run
        wandb.login()
        wandb.init(project="DSI", name=training_args.run_name)

    if 'mt5' in run_args.model_name:
        tokenizer = MT5Tokenizer.from_pretrained(run_args.model_name, cache_dir='cache')
        fast_tokenizer = MT5TokenizerFast.from_pretrained(run_args.model_name, cache_dir='cache')
        if run_args.model_path:
            model = MT5ForConditionalGeneration.from_pretrained(run_args.model_path, cache_dir='cache')
        else:
            model = MT5ForConditionalGeneration.from_pretrained(run_args.model_name, cache_dir='cache')
    else:
        tokenizer = T5Tokenizer.from_pretrained(run_args.model_name, cache_dir='cache')
        fast_tokenizer = T5TokenizerFast.from_pretrained(run_args.model_name, cache_dir='cache')
        if run_args.model_path:
            model = T5ForConditionalGeneration.from_pretrained(run_args.model_path, cache_dir='cache')
        else:
            model = T5ForConditionalGeneration.from_pretrained(run_args.model_name, cache_dir='cache')

    if run_args.task == "docTquery":
        train_dataset = IndexingTrainDataset(path_to_data=run_args.train_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer)

        valid_dataset = IndexingTrainDataset(path_to_data=run_args.valid_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             remove_prompt=run_args.remove_prompt,
                                             tokenizer=tokenizer)
        trainer = DocTqueryTrainer(
            do_generation=False,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
        )
        trainer.train()

    elif run_args.task == "DSI":
        train_dataset = IndexingTrainDataset(path_to_data=run_args.train_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer)

        valid_dataset = IndexingTrainDataset(path_to_data=run_args.valid_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             remove_prompt=run_args.remove_prompt,
                                             tokenizer=tokenizer)
        ################################################################
        # docid generation constrain, we only generate integer docids.
        SPIECE_UNDERLINE = "▁"
        INT_TOKEN_IDS = []
        for token, id in tokenizer.get_vocab().items():
            if token[0] == SPIECE_UNDERLINE:
                if token[1:].isdigit():
                    INT_TOKEN_IDS.append(id)
            if token == SPIECE_UNDERLINE:
                INT_TOKEN_IDS.append(id)
            elif token.isdigit():
                INT_TOKEN_IDS.append(id)
        INT_TOKEN_IDS.append(tokenizer.eos_token_id)

        def restrict_decode_vocab(batch_idx, prefix_beam):
            return INT_TOKEN_IDS
        ################################################################

        trainer = DSITrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
            compute_metrics=make_compute_metrics(fast_tokenizer, train_dataset.valid_ids),
            restrict_decode_vocab=restrict_decode_vocab,
            id_max_length=run_args.id_max_length
        )
        trainer.train()

    elif run_args.task == 'generation':
        generate_dataset = GenerateDataset(path_to_data=run_args.valid_file,
                                           max_length=run_args.max_length,
                                           cache_dir='cache',
                                           tokenizer=tokenizer)

        trainer = DocTqueryTrainer(
            do_generation=True,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=QueryEvalCollator(
                tokenizer,
                padding='longest',
            ),
        )
        predict_results = trainer.predict(generate_dataset,
                                          top_k=run_args.top_k,
                                          num_return_sequences=run_args.num_return_sequences,
                                          max_length=run_args.q_max_length)
        with open(f"{run_args.valid_file}.q{run_args.num_return_sequences}.docTquery", 'w') as f:
            for batch_tokens, batch_ids in tqdm(zip(predict_results.predictions, predict_results.label_ids),
                                                desc="Writing file"):
                for tokens, docid in zip(batch_tokens, batch_ids):
                    query = fast_tokenizer.decode(tokens, skip_special_tokens=True)
                    jitem = json.dumps({'text_id': docid.item(), 'text': query})
                    f.write(jitem + '\n')

    else:
        raise NotImplementedError("--task should be in 'DSI' or 'docTquery' or 'generation'")


if __name__ == "__main__":
    main()

