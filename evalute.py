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

        hit_at_1 = 0
        hit_at_10 = 0
        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
            rank_list = tokenizer.batch_decode(beams,
                                               skip_special_tokens=True)
            label_id = tokenizer.decode(label, skip_special_tokens=True)
            # filter out duplicates and invalid docids
            filtered_rank_list = []
            for docid in rank_list:
                if docid not in filtered_rank_list and docid in valid_ids:
                    filtered_rank_list.append(docid)

            hits = np.where(np.array(filtered_rank_list)[:10] == label_id)[0]
            if len(hits) != 0:
                hit_at_10 += 1
                if hits[0] == 0:
                    hit_at_1 += 1
        
        # Gộp kết quả
        return {**recall_results, **mrr_results, **map_results, "Hits@1": hit_at_1 / len(eval_preds.predictions), "Hits@10": hit_at_10 / len(eval_preds.predictions)}


def compute_metrics(eval_preds):
        hit_at_1 = 0
        hit_at_10 = 0
        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
            rank_list = tokenizer.batch_decode(beams,
                                               skip_special_tokens=True)
            label_id = tokenizer.decode(label, skip_special_tokens=True)
            # filter out duplicates and invalid docids
            filtered_rank_list = []
            for docid in rank_list:
                if docid not in filtered_rank_list and docid in valid_ids:
                    filtered_rank_list.append(docid)

            hits = np.where(np.array(filtered_rank_list)[:10] == label_id)[0]
            if len(hits) != 0:
                hit_at_10 += 1
                if hits[0] == 0:
                    hit_at_1 += 1
        return {"Hits@1": hit_at_1 / len(eval_preds.predictions), "Hits@10": hit_at_10 / len(eval_preds.predictions)}