import faiss
import numpy as np
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import multiprocessing as mp
import argparse
from functools import partial
import random

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
parser.add_argument("--sampling", type=str, default="high_correlation_low_similarity", choices=["random", "high_correlation", "low_similarity", "high_correlation_low_similarity"],
                   help="Sampling strategy for attribute expansion")
parser.add_argument("--index_path", type=str, required=True, help="Path to the FAISS index file")
parser.add_argument("--mapping_path", type=str, required=True, help="Path to the ID to text mapping file")
parser.add_argument("--mapping_path_soft", type=str, required=True, help="Path to the soft ID to text mapping file")
parser.add_argument("--num_seeds", type=int, default=2000, help="Number of seed attributes to expand")
args = parser.parse_args()

DEBUG = args.debug
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

class AttributeExpander:
    def __init__(self, index_path, mapping_path, mapping_path_soft, k=1024):
        self.index = faiss.read_index(index_path)
        self.dimension = self.index.d
        self.total_attributes = self.index.ntotal 
        self.id_to_text = json.load(open(mapping_path, "r"))
        self.id_to_text_soft = json.load(open(mapping_path_soft, "r"))
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large")
        self.model = AutoModel.from_pretrained("intfloat/e5-large").to("cuda:0")
        self.soft_indices = set([int(id) for id in self.id_to_text_soft.keys()])
        self.hard_indices = set(range(self.total_attributes)) - self.soft_indices
        self.k = k

    def get_embedding(self, idx):
        return self.index.reconstruct(int(idx)).reshape(1, -1)

    def search_candidates(self, query_idx, k=1024):
        query_embedding = self.get_embedding(query_idx)
        distances, indices = self.index.search(query_embedding, k)
        return indices[0]
    
    def process_batch(self, batch_texts):
        encoded_input = self.tokenizer(
            batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        encoded_input = {k: v.to("cuda:0") for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = torch.mean(model_output.last_hidden_state, dim=1)
        return embeddings.cpu().numpy()

    def expand_attribute_set(self, seed_idx, min_size=10, max_size=50):
        target_size = np.random.randint(min_size, max_size + 1)
        target_soft_count = np.random.randint(1, min(target_size, 10))
        target_hard_count = target_size - target_soft_count

        candidates = self.search_candidates(seed_idx, self.k)
        # Original high correlation low similarity strategy
        candidates_texts = [self.id_to_text[str(i)] for i in candidates]
        semantic_embeddings = self.process_batch(candidates_texts)
        candidate_pos = {idx: i for i, idx in enumerate(candidates)}

        seed_pos = candidate_pos[seed_idx]
        current_emb = semantic_embeddings[seed_pos]

        mask = np.ones(len(candidates), dtype=bool)
        mask[seed_pos] = False

        candidates = candidates[mask]
        semantic_embeddings = semantic_embeddings[mask]

        distance = np.dot(semantic_embeddings, current_emb) / \
            (np.linalg.norm(semantic_embeddings, axis=1) * np.linalg.norm(current_emb))

        current_set = [seed_idx]
        current_soft_count = 1
        current_hard_count = 0
        # the first while loop is to ensure the target soft ratio
        while current_soft_count < target_soft_count or current_hard_count < target_hard_count:
            new_idx = np.argmin(distance)
            selected_attr = candidates[new_idx]
            
            # Check if adding this attribute maintains the target ratio
            is_soft = selected_attr in self.soft_indices
            if is_soft and current_soft_count < target_soft_count:
                current_set.append(selected_attr)
                current_soft_count += 1
                current_emb = semantic_embeddings[new_idx]
            elif not is_soft and current_hard_count < target_hard_count:
                current_set.append(selected_attr)
                current_hard_count += 1
                current_emb = semantic_embeddings[new_idx]
                
            mask = np.ones(len(candidates), dtype=bool)
            mask[new_idx] = False
            candidates = candidates[mask]
            semantic_embeddings = semantic_embeddings[mask]
            distance = distance[mask]
            if len(candidates) > 0:
                new_distances = np.dot(semantic_embeddings, current_emb) / \
                    (np.linalg.norm(semantic_embeddings, axis=1) * np.linalg.norm(current_emb))
                distance = np.minimum(distance, new_distances)
        return [self.id_to_text[str(i)] for i in current_set[:target_size]]

    def generate_all_sets(self, num_seeds=2000):
        np.random.seed(1234)
        soft_seeds = [int(id) for id in self.id_to_text_soft.keys()]

        seed_indices = np.random.choice(soft_seeds, num_seeds, replace=False)
        seed_indices = seed_indices.tolist()
        
        all_sets = []
        for seed in tqdm(seed_indices, desc="Processing seeds"):
            attribute_set = self.expand_attribute_set(seed)
            all_sets.append(attribute_set)

        return all_sets

if __name__ == "__main__":
    expander = AttributeExpander(args.index_path, args.mapping_path, args.mapping_path_soft, k=1024)
    num_seeds = args.num_seeds if not DEBUG else 400
    attribute_sets = expander.generate_all_sets(num_seeds)
    with open(f'contrastive_learning/attribute_sets_{args.sampling}.json', 'w') as f:
        json.dump(attribute_sets, f)
    print(f"Generated {len(attribute_sets)} attribute sets")