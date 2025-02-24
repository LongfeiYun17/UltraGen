import os
import argparse
import re
import numpy as np
import pandas as pd
import json
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from torch.cuda.amp import autocast, GradScaler

import torch
from torch.nn.functional import cosine_similarity, normalize
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# add data_path
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/dolma.json")
parser.add_argument("--ckpt_path", type=str, default="checkpoints/contrastive_learning")
args = parser.parse_args()

# Initialize wandb
wandb.init(project="contrastive-learning", name="e5-large")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "intfloat/e5-large"
text_encoder = AutoModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_triplets(path):
    with open(path, "r") as f:
        data = json.load(f)
        
    # Define all possible triplet types with more negative samples
    triplet_types = [
        ('soft', 'hard', 'hard'),
        ('soft', 'hard', 'soft'),
        ('soft', 'soft', 'hard'), 
        ('soft', 'soft', 'soft'),
        ('hard', 'soft', 'hard'),
        ('hard', 'soft', 'soft'),
        ('hard', 'hard', 'soft'),
        ('hard', 'hard', 'hard')
    ]
    
    samples_per_type = 100000 // len(triplet_types)  # Equal distribution
    triplets = []
    
    for triplet_type in triplet_types:
        type_triplets = []
        pbar = tqdm(total=samples_per_type, desc=f"Generating {triplet_type} triplets")
        
        while len(type_triplets) < samples_per_type:
            d = np.random.choice(data)
            # Select 8 different negative samples
            other = np.random.choice([x for x in data if x != d], replace=False)
            
            hard_attrs = [attr['description'] for attr in d['hard_attributes']]
            soft_attrs = d['soft_attributes']
            
            # Skip if not enough attributes for positive pairs
            if (not hard_attrs and ('hard' in triplet_type[:2])) or \
               (not soft_attrs and ('soft' in triplet_type[:2])):
                continue
                
            # Select positive attributes based on triplet type
            attr1_pool = soft_attrs if triplet_type[0] == 'soft' else hard_attrs
            attr2_pool = soft_attrs if triplet_type[1] == 'soft' else hard_attrs
            
            if len(attr1_pool) == 0 or len(attr2_pool) == 0:
                continue
                
            attr1 = np.random.choice(attr1_pool)
            attr2 = np.random.choice([x for x in attr2_pool if x != attr1]) if attr1 in attr2_pool else np.random.choice(attr2_pool)
            
            # Generate 1 negative pairs for each positive pair
            other_hard = [attr['description'] for attr in other['hard_attributes']]
            other_soft = other['soft_attributes']
                
            neg_pool = other_soft if triplet_type[2] == 'soft' else other_hard
            
            if len(neg_pool) == 0:
                continue
                
            neg_attr = np.random.choice(neg_pool)
            type_triplets.append((attr1, attr2, neg_attr))
                
            pbar.update(1)
        
        pbar.close()
        triplets.extend(type_triplets)
    
    # Split into train and validation sets (80-20 split)
    total_samples = len(triplets)
    train_size = int(0.8 * total_samples)
    
    # Create indices and shuffle them
    np.random.shuffle(triplets)
    train_triplets = triplets[:train_size]
    valid_triplets = triplets[train_size:]
    
    return train_triplets, valid_triplets

train_triplets, valid_triplets = get_triplets(args.data_path)

bs = 16  # Doubled batch size for 2 GPUs
num_epoch = 3
ids = list(range(0, len(train_triplets), bs))

# Scale learning rate linearly with batch size
base_lr = 2e-5
lr = base_lr * (bs / 8)  # Linear scaling rule
optimizer = AdamW([p for module in [text_encoder] for p in module.parameters()], lr=2e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=len(ids)*num_epoch)
scaler = GradScaler()  # For mixed precision training

margin = 1.0  # Margin for triplet loss

def evaluate():
    correct = 0
    total = 0
    for pos_attr1, pos_attr2, neg_attr in tqdm(valid_triplets):
        with torch.no_grad(), autocast():
            _repr_pos1 = normalize(text_encoder(**tokenizer([pos_attr1], padding=True, return_tensors="pt", truncation=True).to(device))[-1])
            _repr_pos2 = normalize(text_encoder(**tokenizer([pos_attr2], padding=True, return_tensors="pt", truncation=True).to(device))[-1])
            _repr_neg = normalize(text_encoder(**tokenizer([neg_attr], padding=True, return_tensors="pt", truncation=True).to(device))[-1])
        
        pos_distance = ((_repr_pos1 - _repr_pos2) ** 2).sum(-1)
        neg_distance = ((_repr_pos1 - _repr_neg) ** 2).sum(-1)
        
        correct += (pos_distance < neg_distance).sum().item()
        total += 1
        
    accuracy = (correct / total) * 100
    print(f"Accuracy: {accuracy:.4}%")
            
    # Log validation metrics
    wandb.log({
        "val/accuracy": accuracy
    })
    return accuracy

#evaluate()

best_accuracy = 0
for epoch in range(num_epoch):
    np.random.shuffle(train_triplets)
    bar = tqdm(ids)
    for idx in bar:
        _triplets = train_triplets[idx:idx+bs]
        _texts = [t[0] for t in _triplets]
        _pos_attrs = [t[1] for t in _triplets]
        _neg_attrs = [t[2] for t in _triplets]
        
        # Use autocast for mixed precision training
        with autocast():
            _repr_text = normalize(text_encoder(**tokenizer(_texts, padding=True, return_tensors="pt", truncation=True).to(device))[-1])
            _repr_pos = normalize(text_encoder(**tokenizer(_pos_attrs, padding=True, return_tensors="pt", truncation=True).to(device))[-1])
            _repr_neg = normalize(text_encoder(**tokenizer(_neg_attrs, padding=True, return_tensors="pt", truncation=True).to(device))[-1])
            
            # Calculate distances
            pos_distance = ((_repr_text - _repr_pos) ** 2).sum(-1)
            neg_distance = ((_repr_text - _repr_neg) ** 2).sum(-1)
            
            # Triplet loss
            loss = torch.clamp(pos_distance - neg_distance + margin, min=0).mean()

        text_encoder.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        bar.set_description(f"#Loss: {loss:.4}")
        
        # Log training metrics
        wandb.log({
            "train/loss": loss.item(),
            "train/pos_distance": pos_distance.mean().item(),
            "train/neg_distance": neg_distance.mean().item(),
            "train/learning_rate": scheduler.get_last_lr()[0]
        })
     
    accuracy = evaluate()
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        # save best model
        ckpt_path = args.ckpt_path
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        torch.save(text_encoder.state_dict(), os.path.join(ckpt_path, "text_encoder_best.pth"))
        #torch.save(text_encoder.module.state_dict(), os.path.join(ckpt_path, "text_encoder_best.pth"))

# save final model
ckpt_path = args.ckpt_path
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
torch.save(text_encoder.state_dict(), os.path.join(ckpt_path, "text_encoder_final.pth"))

# Finish wandb run
wandb.finish()
