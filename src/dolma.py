import json
import os
from datasets import load_dataset
from tqdm import tqdm
import argparse
from pathlib import Path

# CLI arguments
parser = argparse.ArgumentParser(description='Sample data from Dolma dataset')
parser.add_argument('--num_samples', type=int, default=10_000,
                    help='Number of samples to extract')
parser.add_argument('--data_dir', type=str, default="/data/longfei/dolma",
                    help='Base directory containing Dolma datasets')
parser.add_argument('--output_dir', type=str, default="data",
                    help='Output directory for sampled data')
parser.add_argument('--output_file', type=str, default="data/dolma_new.json",
                    help='Output JSON file path')

args = parser.parse_args()

# Base path for the dataset
DATA_DIR = args.data_dir
DOMAINS = ["books", "c4-filtered", "cc_news_middle", "redpajama-arxiv", "cc_en_middle", "reddit",  "redpajama-stackexchange", "wiki", "falcon-refinedweb-filtered", "pes2o"]
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)
total_samples = args.num_samples

def truncate_text(example):
    words = example['text'].split()
    if len(words) > 512:
        example['text'] = ' '.join(words[:512])
    return example

# Function to sample 1000 examples per domain
def sample_from_domain(domain, num_samples=1000):
    # Load dataset
    dataset_path = os.path.join(DATA_DIR, domain)
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(
        dataset_path, 
        data_files="*.json.gz",
        split="train",
        streaming=True
    )
    data = []
    for sample in dataset:
        if len(sample['text']) < 128:
            continue

        sample = truncate_text(sample)
        data.append({
            'text': sample['text'],
            'domain': domain
        })
        if len(data) >= num_samples:
            break
    print(f"Sampled {len(data)} examples from {domain}.")
    return data

all_data = []
for domain in DOMAINS:
    all_data.extend(sample_from_domain(domain, total_samples // len(DOMAINS)))

print(f"All sampled datasets saved to {OUTPUT_DIR}.")

with open(args.output_file, 'w') as f:
    json.dump(all_data, f, indent=4)