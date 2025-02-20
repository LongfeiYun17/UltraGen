import os
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path    

# CLI arguments
parser = argparse.ArgumentParser(description='Sample data from FineWeb dataset')
parser.add_argument('--dataset_name', type=str, default="CC-MAIN-2024-10",
                    help='Dataset name (e.g. CC-MAIN-2024-10 or sample-10BT)')
parser.add_argument('--num_samples', type=int, default=10000,
                    help='Number of samples to extract')
parser.add_argument('--output_dir', type=str, default="data",
                    help='Output directory path')

args = parser.parse_args()

# Load dataset
en = load_dataset("HuggingFaceFW/fineweb", 
                 name=args.dataset_name, 
                 split="train", 
                 streaming=True)

# Sample data
data = []
for i, example in tqdm(enumerate(en), total=args.num_samples):
    data.append({
        'id': i,
        'text': example['text']
    })
    if i == args.num_samples - 1:
        break

# Save results
output_file = Path(args.output_dir) / "fineweb.json"
if not output_file.exists():
    output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)