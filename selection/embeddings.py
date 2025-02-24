import os
import faiss
import numpy as np
import json
import torch
from transformers import AutoTokenizer, AutoModel
import re
from tqdm import tqdm
import shutil
import random
import argparse

def process_batch(batch_texts, device="cuda", tokenizer=None, model=None):
    encoded_input = tokenizer(
        batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**encoded_input)
        embeddings = torch.mean(model_output.last_hidden_state, dim=1)
    return embeddings.cpu().numpy()

def generate_and_save_embeddings(texts, output_dir, model_path, batch_size=128):
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large")
    model = AutoModel.from_pretrained("intfloat/e5-large").to(f"cuda:0")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    os.makedirs(os.path.join(output_dir, "chunks"), exist_ok=True)
    chunk_id = 0
    for i in tqdm(range(0, len(texts), batch_size), desc=f"GPU 0"):
        batch = texts[i : i + batch_size]
        embeddings = process_batch(batch, device=f"cuda:0", tokenizer=tokenizer, model=model)
        np.save(os.path.join(output_dir, "chunks", f"emb_{chunk_id}.npy"), embeddings)
        chunk_id += 1

def build_index_incrementally(output_dir, embedding_dim=1024):
    index = faiss.IndexFlatL2(embedding_dim)
    chunk_files = [f for f in os.listdir(os.path.join(output_dir, "chunks")) if f.endswith(".npy")]
    chunk_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    for chunk_file in tqdm(chunk_files, desc=f"Building Index"):
        chunk_path = os.path.join(output_dir, "chunks", chunk_file)
        embeddings = np.load(chunk_path).astype("float32")
        index.add(embeddings) 
    faiss.write_index(index, os.path.join(output_dir, "redpajama_embeddings.index"))
    shutil.rmtree(os.path.join(output_dir, "chunks"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/data/longfei/redpajama", help="Output directory for embeddings and index")
    parser.add_argument("--data_path", type=str, default="contrastive_learning/redpajama.json", help="Path to the input data file")
    parser.add_argument("--model_path", type=str, default="/data/longfei/ckpts/contrastive_learning/text_encoder_best.pth", help="Path to the model checkpoint")
    args = parser.parse_args()

    output_dir = args.output_dir
    model_path = args.model_path
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(os.path.join(output_dir, "chunks")):
        shutil.rmtree(os.path.join(output_dir, "chunks"))

    with open(args.data_path, "r") as f:
        all_data = json.load(f)
    
    data = []
    id_to_text = {}
    id_to_text_soft = {}
    current_id = 0
    freq_set = set()
    include_set = set()
    other_set = set()
    
    for item in tqdm(all_data, desc="Loading Data"):
        for attr in item["hard_attributes"]:
            if attr['instruction'] == 'include keywords':
                match = re.search(r"Include the keyword '(.+?)' in your response.", attr['description'])
                if match:
                    include_set.add(match.group(1))
            elif attr['instruction'] == 'keywords frequency':
                match = re.search(r"The word '(.+?)' should appear", attr['description'])
                if match:
                    freq_set.add(match.group(1))
            else:
                other_set.add(attr['description'])

    for item in include_set:
        if item in freq_set:
            freq_set.remove(item)

    for item in freq_set:
        freq = random.randint(2, 3)
        text = f"The word '{item}' should appear {freq} times in your response."
        data.append(text)
        id_to_text[str(current_id)] = text
        current_id += 1

    for item in include_set:
        text = f"Include the keyword '{item}' in your response."
        data.append(text)
        id_to_text[str(current_id)] = text
        current_id += 1

    for item in other_set:
        text = item
        data.append(text)
        id_to_text[str(current_id)] = text
        current_id += 1

    for item in tqdm(all_data, desc="Loading Data"):
        soft_attrs = item["soft_attributes"]
        for text in soft_attrs:
            data.append(text)
            id_to_text[str(current_id)] = text
            id_to_text_soft[str(current_id)] = text
            current_id += 1

    print(f'Total data: {len(data)}')
    with open(os.path.join(output_dir, "id_to_text.json"), "w") as f:
        json.dump(id_to_text, f)
    print(f'Total soft data: {len(id_to_text_soft)}')
    with open(os.path.join(output_dir, "id_to_text_soft.json"), "w") as f:
        json.dump(id_to_text_soft, f)

    generate_and_save_embeddings(data, output_dir, model_path)
    build_index_incrementally(output_dir)
