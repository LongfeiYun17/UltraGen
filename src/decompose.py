import json
from pathlib import Path
from functools import partial
from multiprocessing import Pool, Manager
import argparse

from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset

# CLI arguments
parser = argparse.ArgumentParser(description='Configuration for decomposition')
parser.add_argument('--openai_api_key', type=str, required=True,
                    help='OpenAI API key')
parser.add_argument('--prompt_dir', type=str, default="prompt",
                    help='Directory containing prompt files')
parser.add_argument('--data_file', type=str, default="data/train_data.json",
                    help='Input data file path')
parser.add_argument('--output_dir', type=str, default="data",
                    help='Output directory path')
parser.add_argument('--model', type=str, default="gpt-4o",
                    help='Model to use (e.g. gpt-4-turbo)')
parser.add_argument('--decomposition_level', type=str, default="implicit_v3",
                    help='Decomposition level (sentence, phrase, word, etc)')

# Initialize
args = parser.parse_args()
client = OpenAI(api_key=args.openai_api_key)

# Prompt mapping
decomposition_prompts = {
    "implicit_v3": f"{args.prompt_dir}/decompose_attributes_implicit_v3.md",
}

# Load prompt and dataset
prompt_file = decomposition_prompts[args.decomposition_level]
with open(prompt_file, "r") as file:
    prompt = file.read()

dataset = load_dataset("json", data_files=args.data_file, split="train")

def process_item(obj, prompt, args, shared_prompt_tokens, shared_completion_tokens):
    """Process a single dataset item through the OpenAI API"""
    text_to_send = obj.get("text", None)
    if not text_to_send:
        print(f"Skipping object because it has no 'text' field.")
        return None
    
    retry_count, max_retries = 0, 3 
    while retry_count < max_retries:
        try:
            # Format prompt based on decomposition level
            format_prompt = prompt.format(text=text_to_send)
            
            # Make API call
            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": format_prompt}
                ]
            )
            
            # Track token usage
            token_usage = completion.usage
            shared_prompt_tokens.value += token_usage.prompt_tokens
            shared_completion_tokens.value += token_usage.completion_tokens
            
            # Process result
            result = completion.choices[0].message.content
            obj["soft_attributes"] = [line.strip() for line in result.strip().splitlines() if line.strip()]
            
            return {
                "text": obj["text"],
                "soft_attributes": obj["soft_attributes"],
                "hard_attributes": obj["hard_attributes"],
            }
        
        except Exception as e:
            print(f">>> Error processing object: {e}")
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Max retries reached for object: {obj['text']}")
                return None

# Set up multiprocessing
manager = Manager()
shared_prompt_tokens = manager.Value('i', 0)
shared_completion_tokens = manager.Value('i', 0)
pool = Pool(processes=16)

# Process dataset
process_func = partial(process_item, prompt=prompt, args=args,
                      shared_prompt_tokens=shared_prompt_tokens,
                      shared_completion_tokens=shared_completion_tokens)

results = []
for result in tqdm(pool.imap_unordered(process_func, dataset), total=len(dataset)):
    if result is not None:
        results.append(result)
        if len(results) % 1000 == 99:
            print(f'Prompt tokens used: {shared_prompt_tokens.value}')
            print(f'Completion tokens used: {shared_completion_tokens.value}')

pool.close()
pool.join()

# Save results
output_file = Path(args.data_file).parent / f"{Path(args.data_file).stem}.json"

output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

# Print cost summary for gpt-4o
price = 2.5 * shared_prompt_tokens.value / 1000000 + 10 * shared_completion_tokens.value / 1000000
print(f"Total price: ${price:.2f}")
