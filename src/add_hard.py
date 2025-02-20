import json
import re
from tqdm import tqdm
import spacy
import random
import argparse
from spacy.lang.en.stop_words import STOP_WORDS
from datasets import load_dataset
from multiprocessing import Pool
from functools import partial

# CLI arguments
parser = argparse.ArgumentParser(description='Add hard attributes to dataset')
parser.add_argument('--input_file', type=str, default="data/input.json",
                    help='Input data file path')
parser.add_argument('--output_file', type=str, default="data/output.json", 
                    help='Output file path')
parser.add_argument('--num_processes', type=int, default=10,
                    help='Number of processes to use')

args = parser.parse_args()

nlp = spacy.load("en_core_web_sm")

dataset = load_dataset("json", data_files=args.input_file, split="train")

def process_doc(doc, nlp=nlp):
    """Extract and add hard attributes to a document based on its text content.
    
    Args:
        doc (dict): Document containing 'text' field
        nlp: Spacy language model
        
    Returns:
        dict: Document with added 'hard_attributes' field
    """
    text = doc['text'].strip()

    # Extract keywords and their frequencies
    doc_nlp = nlp(text)
    keywords = {}
    for chunk in doc_nlp.noun_chunks:
        token_text = chunk.text.lower().strip()
        if (token_text not in STOP_WORDS
            and not re.search(r'\d', token_text)
            and not re.search(r'https?://\S+', token_text)
            and len(token_text) > 2
            and not re.search(r'[^a-zA-Z\s\']', token_text)):
            keywords[token_text] = text.lower().count(token_text)

    # Initialize hard attributes list
    doc['hard_attributes'] = []

    # Add keyword frequency constraints
    for token, freq in keywords.items():
        if freq > 1:
            doc['hard_attributes'].append({
                "type": "keyword",
                "instruction": "keywords frequency", 
                "description": f"The word '{token}' should appear {freq} times in your response."
            })
        else:
            doc['hard_attributes'].append({
                "type": "keyword",
                "instruction": "include keywords",
                "description": f"Include the keyword '{token}' in your response."
            })

    # Add length constraints
    word_count = len(text.split())
    doc['hard_attributes'].append({
        "type": "length",
        "instruction": "number of words",
        "description": f"The text should contain between {(word_count//100)*100} and {((word_count//100)+1)*100} words."
    })

    # Add sentence count constraint
    sentences = len(list(doc_nlp.sents))
    doc['hard_attributes'].append({
        "type": "length", 
        "instruction": "number of sentences",
        "description": f"The text should contain between {(sentences//10)*10} and {(sentences//10)*10 + 10} sentences."
    })

    # Add paragraph count constraint
    paragraphs = len(text.split('\n\n'))
    doc['hard_attributes'].append({
        "type": "length",
        "instruction": "number of paragraphs", 
        "description": f"The text should contain {paragraphs} paragraphs."
    })

    # Randomly add case constraints
    rand_num = random.random()
    if rand_num < 0.05:
        doc['hard_attributes'].append({
            "type": "case",
            "instruction": "all uppercase",
            "description": "Your response should be in all uppercase."
        })
    elif rand_num > 0.95:
        doc['hard_attributes'].append({
            "type": "case", 
            "instruction": "all lowercase",
            "description": "Your response should be in all lowercase."
        })

    # Add starting word constraint
    first_word = text.strip().split()[0].lower()
    if len(first_word) > 2 and not re.search(r'\d', first_word):
        doc['hard_attributes'].append({
            "type": "start with",
            "instruction": "start with",
            "description": f"Your response should start with the word '{first_word}'."
        })

    return doc

def main():
    # Process documents in parallel
    pool = Pool(processes=args.num_processes)
    
    processed_data = []
    for result in tqdm(pool.imap(process_doc, dataset), total=len(dataset)):
        processed_data.append(result)

    pool.close()
    pool.join()

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(processed_data, f, indent=4)

if __name__ == '__main__':
    main()
