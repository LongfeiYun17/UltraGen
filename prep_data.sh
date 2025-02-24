#! /bin/bash

# Function to create directory if it doesn't exist
create_dir() {
    [ ! -d "$1" ] && mkdir -p "$1"
}

# Reconstruct Data
python src/reconstruct_data.py --dataset_name CC-MAIN-2024-10 --num_samples 10000 --output_dir data

python src/add_hard.py --input_file data/fineweb.json --output_file data/fineweb.json

python src/decompose.py --data_file data/fineweb.json --output_dir data --openai_api_key <YOUR_OPENAI_API_KEY>

# Global Data
DATA_DIR=<YOUR_DATA_DIR>
create_dir "${DATA_DIR}"

PARALLEL_DOWNLOADS=10
DOLMA_VERSION="v1_7"

DOMAINS=(
    "c4-filtered"
    "books"
    "cc_news_middlcc_news_middlee" 
    "redpajama-arxiv"
    "cc_en_middle"
    "reddit"
    "tulu_flan"
    "redpajama-stackexchange"
    "wiki"
    "falcon-refinedweb-filtered"
    "pes2o"
)

# Create directories for each domain
for DOMAIN in "${DOMAINS[@]}"; do
    create_dir "${DATA_DIR}/${DOMAIN}"
done

xargs -n 1 -P "${PARALLEL_DOWNLOADS}" wget -q -P "$DATA_DIR" < "${DOLMA_VERSION}.txt"

python src/dolma.py --num_samples 10000 --data_dir "${DATA_DIR}" --output_dir "data" --output_file "data/dolma.json"

python src/add_hard.py --input_file "data/dolma.json" --output_file "data/dolma.json"

python src/decompose.py --data_file "data/dolma.json" --output_dir "data" --openai_api_key <YOUR_OPENAI_API_KEY>

# Build global data
train_contrastive_learning=true
if [ "$train_contrastive_learning" = true ]; then
    python selection/embeddings.py \
        --output_dir "data" \
        --data_path "data/dolma.json" \
        --model_path "data/text_encoder_best.pth"
    
    python selection/embeddings.py \
        --output_dir "data" \
        --data_path "data/dolma.json" \
        --model_path "data/text_encoder_best.pth"

    python selection/select.py \
        --index_path "data/redpajama_embeddings.index" \
        --mapping_path "data/id_to_text.json" \
        --mapping_path_soft "data/id_to_text_soft.json" \

    python selection/solve_conflict.py --input_path "data/attribute_sets_random.json" --output_train_path "data/attribute_sets_train_random.json" --output_valid_path "data/attribute_sets_valid_random.json"
fi