#! /bin/bash 
# Reconstruct Data
python src/reconstruct_data.py  \
    --dataset_name CC-MAIN-2024-10 \
    --num_samples 10000 \
    --output_dir data

python src/add_hard.py \
    --input_file data/fineweb.json \
    --output_file data/fineweb.json

python src/decompose.py \
    --data_file data/fineweb.json \
    --output_dir data \
    --openai_api_key <YOUR_OPENAI_API_KEY>
                   
# Global Data
DATA_DIR=<YOUR_DATA_DIR>
if [ ! -d "${DATA_DIR}" ]; then
    mkdir -p "${DATA_DIR}"
fi
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
    if [ ! -d "${DATA_DIR}/${DOMAIN}" ]; then
        mkdir -p "${DATA_DIR}/${DOMAIN}"
    fi
done

cat "${DOLMA_VERSION}.txt" | xargs -n 1 -P "${PARALLEL_DOWNLOADS}" wget -q -P "$DATA_DIR"

python src/dolma.py \
    --num_samples 10000 \
    --data_dir "${DATA_DIR}" \
    --output_dir "data" \
    --output_file "data/dolma.json"

python src/add_hard.py \
    --input_file "data/dolma.json" \
    --output_file "data/dolma.json"

python src/decompose.py \
    --data_file "data/dolma.json" \
    --output_dir "data" \
    --openai_api_key <YOUR_OPENAI_API_KEY>