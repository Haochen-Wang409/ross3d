#!/bin/bash

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT="$1"
ANWSER_FILE="results/$1/scanrefer.jsonl"
OUTPUT_FILE="results/$1/scanrefer.csv"

mkdir -p "results/$1"
rm -fr $ANWSER_FILE

python3 ross3d/eval/model_scanrefer.py \
    --model-path $CKPT \
    --video-folder data \
    --embodiedscan-folder data/embodiedscan \
    --n_gpu 8 \
    --question-file data/processed/scanrefer/scanrefer_vg_val_llava_style.json \
    --conv-mode qwen_1_5 \
    --answer-file $ANWSER_FILE \
    --frame_sampling_strategy $2 \
    --max_frame_num $3 \
    --overwrite_cfg true

python ross3d/eval/eval_scanrefer.py --input-file $ANWSER_FILE --output-file $OUTPUT_FILE
