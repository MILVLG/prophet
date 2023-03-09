#!/bin/bash
# This script is used to extract image features.

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2;;
    --dataset)
      DATASET="$2"
      shift 2;;
    --clip)
      CLIP_MODEL="$2"
      shift 2;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

DATASET=${DATASET:-ok} # dataset name, one of ['ok', 'aok'], default 'ok'
GPU=${GPU:-0} # GPU id(s) you want to use, default '0'
CLIP_MODEL=${CLIP_MODEL:-RN50x64} # clip model name or path, default 'RN50x64'

# CUDA_VISIBLE_DEVICES=$GPU \
python tools/extract_img_feats.py \
    --dataset $DATASET --gpu $GPU \
    --clip_model $CLIP_MODEL