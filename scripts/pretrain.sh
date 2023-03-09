#!/bin/bash
# This script is used to pretrain the MCAN model.

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2;;
    --task)
      TASK="$2"
      shift 2;;
    --version)
      VERSION="$2"
      shift 2;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

TASK=${TASK:-ok} # task name, one of ['ok', 'aok_val', 'aok_test'], default 'ok'
GPU=${GPU:-0} # GPU id(s) you want to use, default '0'
VERSION=${VERSION:-pretraining_okvqa} # version name, default 'pretraining_for_$TASK'

# CUDA_VISIBLE_DEVICES=$GPU \
python main.py \
    --task $TASK --run_mode pretrain\
    --cfg configs/pretrain.yml \
    --version $VERSION \
    --gpu $GPU --seed 99 --grad_accu 2