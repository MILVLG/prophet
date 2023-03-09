#!/bin/bash
# This script is used to evaluate a finetuned model.

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2;;
    --task)
      TASK="$2"
      shift 2;;
    --ckpt_path)
      CKPT_PATH="$2"
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
CKPT_PATH=${CKPT_PATH:-"ckpts/mcan_ft_okvqa.pkl"} # path to the pretrained model, default is the result from our experiments
VERSION=${VERSION:-"eval_finetuned_${TASK}_model"} # version name, default 'eval_finetuned_$TASK_model'

# CUDA_VISIBLE_DEVICES=$GPU \
python main.py \
    --task $TASK --run_mode finetune_test \
    --cfg configs/finetune.yml \
    --version $VERSION \
    --ckpt_path $CKPT_PATH \
    --gpu $GPU --grad_accu 2
