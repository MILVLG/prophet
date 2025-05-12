#!/bin/bash
# This script is used to finetune the pretrained MCAN model.

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2;;
    --task)
      TASK="$2"
      shift 2;;
    --pretrained_model)
      PRETRAINED_MODEL_PATH="$2"
      shift 2;;
    --version)
      VERSION="$2"
      shift 2;;
    --run_mode)
      RUN_MODE="$2"
      shift 2;;
    --gpu_nums)
      GPU_NUMS="$2"
      shift 2;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

TASK=${TASK:-ok} # task name, one of ['ok', 'aok_val', 'aok_test','text_val','text_test','science'], default 'ok'
GPU=${GPU:-1} # GPU id(s) you want to use, default '0'
GPU_NUMS=${GPU_NUMS:-1}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-"ckpts/mcan_pt_okvqa.pkl"} # path to the pretrained model, default is the result from our experiments
VERSION=${VERSION:-finetuning_okvqa} # version name, default 'finetuning_for_$TASK'
RUN_MODE=${RUN_MODE:-finetune_mplug} #run mode, one of ['finetune_mplug','finetune_mplug_test'],default 'finetune'

# run python script
CONFIG=${CONFIG:-configs/mplug/finetune_mplug.yml}
#CUDA_VISIBLE_DEVICES=$GPU
python -m torch.distributed.launch --nproc_per_node=$GPU_NUMS \
    --nnodes=1 --master_port=3224 \
    --use_env main_mplug.py \
    --cfg $CONFIG \
    --task $TASK \
    --run_mode $RUN_MODE \
    --version $VERSION \
    --pretrained_model $PRETRAINED_MODEL_PATH \
    --gpu $GPU \
    --seed 42 \
    --grad_accu 2\
    --deepspeed_config configs/ds_config.json\
    --mplug
