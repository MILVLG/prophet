#!/bin/bash
# This script is used to generate heuristics from a finetuned model.

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
    --candidate_num)
      CANDIDATE_NUM="$2"
      shift 2;;
    --example_num)
      EXAMPLE_NUM="$2"
      shift 2;;
    --version)
      VERSION="$2"
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
GPU=${GPU:-3} # GPU id(s) you want to use, default '0'
CKPT_PATH=${CKPT_PATH:-"ckpts/mcan_ft_okvqa.pkl"} # path to the pretrained model, default is the result from our experiments
CANDIDATE_NUM=${CANDIDATE_NUM:-10} # number of candidates to be generated
EXAMPLE_NUM=${EXAMPLE_NUM:-100} # number of examples to be generated
VERSION=${VERSION:-"heuristics_okvqa"} # version name, default 'heuristics1_for_$TASK'
if [ $TASK == "science" ]; then
  CONFIG=${CONFIG:-configs/mplug/heuristics_mplug.yml}
elif [ $TASK == "text_val" ]; then
  CONFIG=${CONFIG:-configs/mplug/heuristics_mplug.yml}
elif [ $TASK == "text_test" ]; then
  CONFIG=${CONFIG:-configs/mplug/heuristics_mplug.yml}
else
  CONFIG=${CONFIG:-configs/mplug/heuristics_mplug.yml}
fi

# CUDA_VISIBLE_DEVICES=$GPU \
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=$GPU_NUMS \
    --nnodes=1 --master_port=3224 \
    --use_env main_mplug.py \
    --task $TASK \
    --run_mode heuristics_mplug \
    --version $VERSION \
    --cfg $CONFIG \
    --ckpt_path $CKPT_PATH \
    --candidate_num $CANDIDATE_NUM \
    --example_num $EXAMPLE_NUM \
    --gpu $GPU \
    --deepspeed_config configs/ds_config.json

python prophet/stage1/utils/sim.py