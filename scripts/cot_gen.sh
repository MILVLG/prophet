#!/bin/bash
# This script is used to prompt GPT-4o to generate chain-of-thought.

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2;;
    --version)
      VERSION="$2"
      shift 2;;
    --data_path)
      DATA_PATH="$2"
      shift 2;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

TASK=${TASK:-ok} # task name, one of ['ok', 'aok_val', 'aok_test','text_val','text_test','science_test'], default 'ok'
VERSION=${VERSION:-"cot_okvqa"} # version name, default 'cot_for_$TASK'
DATA_PATH=${DATA_PATH:-"assets/answer_aware_examples_okvqa.json"} # path to the examples, default is the result from our experiments

# CUDA_VISIBLE_DEVICES=$GPU \
python main_mplug.py \
    --task $TASK --run_mode cot_gen \
    --version $VERSION \
    --cfg configs/mplug/cot_gen.yml \