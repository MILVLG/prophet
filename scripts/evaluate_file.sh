#!/bin/bash
# This script is used to evaluate a result file.

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2;;
    --result_path)
      RESULT_PATH="$2"
      shift 2;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

TASK=${TASK:-ok} # task name, one of ['ok', 'aok_val', 'aok_test'], default 'ok'
RESULT_PATH=${RESULT_PATH:-"preds/prophet_611_okvqa.json"} # path to the result file, default is the result from our experiments

if [ $TASK == "ok" ]; then
  python -m evaluation.okvqa_evaluate --result_path $RESULT_PATH \
    --question_path 'datasets/okvqa/OpenEnded_mscoco_val2014_questions.json' \
    --annotation_path 'datasets/okvqa/mscoco_val2014_annotations.json'
elif [ $TASK == "aok_val" ]; then
  python -m evaluation.aokvqa_evaluate --result_path $RESULT_PATH \
    --dataset_path 'datasets/aokvqa/aokvqa_v1p0_val.json' \
    --direct_answer --multiple_choice
elif [ $TASK == "aok_test" ]; then
  echo "Please submit your result to the AOKVQA leaderboard."
else
  echo "Unknown task: $TASK"
  exit 1
fi