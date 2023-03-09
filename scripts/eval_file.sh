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
  python scripts/okvqa_evaluate.py --result_path $RESULT_PATH
elif [ $TASK == "aok_val" ]; then
  python scripts/aokvqa_evaluate.py --result_path $RESULT_PATH
elif [ $TASK == "aok_test" ]; then
  echo "Please submit your result to the AOKVQA leaderboard."
else
  echo "Unknown task: $TASK"
  exit 1
fi