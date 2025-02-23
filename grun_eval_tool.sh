GPU_ID_SINGLE=$1
REFT_PATH=$2
MODEL=$3
JSON_FILE=$4
BATCH_SIZE=$5
JOB_ID=$6
BASE_OR_REFT=$7
TOKENIZER=$8
TEMPLATE=$9
QUANT=${10}
DATA_TASK=${11}
USE_LORA=${12}
LORA_PATH=${13}

CUDA_VISIBLE_DEVICES=$GPU_ID_SINGLE python -u grun_eval.py --reft_path $REFT_PATH --model $MODEL --json $JSON_FILE --batch_size $BATCH_SIZE --job_id $JOB_ID --base_or_reft $BASE_OR_REFT --tokenizer $TOKENIZER --template $TEMPLATE --quant $QUANT --data_task $DATA_TASK --use_lora $USE_LORA --lora_path $LORA_PATH

python collect_results/eval_collect_results.py --json $JSON_FILE --job_id $JOB_ID
python collect_results/eval_collect_results.py --json retain_perturbed --job_id $JOB_ID
python collect_results/eval_collect_results.py --json real_authors_perturbed --job_id $JOB_ID
python collect_results/eval_collect_results.py --json world_facts_perturbed --job_id $JOB_ID
python collect_results/eval_merge_read_results.py --json $JSON_FILE --job_id $JOB_ID > ./results/result_logs/${JOB_ID}/all_read_results.txt
