JOB_ID="local"

MODEL="./paper_models/2_final_ft_noLORA_5_epochs_inst_lr1e-05_llama3.1_full_seed42_1/checkpoint-625"
TOKENIZER="./paper_models/2_final_ft_noLORA_5_epochs_inst_lr1e-05_llama3.1_full_seed42_1"
TEMPLATE="llama3"

# MODEL="./paper_models/2_final_ft_noLORA_5_epochs_inst_lr1e-05_mistral_full_seed42_1/checkpoint-1000"
# TOKENIZER="./paper_models/2_final_ft_noLORA_5_epochs_inst_lr1e-05_mistral_full_seed42_1"
# TEMPLATE="mistral"

GPU_ID="0"

REFT_PATH="REFT_PATH"
BATCH_SIZE=16
JSON_FILE="forget05_perturbed"
BASE_OR_REFT="base" # base for not use reft, reft for using reft
QUANT="no" # 8bit 4bit no
DATA_TASK="all"
USE_LORA="no"
LORA_PATH="None"

sh grun_eval_tool.sh $GPU_ID $REFT_PATH $MODEL $JSON_FILE $BATCH_SIZE $JOB_ID $BASE_OR_REFT $TOKENIZER $TEMPLATE $QUANT $DATA_TASK $USE_LORA $LORA_PATH

