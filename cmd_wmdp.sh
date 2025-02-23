JOB_ID="YOUR_ID"


# MODEL="./paper_models/Mistral-7B-v0.1" # download from huggingface repo
# TOKENIZER="./paper_models/Mistral-7B-v0.1"

MODEL="./paper_models/Llama-3.1-8B" # download from huggingface repo
TOKENIZER="./paper_models/Llama-3.1-8B"

# # # # # 7 9 11 20 25 30 gated_npo_grad_diff gated_rmu
python -u grun_wmdp.py --model $MODEL --tokenizer $TOKENIZER --seed 42 --save_path $JOB_ID --epoch 4 --npo_coeff 100 --retain_weight 1 --forget_loss_type gated_rmu --layers 20 25 31 --gated --save_eval_steps 50 100 150 200 250 300 350 400 450 500 --gate_weight 100 --template llama3 --batch_size 16 --steering_coeff 300 --rmu_layer 31 --forget_corpora merge_forget_data_sum_gpt,cyber_forget_data_sum_gpt --retain_corpora merge_retain_data_sum_gpt

REFT_PATH="PATH"
TEST_FILE="./wmdp_data/bio_test.jsonl"
# TEST_FILE="./wmdp_data/cyber_test.jsonl"
# TEST_FILE="./wmdp_data/merged_mmlu_text.jsonl"
python grun_wdmp_eval_mcq.py --source_dir ${TEST_FILE} --job_id ${JOB_ID} --correct_key corrtect --incorrect_key in_correct_choices --model $MODEL --tokenizer $TOKENIZER --batch_size 1 --use_reft --reft_path $REFT_PATH --test_sample_number 20000 --quant no




# baseline is based on https://github.com/centerforaisafety/wmdp with the following cmd

# python3 -m rmu.unlearn --model_name meta-llama/Llama-3.1-8B --batch_size 4 --layer_ids 13,14,15 --layer_id 15 --max_num_batches 150 --layer_ids 13,14,15 --layer_id 15 --retain_corpora wikitext,wikitext --forget_corpora bio-forget-corpus,cyber-forget-corpus --steering_coeffs 30,30 --alpha 350,350 --lr 5e-5 --seed 42 --output_dir models/llama3_rmu_param18

# python3 -m rmu.unlearn --model_name mistralai/Mistral-7B-v0.1 --batch_size 4 --layer_ids 13,14,15 --layer_id 15 --max_num_batches 150 --retain_corpora wikitext,wikitext --forget_corpora bio-forget-corpus,cyber-forget-corpus --steering_coeffs 27,27 --alpha 1600,1600 --min_len 200 --lr 5e-5 --seed 42 --output_dir models/mistral_rmu_param19
