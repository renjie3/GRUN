JOB_ID="YOUR_ID"

# llama3.1
MODEL="./paper_models/2_final_ft_noLORA_5_epochs_inst_lr1e-05_mistral_full_seed42_1/checkpoint-1000"
TOKENIZER="./paper_models/2_final_ft_noLORA_5_epochs_inst_lr1e-05_mistral_full_seed42_1"
GATE_FUNC="linear"

# mistral
# MODEL="./paper_models/2_final_ft_noLORA_5_epochs_inst_lr1e-05_llama3.1_full_seed42_1/checkpoint-625"
# TOKENIZER="./paper_models/2_final_ft_noLORA_5_epochs_inst_lr1e-05_llama3.1_full_seed42_1"
# GATE_FUNC="mlp_3"

# # losstype: gated_grad_diff gated_idk gated_npo_grad_diff
MY_CMD="python -u grun.py --model $MODEL --tokenizer $TOKENIZER --seed 42 --save_path $JOB_ID --epoch 40 --tofu_split forget05 --npo_coeff 100 --retain_weight 2 --forget_loss_type gated_grad_diff --layers 20 25 31 --gated --save_eval_steps 50 100 150 200 250 300 350 400 450 500 600 700 800 900 --gate_weight 10 --template llama3 --grad_diff_forget_weight 1 --gate_func $GATE_FUNC"

MY_CMD="python -u grun.py --model $MODEL --tokenizer $TOKENIZER --seed 42 --save_path $JOB_ID --epoch 40 --tofu_split forget05 --npo_coeff 100 --retain_weight 1 --forget_loss_type gated_idk --layers 20 25 31 --gated --save_eval_steps 50 100 150 200 250 300 350 400 450 500 600 700 800 900 --gate_weight 10 --template llama3 --grad_diff_forget_weight 1 --gate_func $GATE_FUNC"

MY_CMD="python -u grun.py --model $MODEL --tokenizer $TOKENIZER --seed 42 --save_path $JOB_ID --epoch 40 --tofu_split forget05 --npo_coeff 100 --retain_weight 1 --forget_loss_type gated_npo_grad_diff --layers 20 25 31 --gated --save_eval_steps 50 100 150 200 250 300 350 400 450 500 600 700 800 900 --gate_weight 10 --template llama3 --grad_diff_forget_weight 1 --gate_func $GATE_FUNC"
