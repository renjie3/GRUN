master_port=16791
JOB_ID="YOUR_ID"

# llama3.1
MODEL_PATH="./paper_models/2_final_ft_noLORA_5_epochs_inst_lr1e-05_llama3.1_full_seed42_1/checkpoint-625"
MODEL_FAMILY="llama3.1"

# mistral
# MODEL_PATH="./paper_models/2_final_ft_noLORA_5_epochs_inst_lr1e-05_mistral_full_seed42_1/checkpoint-1000"
# MODEL_FAMILY="mistral"

# MY_CMD="torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget.yaml save_job_name=$JOB_ID forget_loss=grad_diff num_epochs=10 split=forget05 retain_set=retain95 gradient_accumulation_steps=16 npo_coeff=0.1375 beta=0.1 model_family=$MODEL_FAMILY model_path=$MODEL_PATH"
# MY_CMD="torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget.yaml save_job_name=$JOB_ID forget_loss=idk num_epochs=10 split=forget05 retain_set=retain95 gradient_accumulation_steps=16 npo_coeff=0.1375 beta=0.1 model_family=$MODEL_FAMILY model_path=$MODEL_PATH"
MY_CMD="torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget.yaml save_job_name=$JOB_ID forget_loss=npo_grad_diff num_epochs=10 split=forget10 retain_set=retain_perturbed gradient_accumulation_steps=16 batch_size=1 npo_coeff=0.1375 beta=0.1 grad_diff_coeff=1 model_family=$MODEL_FAMILY model_path=$MODEL_PATH"



# LoRA: loss_type manual_grad_diff idk manual_npo_grad_diff
# MY_CMD="torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget.yaml save_job_name=$JOB_ID forget_loss=manual_grad_diff num_epochs=10 split=forget05 retain_set=retain95 gradient_accumulation_steps=16 npo_coeff=0.1375 beta=0.1 model_family=$MODEL_FAMILY model_path=$MODEL_PATH LoRA.r=4 lr=1e-4 grad_diff_coeff=10000"
# MY_CMD="torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget.yaml save_job_name=$JOB_ID forget_loss=idk num_epochs=10 split=forget05 retain_set=retain95 gradient_accumulation_steps=16 npo_coeff=0.1375 beta=0.1 model_family=$MODEL_FAMILY model_path=$MODEL_PATH LoRA.r=4 lr=1e-4"
# MY_CMD="torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget.yaml save_job_name=$JOB_ID forget_loss=manual_npo_grad_diff num_epochs=10 split=forget05 retain_set=retain95 gradient_accumulation_steps=16 npo_coeff=0.1375 beta=0.1 model_family=$MODEL_FAMILY model_path=$MODEL_PATH LoRA.r=4 lr=1e-4"


