master_port=16791

torchrun --nproc_per_node=2 --master_port=$master_port finetune.py --config-name=finetune.yaml model_family=mistral gradient_accumulation_steps=4 batch_size=4 num_epochs=8

torchrun --nproc_per_node=2 --master_port=$master_port finetune.py --config-name=finetune.yaml model_family=llama3.1 gradient_accumulation_steps=4 batch_size=4 num_epochs=5

# Total batch size = nproc_per_node * gradient_accumulation_steps * batch_size = 32 by default
