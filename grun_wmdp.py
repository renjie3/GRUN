import os

import torch, transformers, pyreft
from pyreft.dataset import get_tofu_dataset
import numpy as np
import random
import argparse
from demo_utils import CustomStepCallback
from wmdp_utils import get_data

parser = argparse.ArgumentParser(description="A simple calculator.")

# Add arguments
parser.add_argument("--epoch", type=int, default=10)
# parser.add_argument("--max_steps", type=int, default=-1)
parser.add_argument("--model", type=str, default="NousResearch/Llama-2-7b-chat-hf")
parser.add_argument("--tokenizer", type=str, default="NousResearch/Llama-2-7b-chat-hf")
parser.add_argument("--rank", type=int, default=4)
parser.add_argument("--retain_weight", type=float, default=1.0)
parser.add_argument("--npo_coeff", type=float, default=0.1375)
parser.add_argument("--gate_weight", type=float, default=0)
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=4e-3)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--forget_loss_type", type=str, default="grad_diff")
parser.add_argument("--gated", action="store_true", default=False, help="Enable gated mechanism (default: False)")
parser.add_argument("--warm_up_gate_step", type=int, default=-1)
parser.add_argument('--layers', type=int, nargs='+', default=[16, 30])
parser.add_argument('--save_eval_steps', type=int, nargs='+', default=[50, 100, 150, 200, 250, 400])
parser.add_argument("--tofu_split", type=str, default="forget05")
parser.add_argument("--template", type=str, default="llama2")
parser.add_argument("--save_path", type=str, default="local")
parser.add_argument("--reft_class", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--rmu_layer", type=int, default=12)
parser.add_argument(
        "--retain_corpora",
        type=str,
        default="wikitext,wikitext",
        help="comma-separated list of corpora to retain",
    )
parser.add_argument(
    "--forget_corpora",
    type=str,
    default="bio-forget-corpus,cyber-forget-corpus",
    help="comma-separated list of corpora to forget",
)
parser.add_argument("--steering_coeff", type=float, default=6.5)

# Parse the arguments
args = parser.parse_args()

args.retain_corpora = args.retain_corpora.split(",")
args.forget_corpora = args.forget_corpora.split(",")

if "gate" in args.forget_loss_type:
    if not args.gated:
        raise("Not set gate.")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

device = "cuda"

model_name_or_path = args.model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)

# import pdb ; pdb.set_trace()

# get tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    args.tokenizer, model_max_length=2048, 
    padding_side="right", use_fast=False)
# tokenizer.pad_token = tokenizer.eos_token

random_vector = torch.rand(1,1, model.config.hidden_size, dtype=model.dtype, device=model.device)
control_vec = random_vector / torch.norm(random_vector) * args.steering_coeff

if args.gated:
    if args.reft_class == "gated_linear":
        representations = [
            {"layer": l, "component": "block_output",
            "low_rank_dimension": args.rank,
            "intervention": pyreft.GatedLoreftIntervention_Linear(embed_dim=model.config.hidden_size,
            low_rank_dimension=args.rank)} for l in args.layers
        ]
    else:
        representations = [
            {"layer": l, "component": "block_output",
            "low_rank_dimension": args.rank,
            "intervention": pyreft.GatedLoreftIntervention(embed_dim=model.config.hidden_size,
            low_rank_dimension=args.rank)} for l in args.layers
        ]

else:
    representations = [
        {"layer": l, "component": "block_output",
        "low_rank_dimension": args.rank,
        "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size,
        low_rank_dimension=args.rank)} for l in args.layers
    ]


# get reft model
reft_config = pyreft.ReftConfig(representations=representations)
reft_model = pyreft.get_reft_model(model, reft_config)
reft_model.set_device("cuda")
reft_model.print_trainable_parameters()

forget_data_list, retain_data_list = get_data(
    args.forget_corpora,
    args.retain_corpora,
    200,
    2000,
    4,
    # mmlu_subset=args.mmlu_subset,
    )

data_module = pyreft.make_last_position_supervised_wmdp_data_module(tokenizer, model, forget_data_list, retain_data_list, num_interventions=len(representations), forget_loss_type=args.forget_loss_type, template=args.template)

layer_str = '_'.join([str(l) for l in args.layers])
file_name = f"{args.save_path}_rank{args.rank}_layer{layer_str}_epoch{args.epoch}"
model_save_path = f"./results/{file_name}"

save_eval_steps = args.save_eval_steps
# train
training_args = transformers.TrainingArguments(
    num_train_epochs=args.epoch, 
    output_dir=model_save_path, 
    per_device_train_batch_size=args.batch_size, 
    learning_rate=args.lr, 
    logging_steps=10, 
    report_to=[], 
    # evaluation_strategy="steps",
    # eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    )
trainer = pyreft.WMDPReftTrainerForCausalLM(
    # callbacks=[CustomStepCallback(save_eval_steps, [])],
    model=reft_model, tokenizer=tokenizer, args=training_args, retain_weight=args.retain_weight, forget_loss_type=args.forget_loss_type, npo_coeff=args.npo_coeff, gate_weight=args.gate_weight, warm_up_gate_step=args.warm_up_gate_step, beta=args.beta, control_vec=control_vec, rmu_layer=args.rmu_layer, **data_module)
trainer.num_layers = len(representations)
trainer.output_file_name = file_name
_ = trainer.train()

reft_model.set_device("cpu") # send back to cpu before saving.
reft_model.save(
    save_directory=model_save_path, 
    save_to_hf_hub=False, 
    # include_model=True,
    # hf_repo_name="your_reft_emoji_chat"
)

import shutil
def ensure_config_in_checkpoints(parent_dir, config_file_path):
    items = os.listdir(parent_dir)
    checkpoint_dirs = [os.path.join(parent_dir, item) for item in items if item.startswith("checkpoint-") and os.path.isdir(os.path.join(parent_dir, item))]

    for checkpoint_dir in checkpoint_dirs:
        config_file_target = os.path.join(checkpoint_dir, "intervenable_model/config.json")
        if not os.path.exists(config_file_target):
            try:
                shutil.copy(config_file_path, config_file_target)
            except Exception as e:
                print(f"Failed to copy config.json to {checkpoint_dir}: {e}")

ensure_config_in_checkpoints(model_save_path, f"{model_save_path}/config.json")
