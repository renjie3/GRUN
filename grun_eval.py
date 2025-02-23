import os

import torch, transformers, pyreft
from pyreft.dataset import get_tofu_dataset, get_tofu_eval_dataset
import numpy as np
import random
import argparse
from tqdm import tqdm
import json
from peft import PeftModel
from rouge_score import rouge_scorer

parser = argparse.ArgumentParser(description="A simple calculator.")

# Add arguments
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--model", type=str, default="NousResearch/Llama-2-7b-chat-hf")
parser.add_argument("--reft_path", type=str, default="./models/reft_to_share")
parser.add_argument("--tokenizer", type=str, default="/home/ubuntu/quic-efs/user/rrenjie/unlearning/Unlearn-Simple/TOFU/paper_models/1_final_ft_noLORA_5_epochs_inst_lr1e-05_llama3.1_full_seed42_1")
parser.add_argument("--json", type=str, default="forget05_perturbed")
parser.add_argument("--data_task", type=str, default="all")
parser.add_argument("--rank", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--template", type=str, default='llama2')
parser.add_argument("--job_id", type=str, default='local')
parser.add_argument("--base_or_reft", type=str, default='reft', choices=["base", "reft"])
parser.add_argument("--use_multiple_gate", action="store_true", default=False)
parser.add_argument("--use_lora", type=str, default="no", choices=["no", "lora"])
parser.add_argument('--lora_path', type=str, default=None)
parser.add_argument("--gate_number", type=int, default=1)
parser.add_argument('--layers', type=int, nargs='+', default=[20, 25, 31])
parser.add_argument('--multiple_reft_path', type=str, default=None)
parser.add_argument('--quant', type=str, default="no", choices=["no", "4bit", "8bit"])
parser.add_argument("--multi_gate_coeff", type=float, default=0.6)
parser.add_argument("--multi_gate_bias", type=float, default=0.6)
# parser.add_argument('--multiple_reft_path', type=int, nargs='+', default=[None])

# Parse the arguments
args = parser.parse_args()

if args.multiple_reft_path is not None:
    args.multiple_reft_path = args.multiple_reft_path.split(",")

if args.use_multiple_gate and len(args.multiple_reft_path) != args.gate_number:
    print(args.multiple_reft_path)
    print(args.gate_number)
    print(len(args.multiple_reft_path))
    raise("Wrong gate_number")

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss

def eval_rouge_recall(gen_outputs, ground_truths):
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_recall = []
    rougeL_recall = []
    for gen, gt in zip(gen_outputs, ground_truths):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall.append(rouge_scores['rouge1'].recall)
        rougeL_recall.append(rouge_scores['rougeL'].recall)


    return {'rouge1_recall': rouge1_recall, 'rougeL_recall': rougeL_recall}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_generation(inputs, reft_model):

    unit_locations = None
    if "gen_intervention_locations" in inputs:
        if inputs["gen_intervention_locations"].dim() == 3:
            unit_locations={"sources->base": (
                None,
                inputs["gen_intervention_locations"].permute(1, 0, 2).tolist()
            )}
        else:
            # this is dummy for lora only baseline
            unit_locations={"sources->base": (None, 0)}

    # import pdb ; pdb.set_trace()
    with torch.no_grad():
        base_response, reft_response = reft_model.generate(
            {
                "input_ids": inputs["gen_input_ids"],
                "attention_mask": inputs["gen_attention_mask"],
            },
            unit_locations=unit_locations,
            intervene_on_prompt=True, max_new_tokens=256, do_sample=False, 
            eos_token_id=tokenizer.eos_token_id, early_stopping=True,
            output_original_output = True,
        )
    # decoded_response = tokenizer.decode_batch(reft_response, skip_special_tokens=True)
    decoded_batch = []
    if args.base_or_reft == "base":
        for _idx, tokens in enumerate(base_response):
            output_text = tokenizer.decode(tokens[inputs["gen_input_ids"].shape[-1]:], skip_special_tokens=True) 
            # output_text = tokenizer.decode(tokens, skip_special_tokens=True)
            decoded_batch.append(output_text)
    elif args.base_or_reft == "reft":
        for _idx, tokens in enumerate(reft_response):
            output_text = tokenizer.decode(tokens[inputs["gen_input_ids"].shape[-1]:], skip_special_tokens=True) 
            decoded_batch.append(output_text)
    else:
        raise("wrong!")

    return decoded_batch

def run_generation_lora(inputs, model):

    with torch.no_grad():
        base_response = model.generate(inputs["gen_input_ids"], attention_mask=inputs["gen_attention_mask"], max_new_tokens=256)
    decoded_batch = []
    for _idx, tokens in enumerate(base_response):
        output_text = tokenizer.decode(tokens[inputs["gen_input_ids"].shape[-1]:], skip_special_tokens=True) 
        decoded_batch.append(output_text)

    return decoded_batch

def get_forward(inputs, reft_model):
    unit_locations = None
    if "intervention_locations" in inputs:
        if inputs["intervention_locations"].dim() == 3:
            unit_locations={"sources->base": (
                None,
                inputs["intervention_locations"].permute(1, 0, 2).tolist()
            )}
        else:
            # this is dummy for lora only baseline
            unit_locations={"sources->base": (None, 0)}
    # import pdb ; pdb.set_trace()
    with torch.no_grad():
        base_outputs, cf_outputs = reft_model(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "output_hidden_states": True,
            },
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=inputs["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in inputs else None,
            output_original_output=True,
        )

    if isinstance(base_outputs, tuple):
        return base_outputs[0], base_outputs[1], cf_outputs
    else:
        return base_outputs, None, cf_outputs

set_seed(args.seed)

device = "cuda"

# if args.tokenizer in ["openai-community/gpt2-xl", "EleutherAI/gpt-neo-2.7B"]:
#     # prompt_no_input_template = """[INST] %s [/INST]"""
#     template = "llama3"
# else:
#     prompt_no_input_template = """[INST] %s [/INST]"""
#     template = "llama3"

template = args.template
model_name_or_path = args.model

# get tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, model_max_length=2048, padding_side="right", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
left_tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, model_max_length=2048, padding_side="left", use_fast=False)
left_tokenizer.pad_token = left_tokenizer.eos_token

if args.quant == "4bit":
    # import pdb ; pdb.set_trace()
    bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path, quantization_config=bnb_config, torch_dtype=torch.bfloat16, device_map=device)
elif args.quant == "8bit":
    bnb_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path, quantization_config=bnb_config, torch_dtype=torch.bfloat16, device_map=device)
elif args.quant == "no":
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)
else:
    raise("quant.")

if args.use_multiple_gate:
    representations = [ # GatedLoreftIntervention_MultipleGate
        {"layer": l, "component": "block_output",
        "low_rank_dimension": args.rank,
        "intervention": pyreft.GatedLoreftIntervention_MultipleGate(embed_dim=model.config.hidden_size, 
        gate_number=args.gate_number,
        multi_gate_coeff=args.multi_gate_coeff,
        multi_gate_bias=args.multi_gate_bias,
        low_rank_dimension=args.rank)} for l in args.layers
    ]
    reft_config = pyreft.ReftConfig(representations=representations)
    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.set_device("cuda")
    reft_model.print_trainable_parameters()

    file_names = []
    for i in range(len(args.layers)):
        file_names.append(f"layer.{args.layers[i]}.comp.block_output.unit.pos.nunit.1#0")

    for i in range(args.gate_number):
        for file_name in file_names:
            state_dict = torch.load(f"{args.multiple_reft_path[i]}/intkey_{file_name}.bin")
            reft_model.interventions[file_name][0]

            # rotate_layer_state = {'weight': state_dict["rotate_layer"]}
            # reft_model.interventions[file_name][0].rotate_layer[i].load_state_dict(rotate_layer_state)

            # We have to recreate a layer, and load back the columns.
            overload_w = state_dict["rotate_layer"].to("cuda")
            overload_w_width = overload_w.shape[-1]
            rotate_layer = pyreft.LowRankRotateLayer(
                model.config.hidden_size, overload_w_width, init_orth=True).to("cuda")
            reft_model.interventions[file_name][0].rotate_layer[i] = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
            reft_model.interventions[file_name][0].rotate_layer[i].parametrizations.weight[0].base[:,:overload_w_width] = overload_w

            # print(reft_model.interventions[file_name][0].rotate_layer[i].weight)

            gate_func_state = {k.split("gate_func.")[1]: v for k, v in state_dict.items() if k.startswith("gate_func.")}
            reft_model.interventions[file_name][0].gate_func[i].load_state_dict(gate_func_state, strict=False)
            reft_model.interventions[file_name][0].gate_func[i].to("cuda")
            reft_model.interventions[file_name][0].learned_source[i].load_state_dict(state_dict, strict=False)
            reft_model.interventions[file_name][0].learned_source[i].to("cuda")

    reft_model.set_device("cuda")

elif args.use_lora == "lora":
    #now use the checkpoint to add the LoRA modules
    model = PeftModel.from_pretrained(model, model_id = args.lora_path).to(device)
    #save this as a standard model so that we can again do PEFT style finetuneing from scratch
    model = model.merge_and_unload()

    representations = [None, None]

else:
    reft_model = pyreft.ReftModel.load(
        args.reft_path, model
    )
    reft_model.set_device(device, set_model=False)

    layers = []
    with open(f"{args.reft_path}/config.json", 'r') as file:
        data = json.load(file)
    for item in data['representations']:
        layers.append(item[0])
    representations = [
        {"layer": l, "component": "block_output",
        "low_rank_dimension": args.rank,
        "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size,
        low_rank_dimension=args.rank)} for l in layers
    ]
print("check2")

if args.json == "forget05_para_perturbed":
    all_files = [args.json]
else:
    all_files = [args.json, "retain_perturbed", "real_authors_perturbed", "world_facts_perturbed"]

for json_file in all_files:

    if args.data_task == "all":
        data_task_list = ["normal", "base", "perturb"]
    elif args.data_task == "normal":
        data_task_list = ["normal"]
    else:
        raise("data_task wrong")

    for data_task in data_task_list:

        flag_gen = False
        if data_task == "normal":
            ans_key = 'answer'
            flag_gen = True
        elif data_task == "base":
            ans_key = 'paraphrased_answer'
            if json_file == "real_authors_perturbed" or json_file == "world_facts_perturbed":
                ans_key = 'answer'
        elif data_task == "perturb":
            ans_key = 'perturbed_answer'

        data = get_tofu_eval_dataset("./TOFU_data", split=json_file)

        dataloader = pyreft.make_last_position_supervised_tofu_eval_dataloader(tokenizer, left_tokenizer, model, data, num_interventions=len(representations), batchsize=args.batch_size, qs_key='question', ans_key=ans_key, template=template)

        ave_gt_log = {}
        rouge_log = {}
        gen_string_log = {}
        gate_log = {}
        for inputs in tqdm(dataloader):

            inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}

            if args.use_lora == "no":
                base_output, gate_output, cf_outputs = get_forward(inputs, reft_model)
            else:
                with torch.no_grad():
                    base_output = model(inputs["input_ids"], labels=inputs["labels"], attention_mask=inputs["attention_mask"])
            if args.base_or_reft == "base":
                gt_loss = get_batch_loss(base_output.logits, inputs["labels"])
            elif args.base_or_reft == "reft":
                gt_loss = get_batch_loss(cf_outputs.logits, inputs["labels"])

            if flag_gen:
                if args.use_lora == "lora":
                    gen_output_string = run_generation_lora(inputs, model)
                else:
                    gen_output_string = run_generation(inputs, reft_model)
                rouge_results = eval_rouge_recall(gen_output_string, inputs['gt_output_string'])['rougeL_recall']

            # import pdb ; pdb.set_trace()

            for i in range(len(inputs["labels"])):
                num_token_gt = (inputs['labels'][i]!=-100).sum(-1).item()
                idx = inputs["idx"][i]
                if idx in ave_gt_log:
                    ave_gt_log[idx].append(gt_loss[i].item() / num_token_gt)
                else:
                    ave_gt_log[idx] = [gt_loss[i].item() / num_token_gt]

                if args.use_lora == "no" and gate_output is not None and data_task == "normal":

                    if idx in gate_log:
                        gate_log_item = []
                        for gate_i in gate_output:
                            gate_log_item.append(gate_i[i].item())
                        gate_log[idx].append(gate_log_item)
                    else:
                        gate_log_item = []
                        for gate_i in gate_output:
                            gate_log_item.append(gate_i[i].item())
                        gate_log[idx] = [gate_log_item]

                if flag_gen:
                    if idx in rouge_log:
                        rouge_log[idx].append(rouge_results[i])
                        output_string = {"gen": gen_output_string[i],
                                        "gt": inputs['gt_output_string'][i]}
                        gen_string_log[idx].append(output_string)
                    else:
                        rouge_log[idx] = [rouge_results[i]]
                        output_string = {"gen": gen_output_string[i],
                                        "gt": inputs['gt_output_string'][i]}
                        gen_string_log[idx] = [output_string]

                # import pdb ; pdb.set_trace()

        save_folder = f"./results/result_logs/{args.job_id}"

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        file_path = f"{save_folder}/{json_file}_{data_task}.json"
        with open(file_path, "w") as save_file:
            json.dump(ave_gt_log, save_file, indent=4)

        if args.use_lora == "no" and gate_output is not None and data_task == "normal":
            file_path = f"{save_folder}/{json_file}_gate.json"
            with open(file_path, "w") as save_file:
                json.dump(gate_log, save_file, indent=4)

        if flag_gen:
            file_path = f"{save_folder}/{json_file}_rouge.json"
            with open(file_path, "w") as save_file:
                json.dump(rouge_log, save_file, indent=4)

            file_path = f"{save_folder}/{json_file}_gen_string.json"
            with open(file_path, "w") as save_file:
                json.dump(gen_string_log, save_file, indent=4)


