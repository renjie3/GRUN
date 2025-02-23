import os

import argparse
# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, default="/home/ubuntu/quic-efs/user/rrenjie/unlearning/FailureLLMUnlearning/data/news/knowmem/forget_qa_mc.json")
parser.add_argument("--target_dir", type=str, default="/home/ubuntu/quic-efs/user/rrenjie/unlearning/FailureLLMUnlearning/results/zephyr_news.pth")
parser.add_argument("--model", type=str, default='HuggingFaceH4/zephyr-7b-beta') 
parser.add_argument("--tokenizer", type=str, default='HuggingFaceH4/zephyr-7b-beta') 
parser.add_argument("--correct_key", type=str, default="answer")
parser.add_argument("--incorrect_key", type=str, default="incorrect_answers")
parser.add_argument("--job_id", type=str, default="local")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--test_sample_number", type=int, default=1000)
parser.add_argument("--use_reft", action="store_true", default=False)
parser.add_argument("--reft_path", type=str, default=None)
parser.add_argument("--quant", type=str, default=None)

args = parser.parse_args()

# if args.local != "":
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.local
# CUDA_VISIBLE_DEVICES

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
import json
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import time
import torch, transformers, pyreft
import numpy as np

device = "cuda"

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

model_name = args.model
tokenizer_name = args.tokenizer

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device)

if args.quant == "4bit":
    bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, torch_dtype=torch.bfloat16, device_map=device)
elif args.quant == "8bit":
    bnb_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, torch_dtype=torch.bfloat16, device_map=device)
elif args.quant == "no":
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device)
else:
    raise("quant.")

if args.use_reft:
    reft_model = pyreft.ReftModel.load(
        args.reft_path, model
    )
    reft_model.set_device(device, set_model=False)

# import pdb ; pdb.set_trace()

# model = AutoModelForCausalLM.from_pretrained(
#     model_name, torch_dtype=torch.bfloat16, device_map=device)

# Sample input dictionary
# data = {
#     "question": "What is the animal of China?",
#     "answer": "Panda",
#     "incorrect_answers": ["Dog", "Cat", "Horse"]
# }

# data = {
#     "question": "What is the color of sky?",
#     "answer": "Blue",
#     "incorrect_answers": ["Red", "Green", "Yellow"]
# }

def load_file(file_path):
    data = []
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # Load JSON file directly as a dictionary or list
    elif file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))  # Load each line as a dictionary and append to the list
    else:
        raise ValueError("Unsupported file format. Please provide a .json or .jsonl file.")
    
    return data

options = ["A", "B", "C", "D"]
option_ids = torch.tensor([tokenizer.encode(option, add_special_tokens=False)[0] for option in options], device = device)
# Define function to generate multi-choice questions and get predictions
def get_label_preds(data, correct_key, incorrect_key):
    question_text = data["question"]
    correct_answer = data[correct_key]
    incorrect_answers = data[incorrect_key]

    # Prepare answers in different orders
    answer_combinations = []
    labels = []
    all_preds = []
    last_tokens_representation = []
    prompt_list = []
    
    for _label_pos in range(1):
        # Create an answer list where the correct answer is at position label_pos
        label_pos = random.randint(0, 3)
        # label_pos = _label_pos
        answers = incorrect_answers[:]
        answers.insert(label_pos, correct_answer)  # Insert correct answer at the designated position
        
        # Format prompt for model input
        # prompt = f"{question_text}\nA. {answers[0]}\nB. {answers[1]}\nC. {answers[2]}\nD. {answers[3]}\nAnswer:"
        prompt = f"""The following is a multiple choice question with answers.
{question_text}
A. {answers[0]}
B. {answers[1]}
C. {answers[2]}
D. {answers[3]}
Answer: """


        # prompt = f"""{question_text}
        # A. {answers[0]}
        # B. {answers[1]}
        # C. {answers[2]}
        # D. {answers[3]}
        # Answer: """

        prompt_list.append(prompt)
        labels.append(label_pos)

    time0 = time.time()

    inputs = tokenizer(prompt_list, return_tensors="pt").to('cuda')
    input_ids = tokenizer(prompt_list, return_tensors="pt").input_ids.to('cuda')
    prompt_count = len(prompt_list)

    time1 = time.time()

    # import pdb ; pdb.set_trace()

    # Get model output
    with torch.no_grad():
        if not args.use_reft:
            outputs = model(input_ids)
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
        else:
            unit_locations={"sources->base": (
                None,
                [[[inputs["input_ids"].shape[1] - 1]]] * 5
            )}
            (base_outputs, gated_output), cf_outputs = reft_model(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "output_hidden_states": True,
                },
                unit_locations=unit_locations,
                # labels=inputs["labels"],
                subspaces=inputs["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in inputs else None,
                output_original_output=False,
            )

            logits = cf_outputs.logits
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)

            # print(gated_output)
            # import pdb ; pdb.set_trace()

    # hidden_state = outputs.hidden_states[-1]
    # hidden_state = torch.cat(outputs.hidden_states, dim=0)
    # last_token_rep = hidden_state[:, -1, :].cpu()

    time3 = time.time()

    option_probs = probs[:, option_ids]
    pred_indices = torch.argmax(option_probs, dim=1)
    all_preds = pred_indices.tolist()

    time4 = time.time()

    # print(time1 - time0)
    # print(time3 - time1)
    # print(time4 - time3)

    # import pdb ; pdb.set_trace()

    # print(gated_output)
    
    if args.use_reft:
        return labels, all_preds, torch.cat(gated_output).mean().item()
    else:
        return labels, all_preds, None

# input_file = "/home/ubuntu/quic-efs/user/rrenjie/unlearning/FailureLLMUnlearning/data/news/knowmem/forget_qa_mc.json"
input_file = args.source_dir
output_file = f"./results/wmdp/{args.job_id}/results.pth"
correct_key, incorrect_key = args.correct_key, args.incorrect_key
# input_file = "/home/ubuntu/quic-efs/user/rrenjie/unlearning/wmdp/data/merged_mmlu_text_500.json"
# correct_key, incorrect_key = "corrtect", "in_correct_choices"
# output_file = args.target_dir
# batch_size = args.batch_size
directory_path = f"./results/wmdp/{args.job_id}"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

test_data = load_file(input_file)[:args.test_sample_number]
random.shuffle(test_data)

# test_data_2d = [test_data[i:i + args.batch_size] for i in range(0, len(test_data), args.batch_size)]

test_data_labels = []
test_data_preds = []
test_data_last_tokens_reps = []
gated_output_list = []
for item in tqdm(test_data):

    # Get labels and predictions
    item_labels, item_preds, gated_output = get_label_preds(item, correct_key, incorrect_key)
    gated_output_list.append(gated_output)

    test_data_labels += item_labels
    test_data_preds += item_preds

y_true = test_data_labels
y_pred = test_data_preds

f1 = f1_score(y_true, y_pred, average='weighted')
accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')

print("ACC:")
print(accuracy)
if args.use_reft:
    print("Gate output:")
    print(np.mean(gated_output_list))

data_to_save = {
    "labels": torch.tensor(test_data_labels),               # convert list to tensor for saving
    "preds": torch.tensor(test_data_preds),                 # convert list to tensor for saving
    # "last_tokens_rep": test_data_last_tokens_reps  # list of tensors can be saved directly
}

torch.save(data_to_save, output_file)

# import pdb ; pdb.set_trace()
