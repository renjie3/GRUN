import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="A simple calculator.")

# Add arguments
parser.add_argument("--json", type=str, default="forget05_perturbed")
parser.add_argument("--job_id", type=str, default='local')

# Parse the arguments
args = parser.parse_args()

# List of file names
if args.json == "forget05_para_perturbed":
    file_names = [args.json]
else:
    file_names = [args.json, "retain_perturbed", "real_authors_perturbed", "world_facts_perturbed"]
# file_names = [args.json, 'retain_perturbed', 'real_authors_perturbed', 'world_facts_perturbed']

# Initialize an empty list to store the ROUGE values
rouge_list = []
prob_list = []
ratio_list = []
gate_list = []

# Iterate over the file names and load the JSON files
for file_name in file_names:
    with open(f"./results/result_logs/{args.job_id}/results_{file_name}.json", 'r') as file:
        data = json.load(file)
        # Extract the ROUGE value for each file and append it to the list
        rouge_list.append(f"{data[f'{file_name} ROUGE']:.4f}")
        prob_list.append(f"{data[f'{file_name} Probability']:.4f}")
        if args.json != "forget05_para_perturbed":
            ratio_list.append(f"{data[f'{file_name} Truth Ratio']:.4f}")
            gate_list.append(data[f"{file_name} Gate"])

print_rouge = '\t'.join(rouge_list)
print_prob = '\t'.join(prob_list)
print_ratio = '\t'.join(ratio_list)

print(print_rouge)
print(print_prob)
print(print_ratio)

for i in range(len(gate_list[0])):
    gate_print_list = []
    for item in gate_list:
        gate_print_list.append(f"{item[i]:.4f}")
    print('\t'.join(gate_print_list))
