import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="A simple calculator.")

# Add arguments
parser.add_argument("--json", type=str, default="forget05_perturbed")
parser.add_argument("--job_id", type=str, default='local')

# Parse the arguments
args = parser.parse_args()

normal_file = f"./results/result_logs/{args.job_id}/{args.json}_normal.json"
with open(normal_file, "r") as file:
    data = json.load(file)
num_data = len(data)
normal_list = []
for i in range(num_data):
    normal_list.extend(data[str(i)])

rouge_file = f"./results/result_logs/{args.job_id}/{args.json}_rouge.json"
with open(rouge_file, "r") as file:
    data = json.load(file)
rouge_list = []
for i in range(num_data):
    rouge_list.extend(data[str(i)])

base_file = f"./results/result_logs/{args.job_id}/{args.json}_base.json"
with open(base_file, "r") as file:
    data = json.load(file)
base_list = []
for i in range(num_data):
    base_list.extend(data[str(i)])

perturb_file = f"./results/result_logs/{args.job_id}/{args.json}_perturb.json"
with open(perturb_file, "r") as file:
    data = json.load(file)
perturb_list = []
for i in range(num_data):
    perturb_list.append(data[str(i)])

gate_file = f"./results/result_logs/{args.job_id}/{args.json}_gate.json"
with open(gate_file, "r") as file:
    data = json.load(file)
num_data = len(data)
gate_list = []
for i in range(num_data):
    gate_list.extend(data[str(i)])

# import pdb ; pdb.set_trace()

output_result = {}

no_ratio = False
if "forget" in args.json or "retain" in args.json:
    gt_probs = np.exp(-1 * np.array(normal_list))
    avg_gt_prob = np.mean(gt_probs)
else:
    if not no_ratio:
        avg_true_prob = np.exp(-1 * np.array(normal_list))
        avg_false_prob = np.exp(-1 * np.array(perturb_list))
        avg_all_prob = np.concatenate([np.expand_dims(avg_true_prob, axis=-1), avg_false_prob], axis=1).sum(-1)
        avg_gt_prob = np.mean(avg_true_prob/avg_all_prob)

avg_rouge = np.array(rouge_list).mean()
output_result[f'{args.json} ROUGE'] = avg_rouge
output_result[f'{args.json} Probability'] = avg_gt_prob

if not no_ratio:

    # avg_paraphrase_np_values = - np.array(base_list)
    # avg_perturbed_np_values = - np.array(perturb_list)
    # avg_perturbed_np_values = avg_perturbed_np_values.mean(axis=-1)

    # curr_stat_1 =  np.exp( avg_perturbed_np_values - avg_paraphrase_np_values)

    avg_paraphrase_np_values = np.exp(-1 * np.array(base_list))
    avg_perturbed_np_values = np.exp(-1 * np.array(perturb_list))
    avg_perturbed_np_values = avg_perturbed_np_values.mean(axis=-1)
    curr_stat_1 =  avg_perturbed_np_values / avg_paraphrase_np_values

    # output_result[f'{eval_task_dict[k]} paraphrased_over_perturbed'] = curr_stat_1
    # if 'forget' in args.json:
    #     paraphrased_perturb_ratio = np.mean(np.minimum(curr_stat_1, 1/curr_stat_1))
    #     # paraphrased_perturb_ratio = np.mean(curr_stat_1)
    # else:
    #     paraphrased_perturb_ratio = np.mean(np.maximum(0, 1 - curr_stat_1))
    paraphrased_perturb_ratio = np.mean(np.maximum(0, 1 - curr_stat_1))
    output_result[f'{args.json} Truth Ratio'] = paraphrased_perturb_ratio

gate_score = np.array(gate_list).mean(axis=0).tolist()
output_result[f'{args.json} Gate'] = gate_score

print(output_result)

file_path = f"./results/result_logs/{args.job_id}/results_{args.json}.json"
print(file_path)

# Save the dictionary to a JSON file
with open(file_path, "w") as json_file:
    json.dump(output_result, json_file, indent=4)