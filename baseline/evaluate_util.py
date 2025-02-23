from tqdm import tqdm
from data_module import TextDatasetQA, custom_data_collator, get_batch_loss
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig
import os, hydra
import evaluate
import json
from pathlib import Path
from rouge_score import rouge_scorer
from utils import get_model_identifiers_from_yaml
import torch.nn as nn
import numpy as np

def eval_perturbation_ratio(eval_dataloader, perturb_dataloader, model):
    eval_logs = {}
    for batch, perturb_batch in tqdm(zip(eval_dataloader, perturb_dataloader)):
        input_ids, labels, attention_mask = batch
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        perturb_input_ids, perturb_labels, perturb_attention_mask = perturb_batch
        if len(perturb_input_ids.shape) > 2:
            bsz, seq_len = perturb_input_ids.shape[0:2]
        else:
            bsz = perturb_input_ids.shape[0]
            seq_len = 1
        perturb_batch = {"input_ids": perturb_input_ids.view(bsz*seq_len, -1), "labels": perturb_labels.view(bsz*seq_len, -1), "attention_mask": perturb_attention_mask.view(bsz*seq_len, -1)}


        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)
        for k, v in perturb_batch.items():
            perturb_batch[k] = v.to(model.device)


        with torch.no_grad():
            outputs = model(**batch)
            perturb_outputs = model(**perturb_batch)

        gt_loss = get_batch_loss(outputs.logits, batch['labels'])
        perturb_loss = get_batch_loss(perturb_outputs.logits, perturb_batch['labels']).view(bsz, seq_len)

        num_token_gt = (batch['labels']!=-100).sum(-1)
        num_token_perturb = (perturb_batch['labels']!=-100).view(bsz, seq_len, -1).sum(-1)

        mean_perturb_loss = perturb_loss.mean(dim=1)

        ratio = (mean_perturb_loss - gt_loss).mean()

        
        # eval_logs["perplexity delta"] = eval_logs.get("perplexity delta", []) + [ratio.item()]

        # eval_logs['ground_truth_loss'] = eval_logs.get('ground_truth_loss', []) + [gt_loss.mean().item()]
        # eval_logs['perturb_loss'] = eval_logs.get('perturb_loss', []) + [mean_perturb_loss.mean().item()]

        eval_logs['average_perturb_loss'] = eval_logs.get('average_perturb_loss', []) + (perturb_loss/num_token_perturb).tolist()
        eval_logs['avg_paraphrased_loss'] = eval_logs.get('avg_paraphrased_loss', []) + (gt_loss/num_token_gt).cpu().numpy().tolist()

        eval_logs['paraphrased_loss'] = eval_logs.get('paraphrased_loss', []) + gt_loss.tolist()
        eval_logs['perturb_loss'] = eval_logs.get('perturb_loss', []) + perturb_loss.tolist()

        eval_logs['num_token_paraphrased'] = eval_logs.get('num_token_paraphrased', []) + num_token_gt.tolist()
        eval_logs['num_token_perturb'] = eval_logs.get('num_token_perturb', []) + num_token_perturb.tolist()

    return eval_logs

def get_dataloader(cfg, eval_task, tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key):

    torch_format_dataset = TextDatasetQA( 
            folder, 
            tokenizer=tokenizer, 
            model_family=cfg.model_family, 
            max_length=cfg.generation.max_length, 
            split=split, 
            question_key=question_key, 
            answer_key=answer_key
        ) 
    base_torch_format_dataset = TextDatasetQA(
        folder,
        tokenizer=tokenizer, 
        model_family=cfg.model_family, 
        max_length=cfg.generation.max_length, 
        split=split, 
        question_key=question_key, 
        answer_key=base_answer_key
    )

    perturb_torch_format_dataset = TextDatasetQA(
        folder,
        tokenizer=tokenizer, 
        model_family=cfg.model_family, 
        max_length=cfg.generation.max_length, 
        split=split, 
        question_key=question_key, 
        answer_key=perturbed_answer_key
    )

    if cfg.ds_size:
        torch_format_dataset.data = torch_format_dataset.data.select(range(min(cfg.ds_size, len(torch_format_dataset.data))))
        base_torch_format_dataset.data = base_torch_format_dataset.data.select(range(min(cfg.ds_size, len(base_torch_format_dataset.data))))
        perturb_torch_format_dataset.data = perturb_torch_format_dataset.data.select(range(min(cfg.ds_size, len(perturb_torch_format_dataset.data))))

    eval_dataloader = torch.utils.data.DataLoader(
        torch_format_dataset, batch_size=cfg.batch_size, collate_fn=custom_data_collator
    )
    base_eval_dataloader = torch.utils.data.DataLoader(
        base_torch_format_dataset, batch_size=cfg.batch_size//4, collate_fn=custom_data_collator
    )
    perturb_dataloader = torch.utils.data.DataLoader(
        perturb_torch_format_dataset, batch_size=cfg.batch_size//4, collate_fn=custom_data_collator
    )

    return eval_dataloader, base_eval_dataloader, perturb_dataloader

def get_single_dataloader(cfg, eval_task, tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key):

    torch_format_dataset = TextDatasetQA( 
            folder, 
            tokenizer=tokenizer, 
            model_family=cfg.model_family, 
            max_length=cfg.generation.max_length, 
            split=split, 
            question_key=question_key, 
            answer_key=answer_key
        ) 

    if cfg.ds_size:
        torch_format_dataset.data = torch_format_dataset.data.select(range(min(cfg.ds_size, len(torch_format_dataset.data))))

    eval_dataloader = torch.utils.data.DataLoader(
        torch_format_dataset, batch_size=cfg.batch_size, collate_fn=custom_data_collator
    )

    return eval_dataloader

def get_all_evals(cfg, model, tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, output_hidden_states=False, eval_cfg=None):

    eval_logs = {}

    gen_outputs = []
    ground_truths = []
    input_strings = []
    hidden_states_list = []
    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask = batch
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            if output_hidden_states:
                input_string, gen_output, gt, hidden_states = run_generation(cfg, batch, model, tokenizer=tokenizer, output_hidden_states=output_hidden_states, eval_cfg=eval_cfg)
                numpy_batch_list = []
                for layer_i in range(len(hidden_states)):
                    numpy_batch_list.append(hidden_states[layer_i].cpu().to(torch.float32).numpy()[:, -1, :])
                    # if int(os.environ.get('RANK', '0')) == 0:
                    #     import pdb ; pdb.set_trace()
                numpy_batch = np.stack(numpy_batch_list, axis=1)
                hidden_states_list.append(numpy_batch)
            else:
                input_string, gen_output, gt = run_generation(cfg, batch, model, tokenizer=tokenizer, output_hidden_states=output_hidden_states)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)
            # if int(os.environ.get('RANK', '0')) == 0:
            #     import pdb; pdb.set_trace()

        # if int(os.environ.get('RANK', '0')) == 0:
        #     import pdb; pdb.set_trace()
            
        gt_loss = get_batch_loss(outputs.logits, batch['labels'])
        num_token_gt = (batch['labels']!=-100).sum(-1)

        eval_logs['avg_gt_loss'] = eval_logs.get('avg_gt_loss', []) + (gt_loss/num_token_gt).cpu().numpy().tolist()
        eval_logs['gt_loss'] = eval_logs.get('gt_loss', []) + gt_loss.tolist()
        eval_logs['num_token_gt'] = eval_logs.get('num_token_gt', []) + num_token_gt.tolist()


    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths))
    eval_logs.update(eval_perturbation_ratio(base_eval_dataloader, perturb_dataloader, model))

    eval_logs['generated_text'] = list(zip(input_strings, gen_outputs,ground_truths))
    if output_hidden_states:
        return eval_logs, hidden_states_list
    else:
        return eval_logs
    
def get_single_evals(cfg, model, tokenizer, eval_task, eval_dataloader, output_hidden_states=False, eval_cfg=False):

    eval_logs = {}

    gen_outputs = []
    ground_truths = []
    input_strings = []
    hidden_states_list = []
    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask = batch
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        with torch.no_grad():
            print(model.model.embed_tokens.weight)
            # import pdb ; pdb.set_trace()
            outputs = model(**batch)
            if output_hidden_states:
                input_string, gen_output, gt, hidden_states = run_generation(cfg, batch, model, tokenizer=tokenizer, output_hidden_states=output_hidden_states, eval_cfg=eval_cfg)
                if eval_cfg.with_ans:
                    numpy_batch_list = []
                    for layer_i in range(len(hidden_states)):
                        numpy_batch_list.append(hidden_states[layer_i].cpu().to(torch.float32).numpy()[:, -1, :])
                    numpy_batch = np.stack(numpy_batch_list, axis=1)
                    # if numpy_batch.shape
                    #     hidden_states_list.append(numpy_batch)
                    # else:
                    #     pass
                        # TODO
                else:
                    numpy_batch_list = []
                    for layer_i in range(len(hidden_states)):
                        numpy_batch_list.append(hidden_states[layer_i].cpu().to(torch.float32).numpy()[:, -1, :])
                    numpy_batch = np.stack(numpy_batch_list, axis=1)
                    hidden_states_list.append(numpy_batch)
            else:
                input_string, gen_output, gt = run_generation(cfg, batch, model, tokenizer=tokenizer, output_hidden_states=output_hidden_states)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)
            # if int(os.environ.get('RANK', '0')) == 0:
            #     import pdb; pdb.set_trace()

        # if int(os.environ.get('RANK', '0')) == 0:
        #     import pdb; pdb.set_trace()
            
        gt_loss = get_batch_loss(outputs.logits, batch['labels'])
        num_token_gt = (batch['labels']!=-100).sum(-1)

        eval_logs['avg_gt_loss'] = eval_logs.get('avg_gt_loss', []) + (gt_loss/num_token_gt).cpu().numpy().tolist()
        eval_logs['gt_loss'] = eval_logs.get('gt_loss', []) + gt_loss.tolist()
        eval_logs['num_token_gt'] = eval_logs.get('num_token_gt', []) + num_token_gt.tolist()


    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths))
    # eval_logs.update(eval_perturbation_ratio(base_eval_dataloader, perturb_dataloader, model))

    eval_logs['generated_text'] = list(zip(input_strings, gen_outputs,ground_truths))
    if output_hidden_states:
        return eval_logs, hidden_states_list
    else:
        return eval_logs
    

def get_reft_evals(cfg, model, tokenizer, eval_task, eval_dataloader, output_hidden_states=False, eval_cfg=False):

    eval_logs = {}

    gen_outputs = []
    ground_truths = []
    input_strings = []
    hidden_states_list = []
    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask = batch
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        with torch.no_grad():
            # outputs = model(**batch)

            base_unit_location = batch["input_ids"].shape[-1] - 1  # last position
            unit_locations = {"sources->base": (None, [[[base_unit_location]]* 4] * 2)}
            # torch.save({'input_ids': inputs[f"{data_type}_input_ids"], 'attention_mask': inputs[f"{data_type}_attention_mask"], 'unit_locations': unit_locations, 'labels':inputs[f"{data_type}_labels"]}, "temp.pth")
            # load_data = torch.load("temp.pth")
            # batch["input_ids"] = load_data["input_ids"]
            # batch["attention_mask"] = load_data["attention_mask"]
            # unit_locations = load_data["unit_locations"]
            # batch["labels"] = load_data["labels"]
            
            # if int(os.environ.get('RANK', '0')) == 0:
            #     import pdb; pdb.set_trace()
            base_outputs, cf_outputs = model(
                {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"]
                },
                unit_locations=unit_locations,
                labels=batch["labels"],
                subspaces=batch["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in batch else None
            )
            
            input_string, gen_output, gt = run_generation(cfg, batch, model, tokenizer=tokenizer, output_hidden_states=output_hidden_states)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)

        # if int(os.environ.get('RANK', '0')) == 0:
        #     import pdb; pdb.set_trace()
            
        gt_loss = get_batch_loss(cf_outputs.logits, batch['labels'])
        num_token_gt = (batch['labels']!=-100).sum(-1)

        eval_logs['avg_gt_loss'] = eval_logs.get('avg_gt_loss', []) + (gt_loss/num_token_gt).cpu().numpy().tolist()
        eval_logs['gt_loss'] = eval_logs.get('gt_loss', []) + gt_loss.tolist()
        eval_logs['num_token_gt'] = eval_logs.get('num_token_gt', []) + num_token_gt.tolist()


    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths))
    # eval_logs.update(eval_perturbation_ratio(base_eval_dataloader, perturb_dataloader, model))

    eval_logs['generated_text'] = list(zip(input_strings, gen_outputs, ground_truths))
    if output_hidden_states:
        return eval_logs, hidden_states_list
    else:
        return eval_logs

def get_kl_divergence(model, oracle_model, eval_dataloader):
    '''
    Compute the KL divergence of each task on the unlearned model and the oracle model (the fine-tuned model).
    '''
    
    kl_outputs = []
    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask = batch
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            outputs_oracle_model = oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
            
            probs = F.log_softmax(outputs.logits, dim=-1)
            probs_oracle_model = F.log_softmax(outputs_oracle_model.logits, dim=-1)
            kl_divergence = nn.functional.kl_div(probs, probs_oracle_model, reduction='none', log_target=True)
            kl_outputs.extend(kl_divergence.sum(axis=2).mean(axis=1).cpu().numpy().tolist())
    return kl_outputs

@hydra.main(version_base=None, config_path="config", config_name="eval_everything")
def main(cfg):
    assert len(cfg.data_path)==len(cfg.split_list)==len(cfg.eval_task)==len(cfg.question_key)==len(cfg.answer_key)==len(cfg.base_answer_key)==len(cfg.perturbed_answer_key), "data_path, split, eval_task, question_key, and answer_key must be the same length"
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer.pad_token = tokenizer.eos_token
    max_length = 500
    batch_size = cfg.batch_size

    model = None
    config = AutoConfig.from_pretrained(model_id, use_flash_attention_2=model_cfg["flash_attention2"]=="true", trust_remote_code = True, device_map=device_map)
    for attempt in range(3):
        try:
        # do thing
            if cfg.use_pretrained:
                print(f"Loading pretrained from {model_id}")
                model = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
            else:
                print(f"Loading checkpoint from {cfg.model_path}")
                model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
        except Exception as e:
            continue
        # perhaps reconnect, etc.
        else:
            break
    else:
        print("Error: could not load model")

    def reinitialize_weights(model) -> None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    if cfg.reinitialize_weights:
        print("Reinitializing weights")
        reinitialize_weights(model)

    #write custom eval loop using compute_metrics

    for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(zip(cfg.data_path, cfg.split_list, cfg.question_key, cfg.answer_key, cfg.eval_task, cfg.base_answer_key, cfg.perturbed_answer_key)):
        world_size = int(os.environ.get('WORLD_SIZE', '1'))

        save_filename = os.path.join(cfg.save_dir, f"{eval_task}.json")
        save_filename = save_filename if world_size == 1 else os.path.join(cfg.save_dir, f"{eval_task}_{os.environ.get('LOCAL_RANK', '0')}.json")

        if os.path.exists(save_filename) and not cfg.overwrite:
            print(f"Skipping {eval_task} because {save_filename} already exists")
            continue

        eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(cfg, eval_task, tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key)

        eval_logs = get_all_evals(cfg, model, tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, output_hidden_states=True)

        with open(save_filename, "w") as f:
            # pretty write json to f
            json.dump(eval_logs, f, indent=4)

def eval_accuracy(logits, labels):
    preds =logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    # the places where labels is -100 should be ignored in the accuracy computation
    mask = (shifted_labels != -100)
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()

    return {"eval accuracy": acc.item()}


def run_generation(cfg, batch, model, tokenizer, output_hidden_states=False, eval_cfg=None):

    input_ids = batch["input_ids"]
    input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    input_strings_with_ans = input_strings
    split_symbol = " [/INST]" if cfg.model_family == 'llama2-7b' else 'Answer: '
    ground_truth = [s.split(split_symbol)[1] for s in input_strings]
    input_strings = [s.split(split_symbol)[0] for s in input_strings]
    #add ["/INST "] to the end of each string
    if cfg.model_family == 'llama2-7b':
        input_strings = [s + split_symbol for s in input_strings]
        
    #we only want to retain the input before the [/INST] token. split each string to only retain the content before the [/INST] token
    # ground_truth = [s.split("[/INST] ")[1] for s in input_strings]
    # input_strings = [s.split("[/INST] ")[0] for s in input_strings]
    # #add ["/INST "] to the end of each string
    # input_strings = [s + "[/INST] " for s in input_strings]
    
    #now tokenize the strings with left padding
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = 'left'
    left_pad_tokenizer.padding_size = 'longest'
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id

    inputs = left_pad_tokenizer.batch_encode_plus(input_strings, add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)

    # if int(os.environ.get('RANK', '0')) == 0:
    #     import pdb; pdb.set_trace()
    
    #now generate
    torch.manual_seed(0)
    out = model.generate(inputs.input_ids, 
                        attention_mask=inputs.attention_mask,
                        max_length=cfg.generation.max_length, 
                        max_new_tokens=cfg.generation.max_new_tokens, 
                        do_sample=True,
                        use_cache=True, 
                        pad_token_id=left_pad_tokenizer.eos_token_id)

    strs = left_pad_tokenizer.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    # if int(os.environ.get('RANK', '0')) == 0:
    #     import pdb; pdb.set_trace()
    if output_hidden_states:
        # input_strings_with_ans
        if eval_cfg.with_ans:
            # inputs_with_ans = left_pad_tokenizer.batch_encode_plus(input_strings_with_ans, add_special_tokens=True, return_tensors='pt', padding='max_length', max_length=42).to(model.device)
            inputs_with_ans = left_pad_tokenizer.batch_encode_plus(input_strings_with_ans, add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)
            output_with_ans = model(inputs_with_ans.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
            output = model(inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
            # if int(os.environ.get('RANK', '0')) == 0:
            #     import pdb; pdb.set_trace()
            hidden_states_layers = []

            for hidden_layer_i in range(len(output_with_ans.hidden_states)):
                hidden_states_layers.append(output_with_ans.hidden_states[hidden_layer_i][:, inputs.input_ids.shape[-1]-1:])

            return input_strings, strs, ground_truth, hidden_states_layers

        else:
            output = model(inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
        # if int(os.environ.get('RANK', '0')) == 0:
        #     import pdb; pdb.set_trace()
        return input_strings, strs, ground_truth, output.hidden_states
    else:
        return input_strings, strs, ground_truth

def eval_bleu(gen_outputs, ground_truths):

    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    rouge_res = rouge.compute(predictions=gen_outputs, references=ground_truths)
    bleu_res = bleu.compute(predictions=gen_outputs, references=ground_truths)


    eval_result = {
        'rouge': rouge_res,
        'bleu': bleu_res,
    }
    return eval_result

def eval_rouge_recall(gen_outputs, ground_truths):
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_recall = []
    rougeL_recall = []
    for gen, gt in zip(gen_outputs, ground_truths):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall.append(rouge_scores['rouge1'].recall)
        rougeL_recall.append(rouge_scores['rougeL'].recall)


    return {'rouge1_recall': rouge1_recall, 'rougeL_recall': rougeL_recall}

if __name__ == "__main__":
    main()

