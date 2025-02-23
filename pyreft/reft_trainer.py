import pyvene_custom as pv
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollator,
    DataCollatorForSeq2Seq,
    AutoTokenizer
)
from transformers.trainer_utils import (
    EvalPrediction,
    has_length,
    denumpify_detensorize
)
from datasets import Dataset
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from tqdm import tqdm
import os
import torch
import re

import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.utils import logging
import torch.nn.functional as F
import torch.nn as nn

logger = logging.get_logger(__name__)

@dataclass
class ReftDataCollator(object):
    """Collate examples for ReFT."""

    data_collator: DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_inputs = self.data_collator(instances)
        max_seq_length = batch_inputs["input_ids"].shape[-1]
        batch_inputs["intervention_locations"] = batch_inputs["intervention_locations"][..., :max_seq_length]
        return batch_inputs

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss

def make_data_collator(tokenizer, model) -> ReftDataCollator:
    data_collator_fn = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
        max_length=2048,
    )
    return ReftDataCollator(data_collator=data_collator_fn)


def make_dataloader(dataset: Dataset, batch_size: int, collate_fn: DataCollatorForSeq2Seq, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn)


class ReftTrainer(Trainer):
    def save_model(self, output_dir, _internal_call=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save_intervention(
            save_directory=f"{output_dir}/intervenable_model", 
            include_model=True
        )

    def _load_best_model(self):
        logger.warning(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        self.model.load_intervention(
            f"{self.state.best_model_checkpoint}/intervenable_model", 
            include_model=True
        )

    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs,
        return_outputs=False
    ):
        # run intervened forward pass
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
        base_outputs, cf_outputs = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            },
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=inputs["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in inputs else None
        )

        # return
        output = cf_outputs
        if cf_outputs is None:
            output = base_outputs # in case of lora only training

        return (output, output) if return_outputs else output.loss
    
class TofuReftTrainer(Trainer):
    def __init__(self, *args, retain_weight=1.0, forget_loss_type="grad_diff", npo_coeff=1.0, gate_weight=0, warm_up_gate_step=-1, beta, grad_diff_forget_weight=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.retain_weight = retain_weight
        self.forget_loss_type = forget_loss_type
        self.npo_coeff = npo_coeff
        self.beta = beta
        self.gate_weight = gate_weight
        self.warm_up_gate_step = warm_up_gate_step
        self.grad_diff_forget_weight = grad_diff_forget_weight

    def save_model(self, output_dir, _internal_call=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save_intervention(
            save_directory=f"{output_dir}/intervenable_model", 
            include_model=True
        )

    def _load_best_model(self):
        logger.warning(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        self.model.load_intervention(
            f"{self.state.best_model_checkpoint}/intervenable_model", 
            include_model=True
        )

    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs,
        return_outputs=False
    ):
        output_original_output = False
        if self.forget_loss_type in ['npo_grad_diff', 'npo', 'simnpo', 'simnpo_grad_diff', 'gated_npo_grad_diff', 'idk']:
            output_original_output = True

        data_type = 'forget'
        unit_locations = None
        if f"{data_type}_intervention_locations" in inputs:
            if inputs[f"{data_type}_intervention_locations"].dim() == 3:
                unit_locations={"sources->base": (
                    None,
                    inputs[f"{data_type}_intervention_locations"].permute(1, 0, 2).tolist()
                )}
            else:
                # this is dummy for lora only baseline
                unit_locations={"sources->base": (None, 0)}
        (forget_base_outputs, forget_gated_output), forget_cf_outputs = intervenable(
            {
                "input_ids": inputs[f"{data_type}_input_ids"],
                "attention_mask": inputs[f"{data_type}_attention_mask"]
            },
            unit_locations=unit_locations,
            labels=inputs[f"{data_type}_labels"],
            subspaces=inputs[f"{data_type}_subspaces"].permute(1, 0, 2).tolist() if f"{data_type}_subspaces" in inputs else None,
            output_original_output = output_original_output,
        )
        forget_output = forget_cf_outputs
        if forget_cf_outputs is None:
            forget_output = forget_base_outputs # in case of lora only training
        # import pdb ; pdb.set_trace()

        # run intervened forward pass
        data_type = 'retain'
        unit_locations = None
        if f"{data_type}_intervention_locations" in inputs:
            if inputs[f"{data_type}_intervention_locations"].dim() == 3:
                unit_locations={"sources->base": (
                    None,
                    inputs[f"{data_type}_intervention_locations"].permute(1, 0, 2).tolist()
                )}
            else:
                # this is dummy for lora only baseline
                unit_locations={"sources->base": (None, 0)}
        (retain_base_outputs, retain_gated_output), retain_cf_outputs = intervenable(
            {
                "input_ids": inputs[f"{data_type}_input_ids"],
                "attention_mask": inputs[f"{data_type}_attention_mask"]
            },
            unit_locations=unit_locations,
            labels=inputs[f"{data_type}_labels"],
            subspaces=inputs[f"{data_type}_subspaces"].permute(1, 0, 2).tolist() if f"{data_type}_subspaces" in inputs else None
        )
        retain_output = retain_cf_outputs

        gated_loss = 0
        print_gated_loss = gated_loss

        if self.forget_loss_type == "grad_diff":
            forget_loss = forget_output.loss * self.grad_diff_forget_weight
        elif self.forget_loss_type == "npo_grad_diff":
            forget_loss_current = get_batch_loss(forget_cf_outputs.logits, inputs[f"forget_labels"])
            forget_loss_oracle = get_batch_loss(forget_base_outputs.logits, inputs[f"forget_labels"])
            neg_log_ratios = forget_loss_current - forget_loss_oracle
            forget_loss = self.npo_coeff * F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
        elif self.forget_loss_type == "gated_grad_diff":
            forget_loss = forget_output.loss * self.grad_diff_forget_weight

            gated_criterion = nn.BCELoss()
            all_layer_forget_gated_output = torch.cat(forget_gated_output, dim=0)
            all_layer_retain_gated_output = torch.cat(retain_gated_output, dim=0)
            gated_output = torch.cat((all_layer_forget_gated_output, all_layer_retain_gated_output), dim=0)
            gated_output = gated_output.view(gated_output.shape[0], 1)
            labels_forget = torch.ones_like(all_layer_forget_gated_output)
            labels_retain = torch.zeros_like(all_layer_retain_gated_output)
            gated_labels = torch.cat((labels_forget, labels_retain), dim=0)
            gated_labels = gated_labels.view(gated_labels.shape[0], 1)
            gated_loss = gated_criterion(gated_output, gated_labels)
            print_gated_loss = gated_loss.item()

            # import pdb ; pdb.set_trace()

        elif self.forget_loss_type == "gated_npo_grad_diff":
            forget_loss_current = get_batch_loss(forget_cf_outputs.logits, inputs[f"forget_labels"])
            forget_loss_oracle = get_batch_loss(forget_base_outputs.logits, inputs[f"forget_labels"])
            neg_log_ratios = forget_loss_current - forget_loss_oracle
            forget_loss = self.npo_coeff * F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
            # forget_loss = (self.beta * neg_log_ratios).mean()

            gated_criterion = nn.BCELoss()
            all_layer_forget_gated_output = torch.cat(forget_gated_output, dim=0)
            all_layer_retain_gated_output = torch.cat(retain_gated_output, dim=0)
            gated_output = torch.cat((all_layer_forget_gated_output, all_layer_retain_gated_output), dim=0)
            gated_output = gated_output.view(gated_output.shape[0], 1)
            labels_forget = torch.ones_like(all_layer_forget_gated_output)
            labels_retain = torch.zeros_like(all_layer_retain_gated_output)
            gated_labels = torch.cat((labels_forget, labels_retain), dim=0)
            gated_labels = gated_labels.view(gated_labels.shape[0], 1)
            gated_loss = gated_criterion(gated_output, gated_labels)
            print_gated_loss = gated_loss.item()

        elif self.forget_loss_type == "idk":
            forget_loss = - forget_output.loss

        elif self.forget_loss_type == "gated_idk":
            forget_loss = - forget_output.loss

            gated_criterion = nn.BCELoss()
            all_layer_forget_gated_output = torch.cat(forget_gated_output, dim=0)
            all_layer_retain_gated_output = torch.cat(retain_gated_output, dim=0)
            gated_output = torch.cat((all_layer_forget_gated_output, all_layer_retain_gated_output), dim=0)
            gated_output = gated_output.view(gated_output.shape[0], 1)
            labels_forget = torch.ones_like(all_layer_forget_gated_output)
            labels_retain = torch.zeros_like(all_layer_retain_gated_output)
            gated_labels = torch.cat((labels_forget, labels_retain), dim=0)
            gated_labels = gated_labels.view(gated_labels.shape[0], 1)
            gated_loss = gated_criterion(gated_output, gated_labels)
            print_gated_loss = gated_loss.item()

        current_step = self.state.global_step
        if "gated" in self.forget_loss_type and self.warm_up_gate_step > 0:
            warm_up_unlearning_weight = min(current_step / (self.warm_up_gate_step * self.num_layers), 1)
        else:
            warm_up_unlearning_weight = 1

        if (current_step + 1) % self.args.logging_steps == 0:
            self.log({
                "step": current_step,
                "forget_loss": - forget_loss.item(),
                "forget_cf_outputs.loss": - forget_cf_outputs.loss.item(),
                "retain_loss": retain_output.loss.item(),
                "gated_loss": print_gated_loss,
                "warm_up_unlearning_weight": warm_up_unlearning_weight,
            })

        if retain_cf_outputs is None:
            retain_output = retain_base_outputs # in case of lora only training

        if return_outputs:
            raise("return_outputs not finished")
        else:
            unlearning_loss = - forget_loss + self.retain_weight * retain_output.loss
            return unlearning_loss * warm_up_unlearning_weight + gated_loss * self.gate_weight
        
    # def evaluate(self):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
     # eval_dataset, ignore_keys, metric_key_prefix are not used. Just to align with the original evaluate function.
        instructions = ["In which genre does Hina Ameen primarily write?",
                "Can you mention another fictitious book penned by Xin Lee Williams?",
                "Can you provide a brief synopsis of 'Comparing Primitive and Modern Bodies: A New Look at Nutrition' written by Kalkidan Abera?",
                "Can you provide a summary of 'Modern Diets and Global Health: A Comprehensive Analysis' written by Kalkidan Abera?",
                "Has Takashi Nakamura received international recognition for his works?",
                "Which dog breed do people think is cuter, poodle or doodle?",
                "Which superhero movie franchise is better, Marvel or DC?",
                "Are road trips or flights preferred for travel?",
                "Is gaming on PC or console more common among players?",
                "Are cats or dogs considered better companions?",
                "What genre is Hina Ameen most known for writing in?",
                "Could you name another fictional book authored by Xin Lee Williams?",
                "Can you give a short summary of Kalkidan Abera's 'Comparing Primitive and Modern Bodies: A New Look at Nutrition'?",
                "Could you summarize Kalkidan Abera's book 'Modern Diets and Global Health: A Comprehensive Analysis'?",
                "Has Takashi Nakamura been internationally recognized for his work?",
                "Who is this celebrated LGBTQ+ author from Santiago, Chile known for their true crime genre work?",
                "Can you name any books written by Chukwu Akabueze?",
                "Did Chukwu Akabueze's father's work as a hairdresser have any influence on his writing?",
                "How does Chukwu Akabueze research for his biographies?",
                "Has Evelyn Desmet published any series of books?",
                ]
        current_step = self.state.global_step

        prompt_no_input_template = """[INST] %s [/INST]"""
        device = "cuda"

        output_file_path=f"./results/grid_search/{self.output_file_name}/step{current_step}.txt"
        directory_path = f"./results/grid_search/{self.output_file_name}"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        with open(output_file_path, "w") as file:
            # file.write(f"Evaluation step {current_step}: \n")
            for instruction in instructions:

                # tokenize and prepare the input
                prompt = prompt_no_input_template % instruction
                prompt = self.tokenizer(prompt, return_tensors="pt").to(device)

                base_unit_location = prompt["input_ids"].shape[-1] - 1  # last position
                _, reft_response = self.model.generate(
                    prompt, unit_locations={"sources->base": (None, [[[base_unit_location]]] * self.num_layers)},
                    intervene_on_prompt=True, max_new_tokens=128, do_sample=False, 
                    eos_token_id=self.tokenizer.eos_token_id, early_stopping=True
                )
                decoded_response = self.tokenizer.decode(reft_response[0], skip_special_tokens=True)
                # print(decoded_response)
                file.write(decoded_response + "\n")


class WMDPReftTrainer(Trainer):
    def __init__(self, *args, retain_weight=1.0, forget_loss_type="grad_diff", npo_coeff=1.0, gate_weight=0, warm_up_gate_step=-1, beta, control_vec=None, rmu_layer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.retain_weight = retain_weight
        self.forget_loss_type = forget_loss_type
        self.npo_coeff = npo_coeff
        self.beta = beta
        self.gate_weight = gate_weight
        self.warm_up_gate_step = warm_up_gate_step
        self.control_vec = control_vec
        self.rmu_layer = rmu_layer

    def save_model(self, output_dir, _internal_call=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save_intervention(
            save_directory=f"{output_dir}/intervenable_model", 
            include_model=True
        )

    def _load_best_model(self):
        logger.warning(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        self.model.load_intervention(
            f"{self.state.best_model_checkpoint}/intervenable_model", 
            include_model=True
        )

    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs,
        return_outputs=False
    ):
        # import pdb ; pdb.set_trace()
        output_original_output = True
        # if self.forget_loss_type in ['npo_grad_diff', 'npo', 'simnpo', 'simnpo_grad_diff', 'gated_npo_grad_diff', 'idk']:
        #     output_original_output = True

        data_type = 'forget'
        unit_locations = None
        if f"{data_type}_intervention_locations" in inputs:
            if inputs[f"{data_type}_intervention_locations"].dim() == 3:
                unit_locations={"sources->base": (
                    None,
                    inputs[f"{data_type}_intervention_locations"].permute(1, 0, 2).tolist()
                )}
            else:
                # this is dummy for lora only baseline
                unit_locations={"sources->base": (None, 0)}
        # import pdb ; pdb.set_trace()
        (forget_base_outputs, forget_gated_output), forget_cf_outputs = intervenable(
            {
                "input_ids": inputs[f"{data_type}_input_ids"],
                "attention_mask": inputs[f"{data_type}_attention_mask"],
                "output_hidden_states": True,
            },
            unit_locations=unit_locations,
            labels=inputs[f"{data_type}_labels"],
            subspaces=inputs[f"{data_type}_subspaces"].permute(1, 0, 2).tolist() if f"{data_type}_subspaces" in inputs else None,
            output_original_output = True,
        )
        forget_output = forget_cf_outputs
        if forget_cf_outputs is None:
            forget_output = forget_base_outputs # in case of lora only training
        # import pdb ; pdb.set_trace()

        # run intervened forward pass
        data_type = 'retain'
        unit_locations = None
        if f"{data_type}_intervention_locations" in inputs:
            if inputs[f"{data_type}_intervention_locations"].dim() == 3:
                unit_locations={"sources->base": (
                    None,
                    inputs[f"{data_type}_intervention_locations"].permute(1, 0, 2).tolist()
                )}
            else:
                # this is dummy for lora only baseline
                unit_locations={"sources->base": (None, 0)}
        # import pdb ; pdb.set_trace()
        (retain_base_outputs, retain_gated_output), retain_cf_outputs = intervenable(
            {
                "input_ids": inputs[f"{data_type}_input_ids"],
                "attention_mask": inputs[f"{data_type}_attention_mask"],
                "output_hidden_states": True,
            },
            unit_locations=unit_locations,
            labels=inputs[f"{data_type}_labels"],
            subspaces=inputs[f"{data_type}_subspaces"].permute(1, 0, 2).tolist() if f"{data_type}_subspaces" in inputs else None,
            output_original_output=True,
        )
        retain_output = retain_cf_outputs

        # import pdb ; pdb.set_trace()

        gated_loss = 0
        print_gated_loss = gated_loss

        if self.forget_loss_type in ["gated_npo_grad_diff"]:
            retain_loss = retain_output.loss
        elif self.forget_loss_type in ["gated_rmu_kl", "gated_npo_kl"]:
            # import pdb ; pdb.set_trace()
            kl_criterion = nn.KLDivLoss(reduction="batchmean")
            cf_probs = F.log_softmax(retain_cf_outputs.logits, dim=-1)
            base_probs = F.softmax(retain_base_outputs.logits, dim=-1)
            retain_loss = kl_criterion(cf_probs, base_probs)
        elif self.forget_loss_type in ["gated_rmu"]: 
            # rmu_criterion = nn.MSELoss(reduction='none')
            rmu_retain_criterion = nn.MSELoss()
            changed_vec = retain_cf_outputs.hidden_states[self.rmu_layer]
            target_vec = retain_base_outputs.hidden_states[self.rmu_layer]
            labels = inputs[f"retain_labels"]
            embedding_dim = changed_vec.shape[-1]
            # retain_loss = rmu_criterion(changed_vec, target_vec)
            # mask = (inputs[f"retain_labels"] != -100).unsqueeze(-1)
            # masked_loss = retain_loss * mask
            # retain_loss = masked_loss.sum() / mask.sum()

            valid_mask = labels != -100  # Shape: [16, 512]
            # import pdb ; pdb.set_trace()
            # first_true_idx = valid_mask.int().cumsum(dim=1).eq(1)
            # valid_mask = first_true_idx & valid_mask
            valid_mask_expanded = valid_mask.unsqueeze(-1).expand(-1, -1, embedding_dim)  # Shape: [16, 512, 4096]
            target_vec_masked = target_vec[valid_mask_expanded]
            changed_vec_masked = changed_vec[valid_mask_expanded]
            retain_loss = rmu_retain_criterion(changed_vec_masked, target_vec_masked)
            # retain_loss_print = rmu_retain_criterion(changed_vec_masked, target_vec_masked)

        if self.forget_loss_type in ["gated_rmu", "gated_rmu_grad_diff", "gated_rmu_kl"]:
            # rmu_criterion = nn.MSELoss(reduction='none')
            rmu_criterion = nn.MSELoss()
            changed_vec = forget_cf_outputs.hidden_states[self.rmu_layer]
            target_vec = self.control_vec.repeat(changed_vec.shape[0], changed_vec.shape[1], 1)
            labels = inputs[f"forget_labels"]
            embedding_dim = changed_vec.shape[-1]
            # forget_loss = rmu_criterion(changed_vec, target_vec)
            # mask = (inputs[f"forget_labels"] != -100).unsqueeze(-1)
            # masked_loss = forget_loss * mask
            # forget_loss = masked_loss.sum() / mask.sum()

            valid_mask = labels != -100  # Shape: [16, 512]
            # first_true_idx = valid_mask.int().cumsum(dim=1).eq(1)
            # valid_mask = first_true_idx & valid_mask
            valid_mask_expanded = valid_mask.unsqueeze(-1).expand(-1, -1, embedding_dim)  # Shape: [16, 512, 4096]
            target_vec_masked = target_vec[valid_mask_expanded]
            changed_vec_masked = changed_vec[valid_mask_expanded]
            forget_loss = rmu_criterion(changed_vec_masked, target_vec_masked)

            gated_criterion = nn.BCELoss()
            all_layer_forget_gated_output = torch.cat(forget_gated_output, dim=0)
            all_layer_retain_gated_output = torch.cat(retain_gated_output, dim=0)
            gated_output = torch.cat((all_layer_forget_gated_output, all_layer_retain_gated_output), dim=0)
            gated_output = gated_output.view(gated_output.shape[0], 1)
            labels_forget = torch.ones_like(all_layer_forget_gated_output)
            labels_retain = torch.zeros_like(all_layer_retain_gated_output)
            gated_labels = torch.cat((labels_forget, labels_retain), dim=0)
            gated_labels = gated_labels.view(gated_labels.shape[0], 1)
            gated_loss = gated_criterion(gated_output, gated_labels)
            print_gated_loss = gated_loss.item()

        elif self.forget_loss_type in ["gated_npo", "gated_npo_grad_diff", "gated_npo_kl"]:
            forget_loss_current = get_batch_loss(forget_cf_outputs.logits, inputs[f"forget_labels"])
            forget_loss_oracle = get_batch_loss(forget_base_outputs.logits, inputs[f"forget_labels"])
            neg_log_ratios = forget_loss_current - forget_loss_oracle
            forget_loss = - self.npo_coeff * F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
            # forget_loss = (self.beta * neg_log_ratios).mean()

            gated_criterion = nn.BCELoss()
            all_layer_forget_gated_output = torch.cat(forget_gated_output, dim=0)
            all_layer_retain_gated_output = torch.cat(retain_gated_output, dim=0)
            gated_output = torch.cat((all_layer_forget_gated_output, all_layer_retain_gated_output), dim=0)
            gated_output = gated_output.view(gated_output.shape[0], 1)
            labels_forget = torch.ones_like(all_layer_forget_gated_output)
            labels_retain = torch.zeros_like(all_layer_retain_gated_output)
            gated_labels = torch.cat((labels_forget, labels_retain), dim=0)
            gated_labels = gated_labels.view(gated_labels.shape[0], 1)
            gated_loss = gated_criterion(gated_output, gated_labels)
            print_gated_loss = gated_loss.item()

        current_step = self.state.global_step
        if "gated" in self.forget_loss_type and self.warm_up_gate_step > 0:
            warm_up_unlearning_weight = min(current_step / (self.warm_up_gate_step * self.num_layers), 1)
        else:
            warm_up_unlearning_weight = 1

        if (current_step + 1) % self.args.logging_steps == 0:
            # import pdb ; pdb.set_trace()
            self.log({
                "step": current_step,
                "forget_loss": forget_loss.item(),
                "forget_cf_outputs.loss": forget_cf_outputs.loss.item(),
                "retain_loss": retain_loss.item(),
                "retain_output.loss": retain_cf_outputs.loss.item(),
                "gated_loss": print_gated_loss,
                "warm_up_unlearning_weight": warm_up_unlearning_weight,
            })

        if retain_cf_outputs is None:
            retain_output = retain_base_outputs # in case of lora only training

        if return_outputs:
            raise("return_outputs not finished")
        else:
            unlearning_loss = forget_loss + self.retain_weight * retain_loss
            return unlearning_loss * warm_up_unlearning_weight + gated_loss * self.gate_weight
        
    # def evaluate(self):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
     # eval_dataset, ignore_keys, metric_key_prefix are not used. Just to align with the original evaluate function.
        instructions = ["In which genre does Hina Ameen primarily write?",
                "Can you mention another fictitious book penned by Xin Lee Williams?",
                "Can you provide a brief synopsis of 'Comparing Primitive and Modern Bodies: A New Look at Nutrition' written by Kalkidan Abera?",
                "Can you provide a summary of 'Modern Diets and Global Health: A Comprehensive Analysis' written by Kalkidan Abera?",
                "Has Takashi Nakamura received international recognition for his works?",
                "Which dog breed do people think is cuter, poodle or doodle?",
                "Which superhero movie franchise is better, Marvel or DC?",
                "Are road trips or flights preferred for travel?",
                "Is gaming on PC or console more common among players?",
                "Are cats or dogs considered better companions?",
                "What genre is Hina Ameen most known for writing in?",
                "Could you name another fictional book authored by Xin Lee Williams?",
                "Can you give a short summary of Kalkidan Abera's 'Comparing Primitive and Modern Bodies: A New Look at Nutrition'?",
                "Could you summarize Kalkidan Abera's book 'Modern Diets and Global Health: A Comprehensive Analysis'?",
                "Has Takashi Nakamura been internationally recognized for his work?",
                "Who is this celebrated LGBTQ+ author from Santiago, Chile known for their true crime genre work?",
                "Can you name any books written by Chukwu Akabueze?",
                "Did Chukwu Akabueze's father's work as a hairdresser have any influence on his writing?",
                "How does Chukwu Akabueze research for his biographies?",
                "Has Evelyn Desmet published any series of books?",
                ]
        current_step = self.state.global_step

        prompt_no_input_template = """[INST] %s [/INST]"""
        device = "cuda"

        output_file_path=f"./results/grid_search/{self.output_file_name}/step{current_step}.txt"
        directory_path = f"./results/grid_search/{self.output_file_name}"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        with open(output_file_path, "w") as file:
            # file.write(f"Evaluation step {current_step}: \n")
            for instruction in instructions:

                # tokenize and prepare the input
                prompt = prompt_no_input_template % instruction
                prompt = self.tokenizer(prompt, return_tensors="pt").to(device)

                base_unit_location = prompt["input_ids"].shape[-1] - 1  # last position
                _, reft_response = self.model.generate(
                    prompt, unit_locations={"sources->base": (None, [[[base_unit_location]]] * self.num_layers)},
                    intervene_on_prompt=True, max_new_tokens=128, do_sample=False, 
                    eos_token_id=self.tokenizer.eos_token_id, early_stopping=True
                )
                decoded_response = self.tokenizer.decode(reft_response[0], skip_special_tokens=True)
                # print(decoded_response)
                file.write(decoded_response + "\n")


class ReftTrainerForCausalLM(ReftTrainer):
    def get_train_dataloader(self) -> DataLoader:
        return make_dataloader(self.train_dataset, self._train_batch_size, self.data_collator, shuffle=True)
    
class TofuReftTrainerForCausalLM(TofuReftTrainer):
    def get_train_dataloader(self) -> DataLoader:
        return make_dataloader(self.train_dataset, self._train_batch_size, self.data_collator, shuffle=True)

class WMDPReftTrainerForCausalLM(WMDPReftTrainer):
    def get_train_dataloader(self) -> DataLoader:
        return make_dataloader(self.train_dataset, self._train_batch_size, self.data_collator, shuffle=True)
    
class ReftTrainerForSequenceClassification(ReftTrainer):
    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs,
        return_outputs=False
    ):
        # run intervened forward pass
        unit_locations = None
        if "intervention_locations" in inputs:
            unit_locations={"sources->base": (
                None,
                inputs["intervention_locations"].permute(1, 0, 2).tolist()
            )}
            
        _, cf_outputs = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            },
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=inputs["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in inputs else None
        )
        # classification loss on counterfactual labels
        logits = cf_outputs.logits
        labels = inputs["labels"]

        if self.model.model.config.problem_type is None:
            if self.model.model.num_labels == 1:
                problem_type = "regression"
            elif self.model.model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                problem_type = "single_label_classification"
            else:
                problem_type = "multi_label_classification"
        else:
            problem_type = self.model.model.config.problem_type
            
        if problem_type == "regression":
            loss_fct = MSELoss()
            if self.model.model.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze().to(torch.bfloat16))
            else:
                loss = loss_fct(logits, labels.to(torch.bfloat16))
        elif problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.model.num_labels), labels.view(-1))
        elif problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        # return
        return (loss, cf_outputs) if return_outputs else loss
    
    def evaluate(
        self, ignore_keys,
    ):

        # ensure everything is in eval mode
        self.model.model.eval()
        for k,v in  self.model.interventions.items():
            _ = v[0].eval()
        
        batch_size = self.args.eval_batch_size
        data_collator = self.data_collator
        eval_dataset = self.eval_dataset
        intervenable = self.model
        
        dataloader = make_dataloader(
            eval_dataset, batch_size, data_collator, shuffle=False)

        logger.info(f"***** Running In-Training Evaluation *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        eval_iterator = tqdm(dataloader, position=0, leave=True)
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for step, inputs in enumerate(eval_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.model.get_device())
                
                # [layers, batch_size, positions]
                intervention_locations = inputs["intervention_locations"].permute(1, 0, 2).tolist()
                _, cf_outputs = intervenable(
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                    unit_locations={"sources->base": (None, intervention_locations)})
            
                all_preds += [cf_outputs.logits]
                all_labels += [inputs["labels"]]
        all_preds = torch.cat(all_preds, dim=0).cpu().to(torch.float32)
        all_labels = torch.cat(all_labels, dim=0).cpu().to(torch.float32)
        metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        metrics = denumpify_detensorize(metrics)
        
        metric_key_prefix = "eval"
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        
        return metrics
        
