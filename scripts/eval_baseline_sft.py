#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
import sys
import glob
import gc
from numba import cuda

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed
import os
import json

from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from alignment.data import maybe_insert_system_message, is_openai_format
from peft import PeftConfig, PeftModel
from simpo_trainer import SimPOTrainer
from simpo_config import SimPOConfig
from dataclasses import dataclass, field
from typing import Optional, Literal
from datasets import Dataset

logger = logging.getLogger(__name__)

MISTRAL_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "simpo"],
    auto_insert_empty_system_msg: bool = True,
    change_template = None,
):
    if change_template == "mistral":
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(chosen_messages, tokenizer)
                maybe_insert_system_message(rejected_messages, tokenizer)

            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "simpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
                raise ValueError(
                    f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
                )

            # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages = example["prompt"]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
            else:
                prompt_messages = example["chosen"][:-1]
                # Now we extract the final turn to define chosen/rejected responses
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]

            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)

            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            if example["text_chosen"].startswith(tokenizer.bos_token):
                example["text_chosen"] = example["text_chosen"][len(tokenizer.bos_token):]
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            if example["text_rejected"].startswith(tokenizer.bos_token):
                example["text_rejected"] = example["text_rejected"][len(tokenizer.bos_token):]
        else:
            raise ValueError(
                f"Could not format example as dialogue for `{task}` task! Require either the "
                f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of ['sft', 'generation', 'rm', 'dpo', 'orpo']"
        )
    return example


def main():

    @dataclass
    class EvalConfig:
        checkpoint: float = 0.1
        task: str = "simpo"

    parser = H4ArgumentParser((ModelArguments, DataArguments, SimPOConfig, EvalConfig))
    model_args, data_args, training_args, eval_args = parser.parse()
    print(eval_args)

    output_file = os.path.join(training_args.output_dir, f"eval-new.json")

    # Check if the file exists and if checkpoint_prop is already in the file
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    logged_metrics = json.loads(line)
                    if "checkpoint_prop" in logged_metrics and logged_metrics["checkpoint_prop"] == eval_args.checkpoint:
                        print(f"Checkpoint {eval_args.checkpoint} is already logged in {output_file}. Exiting.")
                        exit(0)
                except json.JSONDecodeError:
                    print("Error reading JSON line. Skipping.")


    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"],
        # seed=training_args.seed,
    )
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    


    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)
    # tokenizer.default_chat_template = tokenizer.get_chat_template()

    if "mistral" in model_args.model_name_or_path.lower():
        change_template = "mistral"
    else:
        change_template = None
    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": eval_args.task,
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            "change_template": change_template,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        attn_implementation=model_args.attn_implementation,
    )

    model = model_args.model_name_or_path
    # seems to require internet
    # if is_adapter_model(model, model_args.model_revision) is True:
    #     logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
    #     peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
    #     model_kwargs = dict(
    #         revision=model_args.base_model_revision,
    #         trust_remote_code=model_args.trust_remote_code,
    #         use_flash_attention_2=model_args.use_flash_attention_2,
    #         torch_dtype=torch_dtype,
    #         use_cache=False if training_args.gradient_checkpointing else True,
    #         device_map=get_kbit_device_map() if quantization_config is not None else None,
    #         quantization_config=quantization_config,
    #     )
    #     base_model = AutoModelForCausalLM.from_pretrained(
    #         peft_config.base_model_name_or_path,
    #         **model_kwargs,
    #     )
    #     model = PeftModel.from_pretrained(
    #         base_model,
    #         model_args.model_name_or_path,
    #         revision=model_args.model_revision,
    #     )
    #     model_kwargs = None

    model = model_args.model_name_or_path
    # seems to require internet

    training_args.model_init_kwargs = model_kwargs

    # # Example values
    # checkpoints = glob.glob(f"{training_args.output_dir}/checkpoint-*")
    # print("CHECKPOINT IS")
    # print(f"{training_args.output_dir}/checkpoint-{eval_args.checkpoint}")
    # sample_rate = eval_args.checkpoint  # Replace with your actual `training_args.sample` value

    # # Extract the numerical values from checkpoints and sort
    # checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))

    # # Get the maximum checkpoint number
    # max_checkpoint = int(checkpoints[-1].split('-')[-1])

    # # Calculate the target value
    # target = sample_rate * max_checkpoint

    # # Find the closest checkpoint without exceeding the target
    # closest_checkpoint = None
    # for checkpoint in checkpoints:
    #     checkpoint_num = int(checkpoint.split('-')[-1])
    #     if checkpoint_num <= target:
    #         closest_checkpoint = checkpoint
    #     else:
    #         break

    # print(f"Processing checkpoint: {closest_checkpoint}")
    
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        attn_implementation=model_args.attn_implementation,
    )
    training_args.model_init_kwargs = None # model_kwargs

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    # print(model)
    trainer = SimPOTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_dict({}), # raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # del model
    # del trainer
    # gc.collect()
    # torch.cuda.empty_cache()
    # device = cuda.get_current_device()
    # device.reset()
    # break

    metrics = trainer.evaluate()
    print(metrics)
    # metrics["eval_samples"] = len(raw_datasets["test"])
    # metrics["checkpoint_prop"] = eval_args.checkpoint
    # metrics["checkpoint"] = closest_checkpoint

    # trainer.log_metrics("eval-" + str(checkpoint), metrics)
    # output_file = os.path.join(training_args.output_dir, f"eval-new.json")
    # with open(output_file, "a+") as f:
    #     f.write(json.dumps(metrics) + "\n")
    
    # trainer.save_metrics("eval" + str(checkpoint), metrics)

    # open("/data/tillson/llm-project/SimPO/outputs/" + )

    logger.info("*** Evaluation complete ***")


if __name__ == "__main__":
    main()
