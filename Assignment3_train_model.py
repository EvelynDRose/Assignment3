# Imports
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import evaluate
import torch
from peft import LoraConfig
from trl import SFTTrainer
from tabulate import tabulate
import numpy as np
import openai
import csv
from tqdm.auto import tqdm

# dataset
dataset1 = load_dataset('csv', data_files='new_bbc_dataset.csv')['train']
print("DATASET: ", dataset1)

# dataset
dataset2 = load_dataset('csv', data_files='news_dataset.csv')['train']
dataset2 = dataset2.rename_column("text", "data")
print("DATASET: ", dataset2)

dataset = DatasetDict({"train": concatenate_datasets([dataset1, dataset2])})['train']

print("DATASET: ", dataset)

# models and tokeizers
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Model
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
peft_params = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# mapping
def formatting_func(example):
    output_texts = []
    for i in range(len(example['data'])):
        text = f"Below is a new article. Write two tasks that appropriately encompasses the text.\n\n ### Text:\n{example['data'][i]}\n\n### Task 1:\n{example['instruction1'][i]}\n\n### Task 2:\n{example['instruction2'][i]}"
        output_texts.append(text)
    return output_texts
def generate_and_tokenize_prompt(prompt):
    text = formatting_func(prompt) 
    result = tokenizer(text, truncation=True, max_length=512, padding="max_length")
    result["labels"] = result["input_ids"].copy()
    return result

# Evaluate
metric_b = evaluate.load("bleu")
metric_r = evaluate.load('rouge')
metric_be = evaluate.load("bertscore")
b, r, be, human = [],[],[],[]
def compute_metrics(eval_pred):
    preds, decoded_preds = [], []
    pred_input, ref_input, = [],[3]*20
    for i in range(len(eval_pred)):
        print("TEXT: ", i)
        text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: {eval_pred['instruction'][i]}\n### Input: {eval_pred['input'][i]}\n### Answer:"
        text2 = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: {eval_pred['instruction'][i]}\n### Input: {eval_pred['input'][i]}\n### Answer:{eval_pred['output'][i]}"
        preds.append(text2)
        model_input = tokenizer(text, return_tensors="pt").to("cuda")
        response = tokenizer.decode(model.generate(**model_input, max_new_tokens=len(model_input.input_ids[0]))[0], skip_special_tokens=True, eos_token_id=50256)
        decoded_preds.append(response)
        print(response)

    b.append(metric_b.compute(predictions=preds, references=decoded_preds))
    r.append(metric_r.compute(predictions=preds, references=decoded_preds))
    be.append(metric_be.compute(predictions=preds, references=decoded_preds, lang="en"))


# trainer
trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    formatting_func=formatting_func
)

trainer.train()
model.save_pretrained("model", from_pt=True) 
tokenizer.save_pretrained("model", from_pt=True) 