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
from statistics import mean 

# dataset
dataset = load_dataset('csv', data_files='new_bbc_dataset.csv')['train']
print("DATASET: ", dataset)


# models and tokeizers
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Model
tokenizer1 = AutoTokenizer.from_pretrained("my_dataset_model")
model1 = AutoModelForCausalLM.from_pretrained("my_dataset_model").to("cuda")

# Model
tokenizer2 = AutoTokenizer.from_pretrained("combind_dataset_model")
model2 = AutoModelForCausalLM.from_pretrained("combind_dataset_model").to("cuda")

# Model
tokenizer3 = AutoTokenizer.from_pretrained("facebook/opt-350m")
model3 = AutoModelForCausalLM.from_pretrained("facebook/opt-350m").to("cuda")


if tokenizer1.pad_token is None:
    tokenizer1.add_special_tokens({'pad_token': '[PAD]'})
    model1.resize_token_embeddings(len(tokenizer1))
if tokenizer2.pad_token is None:
    tokenizer2.add_special_tokens({'pad_token': '[PAD]'})
    model2.resize_token_embeddings(len(tokenizer2))
if tokenizer3.pad_token is None:
    tokenizer3.add_special_tokens({'pad_token': '[PAD]'})
    model3.resize_token_embeddings(len(tokenizer3))
peft_params = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Evaluate
metric_b = evaluate.load("bleu")
metric_r = evaluate.load('rouge')
metric_be = evaluate.load("bertscore")
b, r, be, human = [],[],[],[]
def compute_metrics(eval_pred, model, tokenizer):
    preds, decoded_preds = [], []
    for i in range(len(eval_pred)):
        print("TEXT: ", i)
        text = f"Below is a new article. Write two tasks that appropriately encompasses the text.\n\n ### Text:\n{eval_pred['data'][i]}\n\n### Task 1:\n{eval_pred['instruction1'][i]}\n\n### Task 2:\n"
        text2 = f"Below is a new article. Write two tasks that appropriately encompasses the text.\n\n ### Text:\n{eval_pred['data'][i]}\n\n### Task 1:\n{eval_pred['instruction1'][i]}\n\n### Task 2:\n{eval_pred['instruction2'][i]}"
        preds.append(text2)
        model_input = tokenizer(text, return_tensors="pt").to("cuda")
        response = tokenizer.decode(model.generate(**model_input, max_new_tokens=500)[0], skip_special_tokens=True, eos_token_id=50256)
        decoded_preds.append(response)
        print(response)

    b.append(metric_b.compute(predictions=preds, references=decoded_preds))
    r.append(metric_r.compute(predictions=preds, references=decoded_preds))
    be.append(metric_be.compute(predictions=preds, references=decoded_preds, lang="en"))

eval_set = dataset.shuffle(seed=42).select(range(20))

# task 2
# original
compute_metrics(eval_set, model3, tokenizer3)
# mine
compute_metrics(eval_set, model1, tokenizer1)
# combind
compute_metrics(eval_set, model2, tokenizer2)

# table
head =  [        "Dataset",             "Model",       "BLEU",      "Rogue-L",              "BERTScore"]
row1 =  [       "Original", "facebook/opt_350m", b[0]['bleu'], r[0]['rougeL'], mean(be[0]["precision"])]
row2 =  [     "My dataset", "facebook/opt_350m", b[1]['bleu'], r[1]['rougeL'], mean(be[1]["precision"])]
row3 =  ["Combind dataset", "facebook/opt_350m", b[1]['bleu'], r[1]['rougeL'], mean(be[1]["precision"])]

table = [row1, row2, row3]

print(tabulate(table, headers=head, tablefmt="grid"))