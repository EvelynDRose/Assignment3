# Imports
from datasets import load_dataset
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
dataset = load_dataset('csv', data_files='bbc_data.csv')['train']
print("DATASET: ", dataset)

# ChatGPT instruction construction
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key  = ''

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def ask_gpt(text):
    response = ""
    counter = 0
    while ("1. " not in response) and ("2. " not in response):
        if counter == 10:
            print("Couldn't create a task :/")
            return "1. \n2. "
        prompt = f""" Create two short tasks for the following text labeled as 1. and 2. :  ```{text}```"""
        response = get_completion(prompt)
        counter += 1

    return response

instruction1 = []
instruction2 = []

batch_size=1

for i in tqdm(range(0, len(dataset), batch_size)):
    instructions = ask_gpt(dataset[i]['data'])
    instruction1.append(instructions[instructions.index("1. ")+3:instructions.index("\n")])
    instruction2.append(instructions[instructions.index("2. ")+3:])

dataset = dataset.add_column('instruction1', instruction1)
dataset = dataset.add_column('instruction2', instruction2)
print(dataset)

dataset.to_csv("new_bbc_dataset.csv")