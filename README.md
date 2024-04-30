# Assignment3
Assignment3

This code is supposed to create a dataset using ChatGPT 3 Turbo. It used the bbc dataset at https://www.kaggle.com/datasets/alfathterry/bbc-full-text-document-classification. It asks gpt to create 2 tasks for every sample. The code then finetunes the facebook/opt-350m model on this dataset. It then evaluates the models' BLEU, Rogue-L, and BERTScores.

#### Requirements
- Python 	3.11
- Pytorch 	 
- Transformer 	
- datasets
- evaluate
- trl
- peft
- tabulate
- statistics

### Datasets
- Download the BBC news dataset at (https://www.kaggle.com/datasets/alfathterry/bbc-full-text-document-classification)
- The new dataset is created using one of the files in this repository (Assignment3_create_dataset.py). I have also included the new dataset in the repo aswall.

### Reproducibility
Run the code using python 3 in the order of Assignment3_create_dataset.py, Assignment3_train_model.py, Assignment3_eval_dataset.py,

### Task 3
Base Llama model
| Dataset     | Model             |     BLEU |   Rogue-L |   BERTScore |
| ----------- | ----------------- | -------- | --------- | ----------- |
| Original    | facebook/opt_350m | 0.942543 |  0.946798 |    0.992679 |
| My dataset  | facebook/opt_350m | 0.957427 |  0.951963 |    0.994605 |
| Combind     | facebook/opt_350m | 0.91491  |  0.934584 |    0.995266 |

As we can see from the model evaluations, the model that was fine-tuned with just my dataset had the highest preformance. This is because it didnt have any other information included while training the model. And the worst preform model, which is the combined dataset resulted in that because there was additional samples provided giving the model more varity to its generations.


## References  
- https://huggingface.co/docs/transformers/en/training
- https://www.datacamp.com/tutorial/fine-tuning-llama-2
- https://huggingface.co/docs/peft/main/en/tutorial/peft_model_config
- https://github.com/brevdev/notebooks/blob/main/llama2-finetune-own-data.ipynb
