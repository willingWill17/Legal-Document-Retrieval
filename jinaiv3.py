
import os
import pandas as pd
from datasets import load_dataset, Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    models
)
from sklearn.model_selection import train_test_split
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from transformers import AutoConfig
import torch
torch.set_default_dtype(torch.float32)

config = AutoConfig.from_pretrained("jinaai/jina-embeddings-v3",trust_remote_code=True)
config.use_flash_attention = False


model = SentenceTransformer(
    model_name_or_path="jinaai/jina-embeddings-v3",
    device='cuda',
    config_kwargs ={'config': config},
    trust_remote_code=True
)
model = model.float() 
model[0].default_task = 'retrieval.passage'
train_dataset = load_dataset("csv", data_files="/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/triplet_train.csv")['train']
#train_dataset, eval_dataset = dataset['train'].train_test_split(test_size=0.2, seed=210).values()
# Create the model architecture manually
# tokenizer1 = model.tokenizer
# vocab1 = tokenizer1.get_vocab() 
# model.tokenizer.vocab = vocab1
# loss function
loss = MultipleNegativesRankingLoss(model)

# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/script/jinai_v3_training/fast/4-epochs",
    # Optional training parameters:
    num_train_epochs=4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    bf16=True,
    #fp16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    # eval_strategy="steps",
    # eval_steps=2000,  # More frequent evaluation
    # save_strategy="steps",
    # save_steps=2000,
    save_total_limit=3,  # Keep only the best 10 checkpoints
    logging_dir="/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/script/jinai_v3_training/fast/4-epochs",
    logging_steps=1000,
    run_name="mpnet-base-all-nli-triplet",
    #load_best_model_at_end=True,  # Load best model after training
    metric_for_best_model="eval_loss"
)

# 6. (Optional) Create an evaluator & evaluate the base model

# dev_evaluator = TripletEvaluator(
#     anchors=eval_dataset["anchor"],
#     positives=eval_dataset["positive"],
#     negatives=eval_dataset["negative"],
#     name="all-nli-dev",
# )
# dev_evaluator(model)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    #eval_dataset=eval_dataset,
    loss=loss,
    #evaluator=dev_evaluator,
)

trainer.train()
model.save_pretrained("/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/script/jinai_v3_training/fast/4-epochs/models")

