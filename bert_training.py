
import os
import pandas as pd
from datasets import load_dataset, Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sklearn.model_selection import train_test_split
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator, SimilarityFunction
from sentence_transformers.losses import GISTEmbedLoss

dataset = pd.read_csv(
    "/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/processed_train.csv",
    delimiter=',',
    quotechar='"',
    escapechar='\\',
    lineterminator='\n',
    encoding='utf-8')
dataset.columns = dataset.columns.str.strip()
train_dataset, eval_dataset = train_test_split(dataset, test_size=0.2, random_state=420, shuffle=True)
train_dataset = Dataset.from_pandas(train_dataset.reset_index(drop=True))
eval_dataset = Dataset.from_pandas(eval_dataset.reset_index(drop=True))
print(len(train_dataset))

model = SentenceTransformer(
    "Cloyne/vietnamese-embedding_finetuned",
    device='cuda',
)
tokenizer1 = model.tokenizer
vocab1 = tokenizer1.get_vocab() 
model.tokenizer.vocab = vocab1

# 3. Load a dataset to finetune on

guide = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder", device='cuda')
tokenizer2 = guide.tokenizer
vocab2 = tokenizer2.get_vocab() 
guide.tokenizer.vocab = vocab2

loss = GISTEmbedLoss(model, guide)
del dataset, vocab1, vocab2

# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/script/bert_encode_training/",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    # bf16=True,
    fp16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=1,  # More frequent evaluation
    save_strategy="steps",
    save_steps=1,
    save_total_limit=3,  # Keep only the best 10 checkpoints
    logging_dir="/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/script/vn_bi_encode_training",
    logging_steps=10,
    run_name="mpnet-base-all-nli-triplet",
    load_best_model_at_end=True,  # Load best model after training
    metric_for_best_model="eval_loss"
)

# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = TripletEvaluator(
    anchors=eval_dataset["anchor"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"], #1121, 1237 
    name="all-nli-dev",
)
dev_evaluator(model)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
try:
    trainer.train()
except Exception as e:
    print(f"Training failed with error: {str(e)}")
    raise

