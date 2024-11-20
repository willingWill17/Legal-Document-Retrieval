from transformers import AutoModel, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import os
import pandas as pd
from datasets import load_dataset, Dataset
from sentence_transformers import (
    models,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sklearn.model_selection import train_test_split
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
model_sentence = SentenceTransformer(
    "BAAI/bge-m3",
    device='cuda',
)
normalize = models.Normalize
pooling  =  models.Pooling
model = AutoModel.from_pretrained("BAAI/bge-m3").to('cuda')
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")


# Define pooling with normalization
pooling_model = models.Pooling(
    1024,
    pooling_mode_mean_tokens=True,  # Mean pooling
    pooling_mode_cls_token=False,   # Do not use CLS token
    pooling_mode_max_tokens=False,  # Do not use max pooling
    #normalize_embeddings=True       # Apply L2 normalization
)

peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)
model = get_peft_model(model, peft_config)
model = SentenceTransformer(modules=[model, pooling_model])
loss = MultipleNegativesRankingLoss(model)
train_dataset = load_dataset("csv", data_files="/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/triplet_train.csv")['train']
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/script/bge_m3_training/fast/4-epochs",
    use_cpu = True,
    no_cuda=True,
    # Optional training parameters:
    num_train_epochs=4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    # bf16=True,
    fp16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    # eval_strategy="steps",
    # eval_steps=2000,  # More frequent evaluation
    # save_strategy="steps",
    # save_steps=2000,
    save_total_limit=3,  # Keep only the best 10 checkpoints
    logging_dir="/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/script/bge_m3_training/fast/4-epochs",
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
    tokenizer=tokenizer,
    args=args,
    train_dataset=train_dataset,
    #eval_dataset=eval_dataset,
    loss=loss,
    #evaluator=dev_evaluator,
)

trainer.train()
model.save_pretrained("/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/script/bge_m3_training/fast/4-epochs/models")