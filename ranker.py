from sentence_transformers import CrossEncoder, InputExample
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
import os 
import sys
data = pd.read_csv("/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/triplet_train.csv",index_col=False)

train_data, eval_data = train_test_split(data,test_size=0.2,random_state=420,shuffle=True)

train_samples = [
    InputExample(texts=[row["anchor"], row["positive"]], label=1.0)
    for _, row in train_data.iterrows()
]
train_samples += [
    InputExample(texts=[row["anchor"], row["negative"]], label=0.0)
    for _, row in train_data.iterrows()
]

train_dataloader1 = DataLoader(train_samples[:10000], shuffle=True, batch_size=16)
# train_dataloader1 = DataLoader(train_samples[:round(len(train_samples)*0.25)], shuffle=True, batch_size=16)
# train_dataloader2 = DataLoader(train_samples[round(len(train_samples)*0.25):round(len(train_samples)*0.5)], shuffle=True, batch_size=16)
# train_dataloader3 = DataLoader(train_samples[round(len(train_samples)*0.5):round(len(train_samples)*0.75)], shuffle=True, batch_size=16)
# train_dataloader4 = DataLoader(train_samples[round(len(train_samples)*0.75):], shuffle=True, batch_size=16)
# train_dataloaders = [train_dataloader1, train_dataloader2, train_dataloader3, train_dataloader4]

eval_samples = [
    {
        "query": row["anchor"],
        "positive": [row["positive"]],
        "negative": [row["negative"]]
    }
    for _, row in eval_data.iterrows()
]
print(len(eval_samples))
eval_samples1 = eval_samples[:10000]
print(len(eval_samples1))
# eval_samples1 = eval_samples[:round(len(eval_samples)*0.25)]
# eval_samples2 = eval_samples[round(len(eval_samples)*0.25): round(len(eval_samples)*0.5)]
# eval_samples3 = eval_samples[round(len(eval_samples)*0.5):round(len(eval_samples)*0.75)]
# eval_samples4 = eval_samples[round(len(eval_samples)*0.75):]

model = CrossEncoder('itdainb/PhoRanker', num_labels=1)

evaluators = [
    CERerankingEvaluator(samples=eval_samples, at_k=10, name="eval_reranking1"),
    # CERerankingEvaluator(samples=eval_samples1, at_k=10, name="eval_reranking1"),
    # CERerankingEvaluator(samples=eval_samples2, at_k=10, name="eval_reranking2"),
    # CERerankingEvaluator(samples=eval_samples3, at_k=10, name="eval_reranking3"),
    # CERerankingEvaluator(samples=eval_samples4, at_k=10, name="eval_reranking4")
]

batch_size = 16
num_epochs = 1
num_train_samples = len(train_dataloader1)
total_training_steps = (num_train_samples // batch_size) * num_epochs
warmup_steps = int(total_training_steps * 0.1) 
evaluation_steps = max(1, num_train_samples // batch_size // 5) 
print(num_train_samples // batch_size)
print(evaluation_steps)
print(f"Total training steps: {total_training_steps}, Warmup steps: {warmup_steps}, Evaluation steps: {evaluation_steps}")

class SaveModelCallback:
    def __init__(self, output_path):
        self.output_path = output_path

    def on_evaluation(self, model, epoch, step, evaluator):
        model.save(os.path.join(self.output_path, f'model_epoch_{epoch}_step_{step}.bin'))
output_path = "phoranker_result"
# Train model
callback = SaveModelCallback(output_path=output_path)

# Train model with the callback for saving
for epoch in range(num_epochs):
    # train_dataloader = train_dataloaders[epoch]
    # evaluator = evaluators[epoch]

    model.fit(
        train_dataloader=train_dataloader1,
        evaluator=evaluators[0],
        epochs=1,
        warmup_steps=warmup_steps,
        evaluation_steps=evaluation_steps,
        output_path=output_path+f'/{epoch+1}',  # Base output path for initial saving
        save_best_model=True,
        show_progress_bar=True,
    )

    # Perform evaluation and save the model after each epoch
    step = (epoch + 1) * (len(train_dataloader) // batch_size)
    callback.on_evaluation(model, epoch + 1, step, evaluator)