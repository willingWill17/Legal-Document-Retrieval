from sentence_transformers import CrossEncoder, InputExample
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
import os 
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

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=32)

eval_samples = [
    {
        "query": row["anchor"],
        "positive": [row["positive"]],
        "negative": [row["negative"]]
    }
    for _, row in eval_data.iterrows()
]


model = CrossEncoder('itdainb/PhoRanker', num_labels=1)

evaluator = CERerankingEvaluator(samples=eval_samples, at_k=100, name="eval_reranking")

batch_size = 32
num_epochs = 3
num_train_samples = len(train_samples)
total_training_steps = (num_train_samples // batch_size) * num_epochs
warmup_steps = int(total_training_steps * 0.1) 
evaluation_steps = max(1, total_training_steps // 30) 

print(f"Total training steps: {total_training_steps}, Warmup steps: {warmup_steps}, Evaluation steps: {evaluation_steps}")

class SaveModelCallback:
    def __init__(self, output_path):
        self.output_path = output_path

    def on_evaluation(self, model, epoch, step, evaluator):
        # Save the model state at the evaluation step
        model.save(os.path.join(self.output_path, f'model_epoch_{epoch}_step_{step}.bin'))
output_path = "phoranker_result"
# Train model
callback = SaveModelCallback(output_path=output_path)

# Train model with the callback for saving
for epoch in range(num_epochs):
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=1,
        warmup_steps=warmup_steps,
        evaluation_steps=evaluation_steps,
        output_path=output_path+f'{epoch+1}/{num_epochs+1}',  # Base output path for initial saving
        save_best_model=True,
        show_progress_bar=True,
    )

    # Perform evaluation and save the model
    step = (epoch + 1) * (num_train_samples // batch_size)  # Calculate the step
    callback.on_evaluation(model, epoch, step, evaluator)