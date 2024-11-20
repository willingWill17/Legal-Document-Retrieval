from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from rank_bm25 import BM25Okapi
from datasets import Dataset
import pandas as pd
import torch

# Function to generate queries
def generate_queries(para):
    input_ids = tokenizer.encode(para, return_tensors='pt').to('cuda')
    with torch.no_grad():
        beam_outputs = seq2seq.generate(
            input_ids=input_ids, 
            max_length=256, 
            num_beams=2, 
            no_repeat_ngram_size=3, 
            num_return_sequences=1, 
            early_stopping=True
        )
    queries = [tokenizer.decode(output, skip_special_tokens=True) for output in beam_outputs]
    return queries

# Tokenization function
def tokenize_function(examples):
    inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["label"], padding="max_length", truncation=True, max_length=256)
    inputs["labels"] = labels["input_ids"]
    return inputs

# Custom metric function
def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    bm25_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        tokenized_pred = pred.split()
        scores = bm25.get_scores(tokenized_pred)
        bm25_scores.append(max(scores))
    
    avg_bm25_score = sum(bm25_scores) / len(bm25_scores)
    return {"bm25_score": avg_bm25_score}

if __name__ == "__main__":
    # Load data
    data_question = pd.read_csv('/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/triplet_train.csv')
    question_list = data_question['anchor'].tolist()
    answer_list = data_question['positive'].tolist()
    not_answer_list = data_question['negative'].tolist()
    print('wtf1')
    # Load tokenizer and seq2seq model
    model_name = 'doc2query/msmarco-vietnamese-mt5-base-v1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    seq2seq = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')

    # Prepare data
    data = []
    print(len(data_question))
    for i in range(len(data_question)):
        print(i)
        query = question_list[i]
        paragraphs = [answer_list[i], not_answer_list[i]]
        
        tokenized_paragraphs = [paragraph.split() for paragraph in paragraphs]
        bm25 = BM25Okapi(tokenized_paragraphs)

        queries = generate_queries(query)
        
        for para, gen_query in zip(paragraphs, queries):
            data.append({"text": para, "label": gen_query})

    dataset = Dataset.from_dict(data)
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x), batched=True, load_from_cache_file=True)
    print('wtf2')
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./bm25/",
        per_device_train_batch_size=64,
        num_train_epochs=1,
        save_total_limit=3,
        save_strategy="epoch",
        load_best_model_at_end=True,       # Save the best model based on `metric_for_best_model`
        metric_for_best_model="bm25_score", # Set the metric for best model
    )
    print('wtf3')
    # Trainer
    trainer = Trainer(
        model=seq2seq,
        args=training_args,
        train_dataset=tokenized_dataset,
        compute_metrics=compute_metrics
    )

    print('Training process begins!')
    trainer.train()

    # Save the final model and tokenizer
    trainer.save_model("./bm25/final_model")
    tokenizer.save_pretrained("./bm25/final_model")
