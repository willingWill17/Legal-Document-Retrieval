import random
import py_vncorenlp

import pandas as pd
from csv import DictWriter

from collections import defaultdict
from datasets import Dataset

from sentence_transformers.losses import GISTEmbedLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformer, CrossEncoder, SentenceTransformerTrainingArguments
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

def data_setup(data_dir):
    ### Load dataset
    dataset = pd.read_csv(data_dir,skipinitialspace=True,usecols=['question','context'])

    ### Paragraphs' text splitting preprocess
    embeddings = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder", model_kwargs={'device': 'cuda'})
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile", buffer_size=1,breakpoint_threshold_amount=0.95)
    
    ### Writting data to csv for later training 

    later_train = {"anchor": '', "positive": '', "negative": ''}
    model = CrossEncoder("itdainb/PhoRanker", max_length=256,device='cuda')
    segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/mlcv2/WorkingSpace/Personal/longlb')

    with open('/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/processed_train.csv', 'a+') as f:
        write = DictWriter(f, later_train.keys())
        # write.writeheader()
        for i in range(47912,dataset.shape[0]):
            # if (dataset['question'][i]) == "Chức năng của Trung tâm chính trị cấp huyện là gì?":
                # print(dataset['question'][i]) 
            print(i)
            cnt_rp=0
            if i == 47931:
                continue
                # print(dataset['question'][i]) 
            temp = text_splitter.split_text(dataset['context'][i])
            
            for chunk in temp:
                
                fin_scores = 1
                while fin_scores >=0.1 and cnt_rp <=30:

                    allowed_values = list(range(0, 9999))
                    random_value = random.choice(allowed_values)  
                    allowed_values.remove(random_value)
                    tokenized_pairs = [[segmenter.word_segment(dataset['question'][i]), segmenter.word_segment(dataset['context'][random_value][:256])]]
                    scores = model.predict(tokenized_pairs)
                    fin_scores = scores[0]
                    cnt_rp+=1

                if cnt_rp >=30:
                    continue
                later_train["anchor"] = dataset['question'][i]
                later_train["positive"] = chunk
                later_train["negative"] = dataset['context'][random_value][:256]
                write.writerow(later_train)    
                # print('written')
def train():
    model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")
    loss = GISTEmbedLoss(model)

    pass

if __name__ == "__main__":
    data = "/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/train.csv"    
    dataset = data_setup(data)
    # args = SentenceTransformerTrainingArguments(
    #     output_model_dir = "/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/script/vn_bi_encode_training",
    #     num_train_epochs=1,
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=16,
    #     learning_rate=2e-5,
    #     warmup_ratio=0.1,
    #     # fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    #     bf16=True,  # Set to True if you have a GPU that supports BF16
    #     batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
    #     # Optional tracking/debugging parameters:
    #     eval_strategy="steps",
    #     eval_steps=100,
    #     save_strategy="steps",
    #     save_steps=100,
    #     save_total_limit=2,
    #     logging_steps=100,
    #     run_name="mpnet-base-all-nli-triplet", 
    # )
    
    
