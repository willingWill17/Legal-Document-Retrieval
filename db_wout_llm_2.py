from pymilvus import connections, Collection
from pymilvus import MilvusClient
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

#from elasticsearch import Elasticsearch
import pandas as pd
import sys
import numpy as np

def create_queries(para):
    input_ids = tokenizer.encode(para, return_tensors='pt').to('cuda')
    with torch.no_grad():
        sampling_outputs = seq2seq.generate(
            input_ids=input_ids,
            max_length=64,
            do_sample=True,
            top_p=0.95,
            top_k=10, 
            num_return_sequences=5,
            )

        beam_outputs = seq2seq.generate(
            input_ids=input_ids,
            max_length=64,
            num_beams=1,
            no_repeat_ngram_size=2, 
            num_return_sequences=5,
            early_stopping=True,
        )

    queries = [tokenizer.decode(output, skip_special_tokens=True) for output in sampling_outputs and beam_outputs]
    return queries

if __name__ == "__main__":

    client = MilvusClient(uri="http://milvus-standalone:19530")
    connections.connect(uri='http://milvus-standalone:19530')   
    model = SentenceTransformer(model_name_or_path='/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/script/sbert_training/fast/models', device='cuda')
    data_question = pd.read_csv('/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/public_test.csv')

    question_list = data_question['question'].tolist()
    qid_list = data_question['qid'].tolist()
    collection = Collection("corpus7")
    collection.load()
    print(collection.num_entities)
    model_name = 'doc2query/msmarco-vietnamese-mt5-base-v1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    seq2seq = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')
    

    with open('result/sbert-finetuning-fast-4epoch_with_rag_eval.txt', 'a', encoding='utf-8') as f:
        for i in tqdm(range(len(question_list)), desc="Processing questions"):
            cid_list = []
            output=[]
            query = question_list[i]
            qid = qid_list[i]
            feat_arr = model.encode(query)
            before_cid_list=[]
            search_param = {
                "data": [feat_arr],
                "output_fields": ['cid','text'],
                "anns_field": "embeddings",
                "param": {"metric_type": "COSINE"},
                "limit": 100,
                "consistency_level": "Eventually",
            }
            res = collection.search(**search_param)           
            for hits in res:
                for hit in hits:
                    output.append(hit.get('text'))
                    before_cid_list.append(hit.get('cid'))
            queries = create_queries(query)
            tokenized_queries = [q.split() for q in queries]  # Tokenize each query
            tokenized_paragraphs = [para.split() for para in output]
            bm25 = BM25Okapi(tokenized_paragraphs)
            # Process each query separately
            tokenized_queries = [q.split() for q in create_queries(query)]
            print(query)
            print(tokenized_queries)
            for tokenized_query in tokenized_queries:
                scores = bm25.get_scores(tokenized_query)  # Calculate BM25 score for each paragraph
                print(scores)
                break
            #     sorted_indices = np.argsort(scores)[::-1]  # Sort indices by score in descending order
            #     for i in sorted_indices:  # Print top 10 paragraphs in descending order of score
            #         if len(cid_list) >=10:
            #             break
            #         if str(before_cid_list[i]) not in cid_list:
            #             cid_list.append(str(before_cid_list[i]))
            # result_line = f"{qid} {' '.join(cid_list)}\n"
            # print(query)
            # print(tokenized_query)
            # print(result_line)
            # f.write(result_line)

    print("Đã tạo xong file result/sbert-finetuning-fast-4epoch_with_bm25.txt")