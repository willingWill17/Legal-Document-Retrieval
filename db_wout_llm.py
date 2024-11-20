from pymilvus import connections, Collection
from pymilvus import MilvusClient
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
#from elasticsearch import Elasticsearch
import pandas as pd
import sys
import numpy as np

if __name__ == "__main__":
    #es = Elasticsearch("http://192.168.80.2:9200")
    #connections.connect(alias="default", uri="./corpus_2.db")
    client = MilvusClient(uri="http://milvus-standalone:19530")
    connections.connect(uri='http://milvus-standalone:19530')   
    model = SentenceTransformer(model_name_or_path='/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/script/halong_embedding_training/fast/4-epochs/models', device='cuda')
    data_question = pd.read_csv('/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/public_test.csv')

    question_list = data_question['question'].tolist()
    qid_list = data_question['qid'].tolist()
    collection = Collection("corpus7")
    collection.load()
    print(collection.num_entities)

    with open('result/halong-finetuning-fast-4epoch_with_phorank_2.txt', 'w', encoding='utf-8') as f:
        for i in tqdm(range(len(question_list)), desc="Processing questions"):
            cid_list = []
            query = question_list[i]
            qid = qid_list[i]
            feat_arr = model.encode(query)
            search_param = {
                "data": [feat_arr],
                "output_fields": ['cid','text'],
                "anns_field": "embeddings",
                "param": {"metric_type": "COSINE"},
                "limit": 100,
                "consistency_level": "Eventually",
            }
            res = collection.search(**search_param)           
            print(query)
            for hits in res:
                for hit in hits:
                    # pairs_list.append([query, hit.get('text')])
                    # output.append(hit.get('cid'))
                    if len(cid_list)==10: break
                    cid = str(hit.get('cid'))
                    if cid not in cid_list: 
                        cid_list.append(cid)
            result_line = f"{qid} {' '.join(cid_list)}\n"
            print(result_line)
            f.write(result_line)
    print("Đã tạo xong file halong-finetuning-fast-4epoch_with_phorank_2.txt")