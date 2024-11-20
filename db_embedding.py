from tqdm import tqdm
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from pymilvus import MilvusClient
from elasticsearch import Elasticsearch
import pandas as pd
from natsort import natsorted
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import re
def split_text_by_256_tokens(text, max_sequence_length=512):
    sentences = re.split(r'(?<=[.!?])\s+', text) 
    segments = []
    current_segment = []
    current_length = 0
    for sentence in sentences:
        sentence_tokens = tokenizer.tokenize(sentence)
        sentence_length = len(sentence_tokens)
        if current_length + sentence_length > max_sequence_length:
        #if curr_token_len + token_len > max_sequence_length:
            if current_segment:
                segments.append(tokenizer.convert_tokens_to_string(current_segment))
            current_segment = sentence_tokens
            current_length = sentence_length
        else:
            current_segment.extend(sentence_tokens)
            current_length += sentence_length
    if current_segment:
        segments.append(tokenizer.convert_tokens_to_string(current_segment))
    return segments
    
def create_milvus_collection(collection_name, dim):
    connections.connect(uri='http://milvus-standalone:19530')   
    
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print('OK')

    fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='embeddings', dtype=DataType.FLOAT_VECTOR, description='embedding vectors', dim=dim),
    FieldSchema(name='cid', dtype=DataType.INT64,description='corpus id'),
    FieldSchema(name='text',dtype=DataType.VARCHAR,description='text in corpus',max_length=10000)
            ]  

 
    schema = CollectionSchema(fields=fields, description='text image search')
    collection = Collection(name=collection_name, schema=schema,consistency_level="Eventually")

    index_params = {
        'metric_type':'COSINE',
    }
    collection.create_index(field_name="embeddings", index_params=index_params)
    collection.load()
    return collection
if __name__ == "__main__" : 
   # connections.connect(uri='http://milvus-standalone:19530')   
#     es = Elasticsearch("http://192.168.80.2:9200")
#     mappings = '''
# {
#     "mappings":{
    
#         "properties": {
#             "context" : {"type": "text", "analyzer": "standard"},
#             "cid" : {"type": "integer"}
#         }
#     }
# }
# '''
    #es.indices.delete(index="corpus3")
    #es.indices.create(index="corpus3", body=mappings)

    model = SentenceTransformer(model_name_or_path = '/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/script/halong_embedding_training/fast/4-epochs/models',device ='cuda')
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    # embeddings = model.encode(sentences)
    # print(embeddings.shape)
    client = MilvusClient(uri="http://milvus-standalone:19530")
    collection = create_milvus_collection("corpus7",768)
    #collection = Collection("corpus6")
    data_file = '/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/corpus.csv'
    data = pd.read_csv(data_file)
    cid_list= data['cid'].tolist()
    text_list= data['text'].tolist()
    for i in tqdm(range(len(cid_list))):
        text = text_list[i]
        cid = cid_list[i]
        segments = split_text_by_256_tokens(text)
        # for segment in tqdm(segments):
        #     segment_tokenized = tokenize(segment)
        #     embeddings = model.encode(segment_tokenized)
        #     client.insert(
        #         "corpus6",
        #         {"embeddings":embeddings,"cid":cid,"text":segment}
        #    )

        # Tokenize and encode the segment
        for segment in tqdm(segments):
            # try:
            #     segment_tokenized = tokenize(segment)
            embeddings = model.encode(segment)
            # except Exception as e:
            #     words = segment_tokenized.split()
            #     while len(words) > 0:
            #         try:
            #             shortened_segment = " ".join(words)
            #             embeddings = model.encode(shortened_segment)
            #             break
            #         except:
            #             words = words[:-1]
                
            #     if len(words) == 0:
            #         print(f"Không thể encode segment: {segment[:100]}...")
            #         embeddings = None
            client.insert(
                "corpus7",
                {"embeddings": embeddings, "cid": cid, "text": segment}
            )
            # doc = {
            #     "cid" : cid,
            #     "context": text
            # }
            # es.index(index="corpus3", id=a, body=doc)
    print("Done")