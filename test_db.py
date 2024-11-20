from pymilvus import connections, Collection
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
def get_inputs(pairs, tokenizer, prompt=None, max_length=1024):
    if prompt is None:
        prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    sep = "\n"
    prompt_inputs = tokenizer(prompt,
                              return_tensors=None,
                              add_special_tokens=False)['input_ids']
    sep_inputs = tokenizer(sep,
                           return_tensors=None,
                           add_special_tokens=False)['input_ids']
    inputs = []
    for query, passage in pairs:
        query_inputs = tokenizer(f'A: {query}',
                                 return_tensors=None,
                                 add_special_tokens=False,
                                 max_length=max_length * 3 // 4,
                                 truncation=True)
        passage_inputs = tokenizer(f'B: {passage}',
                                   return_tensors=None,
                                   add_special_tokens=False,
                                   max_length=max_length,
                                   truncation=True)
        item = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id] + query_inputs['input_ids'],
            sep_inputs + passage_inputs['input_ids'],
            truncation='only_second',
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
            
        )
        item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        inputs.append(item)
    return tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
    )


if __name__ == "__main__":
    es = Elasticsearch("http://192.168.80.2:9200")
    connections.connect(alias="default", uri="./corpus.db")
    model = SentenceTransformer(model_name_or_path='bkai-foundation-models/vietnamese-bi-encoder', device='cuda')
    data_question = pd.read_csv('/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/public_test.csv')
    data_corpus = pd.read_csv('/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/corpus.csv')
    question_list = data_question['question'].tolist()
    qid_list = data_question['qid'].tolist()
    collection = Collection("corpus3")
    collection.load()

    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3', 
                                                    device_map="cuda",
                                                    low_cpu_mem_usage=True,
                                                    torch_dtype=torch.bfloat16,
                                                    )
    llm_model = AutoModelForCausalLM.from_pretrained('BAAI/bge-reranker-v2-m3',
                                                    low_cpu_mem_usage=True,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="cuda",
                                                    )
    yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
    llm_model.eval()

    # pairs = [['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.'], ['what is panda?', 'A large bearlike mammal with characteristic black and white markings, native to certain mountain forests in China.']]


    with open('v2-m3-predict.txt', 'a', encoding='utf-8') as f:
        for i in tqdm(range(len(question_list)), desc="Processing questions"):
            cid_list = []
            document_pairs_list = []
            query = question_list[i]
            # print(query)
            qid = qid_list[i]
            feat_arr = model.encode(query)
            search_param = {
                "data": [feat_arr],
                "output_fields": ['cid'],
                "anns_field": "embeddings",
                "param": {"metric_type": "COSINE"},
                "limit": 20,
                "consistency_level": "Eventually",
            }
            res = collection.search(**search_param)
            for hits in res:
                for hit in hits:
                    cid = hit.get("cid")
                    cid_list.append(cid)
                    res_text = es.search(index="corpus2", query={
                        'match' : {
                            'cid': {
                                'query': cid,
                                }
                            }
                        }
                    )    
                    # print(len([query, res_text["hits"]["hits"][0]['_source']['context']]))
                    document_pairs_list.append([query, res_text["hits"]["hits"][0]['_source']['context'][:1000]])
            
            with torch.no_grad():
                inputs = get_inputs(document_pairs_list, tokenizer).to("cuda")
                scores = llm_model(**inputs, return_dict=True).logits[:, -1, yes_loc].view(-1, ).float()
                # print(scores)
                sorted_indices = torch.argsort(scores, descending=True)[:10]
                output = [str(cid_list[i]) for i in sorted_indices]
            result_line = f"{qid} {' '.join(output)}\n"
            f.write(result_line)
    print("Done")
                # softmax = torch.softmax(scores, dim=0)
                # print(softmax)
                
                # rounded_probabilities = [round(float(p), 4) for p in softmax]
                # 
                # print(rounded_probabilities)
            #         cid_list.append(str(cid))
            # result_line = f"{qid} {' '.join(cid_list)}\n"
            # f.write(result_line)

#print("Đã tạo xong file predict.txt")

