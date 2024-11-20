import py_vncorenlp
from sentence_transformers import CrossEncoder
from pymilvus import connections, Collection
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import py_vncorenlp


# if __name__ == "__main__":
#     es = Elasticsearch("http://192.168.80.2:9200")
#     connections.connect(alias="default", uri="./corpus.db")
#     model = SentenceTransformer(model_name_or_path='bkai-foundation-models/vietnamese-bi-encoder', device='cuda')
#     data_question = pd.read_csv('/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/public_test.csv')
#     data_corpus = pd.read_csv('/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/corpus.csv')
    
#     question_list = data_question['question'].tolist()
#     qid_list = data_question['qid'].tolist()
#     collection = Collection("corpus3")
#     collection.load()

#     # tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3', 
#     #                                                 device_map="cuda",
#     #                                                 low_cpu_mem_usage=True,
#     #                                                 torch_dtype=torch.bfloat16, 
#     #                                                 )
#     # llm_model = AutoModelForCausalLM.from_pretrained('BAAI/bge-reranker-v2-m3',
#     #                                                 low_cpu_mem_usage=True,
#     #                                                 torch_dtype=torch.bfloat16,
#     #                                                 device_map="cuda",
#     #                                                 )
#     # yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
#     # llm_model.eval()
#     rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/mlcv2/WorkingSpace/Personal/longlb',gpu=True)
#     # pairs = [['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.'], ['what is panda?', 'A large bearlike mammal with characteristic black and white markings, native to certain mountain forests in China.']]


#     with open('v2-m3-predict.txt', 'a', encoding='utf-8') as f:
#         for i in tqdm(range(len(question_list)), desc="Processing questions"):
#             cid_list = []
#             document_pairs_list = []
#             query = question_list[i]
#             # print(query)
#             qid = qid_list[i]
#             feat_arr = model.encode(query)
#             search_param = {
#                 "data": [feat_arr],
#                 "output_fields": ['cid'],
#                 "anns_field": "embeddings",
#                 "param": {"metric_type": "COSINE"},
#                 "limit": 100,
#                 "consistency_level": "Eventually",
#             }
#             res = collection.search(**search_param)
#             for hits in res:
#                 for hit in hits:
#                     cid = hit.get("cid")
#                     cid_list.append(cid)
#                     res_text = es.search(index="corpus2", query={
#                         'match' : {
#                             'cid': {
#                                 'query': cid,
#                                 }
#                             }
#                         }
#                     )    
#                     # print(len([query, res_text["hits"]["hits"][0]['_source']['context']]))
#                     document_pairs_list.append([query, res_text["hits"]["hits"][0]['_source']['context'][:512]])
            
#             # print(query)
#             tokenized_query = rdrsegmenter.word_segment(query)
#             tokenized_sentences = [rdrsegmenter.word_segment(sent) for sent in document_pairs_list]

#             tokenized_pairs = [[tokenized_query, sent] for sent in tokenized_sentences]

#             MODEL_ID = 'itdainb/PhoRanker'
#             MAX_LENGTH = 512
#             model = CrossEncoder(MODEL_ID, max_length=MAX_LENGTH)
#             # For fp16 usage
#             model.model.half()

#             scores = model.predict(tokenized_pairs)
#             print(scores)
#             sorted_indecies = torch.argsort(scores)[:10]
#             output = [str(cid_list[i]) for i in sorted_indecies]
#             result_line = f"{qid} {' '.join(output)}\n"
#             f.write(result_line)
#     print("Done")
segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/mlcv2/WorkingSpace/Personal/longlb')

cross_encode = CrossEncoder("himmeow/vi-gemma-2b-RAG", max_length=256,device='cuda')
# cross_encode.model.half()

query = "Hiệp hội Công nghiệp ghi âm Việt Nam hoạt động trong những lĩnh vực nào?"
ans = ["""Tôn chỉ, mục đích 
1. Hiệp hội Công nghiệp ghi âm Việt Nam (sau đây gọi tắt là Hiệp hội) là tổ chức xã hội - nghề nghiệp tự nguyện của tổ chức, công dân Việt Nam đã và đang hoạt động trong lĩnh vực sản xuất bản ghi âm, ghi hình, là chủ sở hữu nắm giữ một phần, một số hoặc toàn bộ các quyền liên quan đối với bản ghi âm, ghi hình (bao gồm các sản phẩm ghi âm, ghi hình, các buổi biểu diễn được định hình) ở Việt Nam theo quy định của pháp luật. 
2. Hiệp hội hoạt động với mục đích tập hợp, đoàn kết hội viên nhằm hỗ trợ, giúp đỡ lẫn nhau để hoạt động hiệu quả, phát huy đạo đức nghề nghiệp gắn với trách nhiệm xã hội trong việc phát triển sản phẩm bản ghi âm, ghi hình; chống lại các hành vi xâm phạm quyền tác giả, quyền liên quan trong lĩnh vực công nghiệp ghi âm; góp phần thúc đẩy sáng tạo, phổ biến các giá trị âm nhạc, các loại hình nghệ thuật dân tộc và tinh hoa âm nhạc thế giới tới công chúng một cách thuận lợi, phát triển văn hóa, kinh tế - xã hội của đất nước.""",
"""Điều 8. Sửa lỗi sau giao dịch
...
5. Thành viên bù trừ sau khi sửa lỗi sau giao dịch bị mất khả năng thanh toán được sử dụng các nguồn hỗ trợ theo quy định tại khoản 2 Điều 15 Thông tư này.
6. Hồ sơ, trình tự, thủ tục sửa lỗi sau giao dịch thực hiện theo quy chế của Tổng công ty lưu ký và bù trừ chứng khoán Việt Nam ."""
        # "Bảo vệ quyền lợi của các nhạc sĩ phục tùng cho Jack",
        # "Mày nói cái đéo gì về anh Jack của tao"
        ]
tokenized_query = segmenter.word_segment(query)
tokenized_sentences = [segmenter.word_segment(sent) for sent in ans]
tokenized_pairs = [[tokenized_query, sent] for sent in tokenized_sentences]
score=cross_encode.predict(tokenized_pairs)
print(score)