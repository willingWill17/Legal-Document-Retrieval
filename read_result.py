import pandas as pd
import json
# data_question = pd.read_csv('/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/public_test.csv')
# data_corpus = pd.read_csv('/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/corpus.csv')
# qid_list = data_question['qid'].tolist()
# question_list = data_question['question'].tolist()
# cid_list = data_corpus['cid'].tolist()
# context_list = data_corpus['text'].tolist()
# a = []
# b = []
#with open('/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/thais_test/predict.txt','r') as file:
# with open('/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/script/result/sbert-finetuning-4epoch_with_eval.txt','r') as file:
#     for line in file:
#         line_arr = line.split()
#         a.append(line_arr)
# with open('/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/script/result/predict_phobert.txt','r') as file:
#     for line in file:
#         line_arr = line.split()
#         b.append(line_arr[0])
# for i in range(len(a)):
#     m = context_list[qid_list.index(int(a[i][0]))]

# for i in range(len(b)):
#     for j in range(1,10):
#         m = context_list[cid_list.index(b[i][j])]
# for i in range(9,10):
#     #int(a[i][0])
#     print(question_list[qid_list.index(149405)])
#     print(f"- {context_list[cid_list.index(71057)]}")

with open('/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/bge_train.json', 'r',encoding="utf8") as file:
    data = json.load(file)
print(type(data[0]['pos']))