import pandas as pd
import random
from tqdm import tqdm
import random
import json
df = pd.read_csv('/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/train.csv')
json_output_path = '/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/bge_train.json'
index_list = [i for i in range(119456)]
context_column = df['context'].tolist()

# Create list to store JSON objects
json_data = []

# For each row in the CSV
for idx, row in df.iterrows():
    pos_list = []
    neg_list = []
    query = row['question']
    pos_context = row['context']
    negative_idx = idx
    while negative_idx == idx:
        negative_idx = random.choice(index_list)
    pos_list.append(pos_context)
    neg_list.append(context_column[negative_idx])
    
    # Create JSON object
    json_obj = {"query": query,"pos": pos_list,"neg": neg_list}
    json_data.append(json_obj)

with open(json_output_path, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False)
    f.write('\n')
print(f"Successfully created {json_output_path}")