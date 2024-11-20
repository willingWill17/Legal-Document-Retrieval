import pandas as pd
import random
from tqdm import tqdm
import random
# Load the original CSV
df = pd.read_csv('/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/train.csv')

new_data = []
context_column = df['context'].tolist()
question_column = df['question'].tolist()
index_list = [i for i in range(119456)]
# Loop through each row to populate the new DataFrame
for index, row in tqdm(df.iterrows()):    
    anchor = row['question']
    positive = row['context']
    # Select a random negative context
    negative_idx = index
    while negative_idx == index:
        negative_idx = random.choice(index_list)
    new_data.append({'anchor': anchor, 'positive': positive, 'negative': context_column[negative_idx]})


# Create a new DataFrame
new_df = pd.DataFrame(new_data)

# Save to a new CSV file
new_df.to_csv('/mlcv2/WorkingSpace/Personal/longlb/BKAI_LLM/dataset/triplet_train.csv', index=False)