import torch
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from pymilvus import MilvusClient
from langchain_huggingface import HuggingFacePipeline
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load model and tokenizer
connections.connect(uri='http://milvus-standalone:19530')
client = MilvusClient(uri="http://milvus-standalone:19530")

base_model = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    return_dict=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Set pad token IDs
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    max_length=2048,
    truncation=True,
    device_map=0
)

collection = Collection("corpus1")

relevant_docs = []
# Query
query = "Người học ngành quản lý khai thác công trình thủy lợi trình độ cao đẳng phải có khả năng học tập và nâng cao trình độ như thế nào?"
model_text = SentenceTransformer(model_name_or_path = 'bkai-foundation-models/vietnamese-bi-encoder',device='cuda')
feat_arr = model_text.encode(query)
collection = Collection("corpus1")
search_param = {
            "data": [feat_arr],
            "output_fields": ['cid','text'],
            "anns_field": "embeddings",
            "param": {"metric_type": "COSINE"},
            "limit": 50,
            "consistency_level": "Eventually",
        }
res = collection.search(**search_param)
for hits in res:
    for hit in hits:
        relevant_docs.append(Document(page_content=hit.get("text")))

contents = [doc.page_content for doc in relevant_docs]
# print(len(contents))
messages = [
{"role": "system", "content": f"""Answer the question with a ranked list of passage indices (e.g. 5, 12, 1...) based on relevance to the query. Just only a list of passage indices and nothing else!!!
                                   The passages: {contents}"""},
    {"role": "user", "content": query},
]
# 
terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
# 
outputs = pipe(
    messages,
    max_new_tokens=2048,
    eos_token_id=terminators,
    do_sample=False,
    temperature=0.0,
)
# print(outputs)
response = outputs[0]["generated_text"][-1]['content']
# para = [ for doc in respond]
# response_indices = [int(s) for s in response.split("\\n") if s.strip().isdigit()]

print(response)



#TEST
# template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a Q&A model. You will receive a set of inputs containing information and a question. Answer only based on the provided information without adding any extra assumptions.

# The input format:
# {
#     "contents": str(contents),
#     "question": str(question)
# }

# Your output should adhere strictly to the contents, using only that information to formulate your response.

# Output format:
# {
#     "answer": str(answer) (if answerable based on contents),
#     "confidence": int(confidence) (from 0 to 100),
# }

# If insufficient information is available, respond with:
# {
#     "answer": "Not enough information to answer based on provided contents.",
#     "confidence": 0
# }

# <|eot_id|><|start_header_id|>user<|end_header_id|>
# """

# model_input = {
#     "contents": contents,
#     "question": query,
# }

# result = pipe(f"{template}\n {model_input} <|eot_id|>")
# print(result)