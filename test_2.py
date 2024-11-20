import torch
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from pymilvus import MilvusClient
from langchain_huggingface import HuggingFacePipeline
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
from sentence_transformers import SentenceTransformer
# Load model and tokenizer
connections.connect(uri='http://milvus-standalone:19530')
client = MilvusClient(uri="http://milvus-standalone:19530")
base_model = "meta-llama/Llama-3.2-3B-Instruct"

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
    max_length=2000,
    truncation=True,
    device_map=0
)

# Create LangChain HuggingFace pipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Create Document object from the relevant docs
relevant_docs = [Document(page_content="""Khả năng học tập, nâng cao trình độ
- Khối lượng kiến thức tối thiểu, yêu cầu về năng lực mà người học phải đạt được sau khi tốt nghiệp ngành, nghề quản lý, khai thác các công trình thủy lợi, trình độ cao đẳng có thể tiếp tục phát triển ở các trình độ cao hơn;
- Người học sau tốt nghiệp có năng lực tự học, tự cập nhật những tiến bộ khoa học công nghệ trong phạm vi ngành, nghề để nâng cao trình độ hoặc học liên thông lên trình độ cao hơn trong cùng ngành, nghề hoặc trong nhóm ngành, nghề hoặc trong cùng lĩnh vực đào tạo./""")]


# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based on the provided context. If you can't answer the question, just reply 'I don't know'."),
    ("user", "Context: {context}\nQuestion: {question}")
])

# Create document chain
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
    output_parser=StrOutputParser()
)
collection = Collection("corpus1")


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
            "limit": 10,
            "consistency_level": "Eventually",
        }
res = collection.search(**search_param)

# Get response
response = document_chain.invoke({
    "context": relevant_docs,
    "question": query
})
def extract_answer(response_text):
    if "Answer:" in response_text:
        answer = response_text.split("Answer:")[1].strip()
    else:
        answer = response_text.strip()
    return answer

print(extract_answer(response))