# import torch
from tqdm import tqdm
import pandas as pd
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from langchain_huggingface import HuggingFacePipeline
# from langchain_elasticsearch import ElasticsearchStore
from pymilvus import connections, Collection
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from pymilvus import connections, Collection
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_chroma import Chroma

from uuid import uuid4

# Setup database
URI = "http://192.168.80.2:9200"

es = Elasticsearch(URI)
connections.connect(alias="default", uri="./corpus_2.db")
collection = Collection("corpus4")
collection.load()

embeddings = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder", model_kwargs={'device': 'cuda'})

# Setup model
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     return_dict=True,
#     low_cpu_mem_usage=True,
#     torch_dtype=torch.float16,
#     device_map=0
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# pipe = pipeline(
#     'text-generation',
#     model=model,
#     tokenizer=tokenizer,
#     max_length=3000,
#     truncation=True,
#     device_map=0
# )
# llm = HuggingFacePipeline(pipeline=pipe)

# Main query
# query = "Người học ngành quản lý khai thác công trình thủy lợi trình độ cao đẳng phải có khả năng học tập và nâng cao trình độ như thế nào?"

def find_docs(query):
    feat_arr = embeddings.embed_query(query)
    search_param = {
        "data": [feat_arr],
        "output_fields": ['cid'],
        "anns_field": "embeddings",
        "param": {"metric_type": "COSINE"},
        "limit": 20,
        "consistency_level": "Eventually",
    }
    res = collection.search(**search_param)

    documents = []           
    for hits in res:
        for hit in hits:
            cid = hit.get("cid")
            res_text = es.search(index="corpus2", query={
                'match' : {
                    'cid': {
                        'query': cid,
                        }
                    }
                }
            )
            documents.append(Document(
                page_content=res_text["hits"]["hits"][0]['_source']['context'],
                metadata={"cid": cid},
            )
    ) 
    return documents
# uuids = [str(uuid4()) for _ in range(len(documents))]

# # Retriever
# # retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})
# vectorstore = Chroma.from_documents(
#     documents,
#     embedding=embeddings,
# )
# retriever = vectorstore.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 15},
# )

# Reranker
reranker_model = HuggingFaceCrossEncoder(model_name="itdainb/PhoRanker", model_kwargs={'device':'cuda'})
compressor = CrossEncoderReranker(model=reranker_model, top_n=10)

# Prepare prompt
# system_message_template = SystemMessagePromptTemplate.from_template(
#     "Using the information contained in the context, give a comprehensive answer to the question. "
#     "Respond only to the question asked, response should be concise and relevant to the question. "
#     "If the answer cannot be deduced from the context, do not give an answer."
# )

system_message_template = SystemMessagePromptTemplate.from_template(
    "Using the information contained in the context, arrange the documents based on their relevance to the question."
    "Response contains the cid numbers of the documents, seperated by a space"
    "If the answer cannot be deduced from the context, do not give an answer."
)

# Define the user message template
user_message_template = HumanMessagePromptTemplate.from_template(
    "Context: {context}\n"
    "---\n"
    "Now here is the question you need to answer.\n"
    "Question: {question}"
)
prompt = ChatPromptTemplate.from_messages(
    [system_message_template, user_message_template]
)

# Helper functions
def format_docs(docs):
    return "\n\n".join(f"CID: {doc.metadata['cid']}\n{doc.page_content}" for doc in docs)

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

def print_cid(qid, docs):
    return str(qid) + " " + " ".join(str(doc.metadata['cid']) for doc in docs)

# Prepare test csv
df = pd.read_csv("../dataset/public_test.csv")

# Testing reranker
file = open('predict_phoranker.txt', 'w+', encoding='utf-8')
for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing query"):
    documents = find_docs(row["question"])
    compressed_docs = compressor.compress_documents(query=row["question"], documents=documents)
    ans = print_cid(row["qid"], compressed_docs)
    print(row["question"])
    print(ans)
    file.write(ans)
    file.write("\n")
file.close()


# Main RAG chain
# rag_chain = (
#     {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# res = rag_chain.invoke(query)
# print(res)