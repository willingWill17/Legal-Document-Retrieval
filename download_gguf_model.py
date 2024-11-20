from transformers import AutoModel, AutoTokenizer
# model_name = "gpustack/bge-reranker-v2-m3-GGUF"
# model_file = "bge-reranker-v2-m3-Q8_0.gguf" 
# model_path = hf_hub_download(model_name, filename=model_file)
# print(model_path)
model = AutoModel.from_pretrained("BAAI/bge-m3")