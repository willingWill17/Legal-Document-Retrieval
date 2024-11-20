import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Union
import re
from underthesea import word_tokenize

class PhoBERTEmbedding:
    def __init__(self, device: str = None):
        """
        Initialize PhoBERT embedder
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load PhoBERT-large model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")
        self.model = AutoModel.from_pretrained("vinai/phobert-large")
        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for PhoBERT
        Args:
            text: Input text
        Returns:
            Preprocessed text
        """
        # Lowercase the text
        text = text.lower()
        
        # Replace URLs with special token
        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
        
        # Replace emails with special token
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Vietnamese word tokenization using underthesea
        text = ' '.join(word_tokenize(text))
        
        return text.strip()

    @torch.no_grad()
    def get_embeddings(self, texts: Union[str, List[str]], pooling: str = 'mean') -> torch.Tensor:
        """
        Get embeddings for input texts
        Args:
            texts: Single text string or list of texts
            pooling: Pooling strategy ('mean' or 'cls')
        Returns:
            Tensor of embeddings
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Tokenize
        encoded = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get embeddings based on pooling strategy
        if pooling == 'cls':
            embeddings = outputs.last_hidden_state[:, 0]  # [CLS] token
        else:  # mean pooling
            # Compute mean of token embeddings, considering attention mask
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
        return embeddings

# Example usage
def main():
    # Initialize embedder
    embedder = PhoBERTEmbedding()
    
    # Example texts (Vietnamese)
    # texts = [
    #     'Quản lý, khai thác các công trình thủy lợi trình độ cao đẳng là ngành, nghề phục vụ tưới, tiêu, dân sinh, công nghiệp, nông nghiệp, an ninh quốc phòng, đáp ứng yêu cầu bậc 5 trong Khung trình độ quốc gia Việt Nam. Quản lý khai thác các công trình thủy lợi trình độ cao đẳng bao gồm các nhiệm vụ chính như: Quan trắc khí tượng thủy văn; trắc đạc công trình thủy lợi; quan trắc công trình thủy lợi; quản lý vận hành, khai thác tưới, cấp, tiêu và thoát nước; quản lý vận hành, khai thác công trình thủy lợi đầu mối; quản lý vận hành, khai thác kênh và công trình trên kênh; thi công tu bổ công trình thủy lợi; duy tu bảo dưỡng công trình thủy lợi; phòng chống lụt bão; lập, lưu trữ hồ sơ quản lý công trình; bảo vệ công trình thủy lợi; giám sát an toàn lao động và vệ sinh môi trường, Người hành nghề quản lý, khai thác công trình thủy lợi thường làm việc tại các doanh nghiệp quản lý, khai thác công trình thủy lợi, doanh nghiệp khai thác tài nguyên nước... họ cần có đủ kiến thức, kỹ năng, sức khỏe để làm việc ở văn phòng, công trình hoặc ngoài trời, đôi khi phải làm việc trong những điều kiện khắc nghiệt như gió bão, lũ lụt…',
    #     'Người học ngành quản lý khai thác công trình thủy lợi trình độ cao đẳng phải hoàn thành tối thiểu bao nhiêu tín chỉ mới có thể tốt nghiệp?' ,
    # ]

    texts = ['con người rất hài hước',
             'mình thích học toán'
    ]
    
    # Get embeddings
    embeddings = embedder.get_embeddings(texts)
    
    # Print embedding shapes
    print(f"Embedding shape: {embeddings.shape}")
    
    # Example of computing similarity between sentences
    from torch.nn.functional import cosine_similarity
    sim = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
    print(f"Similarity between first two sentences: {sim.item():.4f}")

if __name__ == "__main__":
    main()