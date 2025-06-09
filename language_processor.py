import torch
from sentence_transformers import SentenceTransformer
import numpy as np

class LanguageProcessor:
    """언어 처리 클래스"""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"):
        self.device = device
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        print(f"✅ Language model loaded: {model_name}")
    
    def encode(self, text):
        """텍스트를 임베딩으로 변환"""
        if isinstance(text, str):
            text = [text]
        
        # 임베딩 생성
        embeddings = self.model.encode(text, convert_to_tensor=True)
        
        # GPU로 이동
        if self.device == "cuda":
            embeddings = embeddings.cuda()
        
        return embeddings
    
    def encode_batch(self, texts):
        """배치 텍스트 임베딩"""
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        
        if self.device == "cuda":
            embeddings = embeddings.cuda()
        
        return embeddings
    
    def get_embedding_dim(self):
        """임베딩 차원 반환"""
        return self.model.get_sentence_embedding_dimension()
    
    def compute_similarity(self, text1, text2):
        """두 텍스트 간 유사도 계산"""
        if self.model is None:
            return 0.5
        
        try:
            emb1 = self.encode(text1)
            emb2 = self.encode(text2)
            
            # 코사인 유사도 계산
            similarity = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
            return similarity.item()
        except Exception as e:
            print(f"Warning: Similarity computation failed: {e}")
            return 0.5
    
    def cleanup(self):
        """메모리 정리"""
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
