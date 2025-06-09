import torch
import torch.nn as nn
import numpy as np
from config import CONFIG

class DummyVisionSystem:
    """더미 비전 시스템 (메모리 절약용)"""
    
    def __init__(self, device="cuda"):
        self.device = device
        print("Dummy Vision System initialized (vision disabled)")
    
    def encode_image(self, image):
        """더미 이미지 인코딩"""
        return np.zeros(512, dtype=np.float32)
    
    def encode_text(self, text):
        """더미 텍스트 인코딩"""
        return np.zeros(512, dtype=np.float32)
    
    def compute_similarity(self, image_features, text_features):
        """더미 유사도 계산"""
        return 0.5
    
    def cleanup(self):
        """정리"""
        pass

def get_vision_system(device="cuda"):
    """비전 시스템 팩토리 함수"""
    if CONFIG.use_vision:
        try:
            # 실제 CLIP 모델 로드 시도
            from transformers import CLIPProcessor, CLIPModel
            
            class LightweightCLIPVision:
                def __init__(self, device):
                    self.device = device
                    self.model = CLIPModel.from_pretrained(CONFIG.clip_model_name).to(device)
                    self.processor = CLIPProcessor.from_pretrained(CONFIG.clip_model_name)
                    self.model.eval()
                    print(f"CLIP Vision System loaded: {CONFIG.clip_model_name}")
                
                def encode_image(self, image):
                    with torch.no_grad():
                        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                        image_features = self.model.get_image_features(**inputs)
                        return image_features.cpu().numpy().squeeze()
                
                def encode_text(self, text):
                    with torch.no_grad():
                        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
                        text_features = self.model.get_text_features(**inputs)
                        return text_features.cpu().numpy().squeeze()
                
                def compute_similarity(self, image_features, text_features):
                    similarity = np.dot(image_features, text_features) / (
                        np.linalg.norm(image_features) * np.linalg.norm(text_features)
                    )
                    return float(similarity)
                
                def cleanup(self):
                    del self.model
                    del self.processor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            return LightweightCLIPVision(device)
            
        except Exception as e:
            print(f"Warning: Failed to load CLIP model: {e}")
            print("Using dummy vision system instead")
            return DummyVisionSystem(device)
    else:
        return DummyVisionSystem(device)