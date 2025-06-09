import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import cv2

class GLIPCubeDetector:
    """GLIP 기반 파란 큐브 감지기"""
    
    def __init__(self, model_name="IDEA-Research/grounding-dino-tiny", confidence_threshold=0.3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        
        try:
            print(f"🔍 Loading GLIP model: {model_name}")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("✅ GLIP model loaded successfully")
            self.model_loaded = True
            
        except Exception as e:
            print(f"⚠️ GLIP model loading failed: {e}")
            print("🔄 Falling back to simple color detection")
            self.model_loaded = False
    
    def detect_cube(self, rgb_image, text_prompt="blue cube"):
        """
        GLIP으로 큐브 감지
        
        Args:
            rgb_image: numpy array (H, W, 3)
            text_prompt: 감지할 객체 설명
            
        Returns:
            boxes: List of [x1, y1, x2, y2] bounding boxes
            scores: List of confidence scores
        """
        if not self.model_loaded:
            return self._fallback_color_detection(rgb_image)
        
        try:
            # numpy를 PIL Image로 변환
            pil_image = Image.fromarray(rgb_image)
            
            # GLIP 입력 준비
            inputs = self.processor(
                images=pil_image, 
                text=text_prompt, 
                return_tensors="pt"
            ).to(self.device)
            
            # 추론
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 결과 후처리
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.confidence_threshold,
                text_threshold=self.confidence_threshold,
                target_sizes=[pil_image.size[::-1]]  # (height, width)
            )[0]
            
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            
            return boxes, scores
            
        except Exception as e:
            print(f"GLIP detection error: {e}")
            return self._fallback_color_detection(rgb_image)
    
    def _fallback_color_detection(self, rgb_image):
        """색상 기반 감지 (GLIP 실패 시 대안)"""
        try:
            # RGB를 BGR로 변환
            bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            
            # 파란색 범위
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            
            # 마스크 생성
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            boxes = []
            scores = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # 최소 크기 필터
                    x, y, w, h = cv2.boundingRect(contour)
                    boxes.append([x, y, x + w, y + h])
                    scores.append(0.8)  # 고정 신뢰도
            
            return np.array(boxes), np.array(scores)
            
        except Exception as e:
            print(f"Fallback detection error: {e}")
            return np.array([]), np.array([])
    
    def get_best_detection(self, rgb_image, depth_image, camera_intrinsics):
        """
        최고 신뢰도의 감지 결과를 3D 좌표로 변환
        
        Args:
            rgb_image: RGB 이미지
            depth_image: 깊이 이미지
            camera_intrinsics: (fx, fy, cx, cy)
            
        Returns:
            world_pos: 3D 월드 좌표 [x, y, z] 또는 None
        """
        boxes, scores = self.detect_cube(rgb_image)
        
        if len(boxes) == 0:
            return None
        
        # 가장 높은 신뢰도의 박스 선택
        best_idx = np.argmax(scores)
        best_box = boxes[best_idx]
        
        # 박스 중심점 계산
        x1, y1, x2, y2 = best_box
        center_u = int((x1 + x2) / 2)
        center_v = int((y1 + y2) / 2)
        
        # 깊이 값 가져오기
        if (0 <= center_v < depth_image.shape[0] and 
            0 <= center_u < depth_image.shape[1]):
            depth = depth_image[center_v, center_u]
        else:
            depth = 2.0  # 기본값
        
        # 3D 좌표 계산
        fx, fy, cx, cy = camera_intrinsics
        X = (center_u - cx) * depth / fx
        Y = (center_v - cy) * depth / fy
        Z = depth
        
        return np.array([X, Y, Z], dtype=np.float32)