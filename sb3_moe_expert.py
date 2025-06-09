import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv
from typing import Optional, Dict, Any
import tempfile
import os

class SB3SACExpert:
    """Stable-Baselines3 기반 SAC 전문가"""
    
    def __init__(self, env: GymEnv, expert_id: int = 0, 
                 learning_rate: float = 3e-4, buffer_size: int = 10000):
        self.expert_id = expert_id
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # SB3 SAC 모델 생성
        self.model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=100,
            batch_size=64,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(
                net_arch=[128, 128]
            ),
            verbose=0,
            device=self.device
        )
        
        # 경험 저장용
        self.experiences = []
        self.update_count = 0
        
        print(f"🤖 SB3 SAC Expert {expert_id} created")
        print(f"  Device: {self.device}")
        print(f"  Buffer size: {buffer_size}")
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """행동 선택"""
        try:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            return action
        except Exception as e:
            print(f"⚠️ Expert {self.expert_id} action selection error: {e}")
            # 랜덤 액션 반환
            return self.env.action_space.sample()
    
    def store_experience(self, obs, action, reward, next_obs, done):
        """경험 저장"""
        self.experiences.append({
            'obs': np.array(obs, dtype=np.float32),
            'action': np.array(action, dtype=np.float32),
            'reward': float(reward),
            'next_obs': np.array(next_obs, dtype=np.float32),
            'done': bool(done)
        })
    
    def update(self, batch_size: int = 64) -> Optional[Dict[str, float]]:
        """모델 업데이트"""
        if len(self.experiences) < batch_size:
            return None
        
        try:
            # 경험을 SB3 버퍼에 추가
            for exp in self.experiences[-batch_size:]:
                self.model.replay_buffer.add(
                    exp['obs'], exp['next_obs'], exp['action'], 
                    exp['reward'], exp['done'], [{}]
                )
            
            # 모델 업데이트 (SB3 내부 메서드 사용)
            if self.model.replay_buffer.size() >= self.model.learning_starts:
                self.model.train(gradient_steps=1, batch_size=batch_size)
                self.update_count += 1
                
                return {
                    'expert_id': self.expert_id,
                    'update_count': self.update_count,
                    'buffer_size': self.model.replay_buffer.size()
                }
            
        except Exception as e:
            print(f"⚠️ Expert {self.expert_id} update error: {e}")
        
        return None
    
    def get_info(self) -> Dict[str, Any]:
        """전문가 정보 반환"""
        return {
            'expert_id': self.expert_id,
            'update_count': self.update_count,
            'buffer_size': self.model.replay_buffer.size() if self.model.replay_buffer else 0,
            'experiences_stored': len(self.experiences)
        }
    
    def save(self, filepath: str):
        """모델 저장"""
        try:
            self.model.save(filepath)
            # 추가 정보 저장
            torch.save({
                'expert_id': self.expert_id,
                'update_count': self.update_count,
                'experiences_count': len(self.experiences)
            }, filepath + "_info.pt")
            print(f"💾 Expert {self.expert_id} saved to {filepath}")
        except Exception as e:
            print(f"⚠️ Expert {self.expert_id} save error: {e}")
    
    def load(self, filepath: str):
        """모델 로드"""
        try:
            self.model = SAC.load(filepath, env=self.env)
            # 추가 정보 로드
            info_path = filepath + "_info.pt"
            if os.path.exists(info_path):
                info = torch.load(info_path)
                self.expert_id = info.get('expert_id', self.expert_id)
                self.update_count = info.get('update_count', 0)
            print(f"📂 Expert {self.expert_id} loaded from {filepath}")
        except Exception as e:
            print(f"⚠️ Expert {self.expert_id} load error: {e}")