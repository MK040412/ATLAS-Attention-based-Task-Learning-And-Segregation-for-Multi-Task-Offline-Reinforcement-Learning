#!/usr/bin/env python3
"""
병렬 환경 지원 Ant Push 래퍼 (IsaacLab 병렬화 활용)
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from typing import Dict, Any, Tuple, Optional, List
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg, MySceneCfg
from config import CONFIG

class VectorizedAntPushWrapper(gym.Env):
    """병렬 환경을 지원하는 Ant Push 래퍼"""
    
    def __init__(self, custom_cfg: Dict[str, Any] = None, num_envs: int = 64):
        super().__init__()
        
        self.num_envs = num_envs
        self.custom_cfg = custom_cfg or {}
        
        # IsaacLab 병렬 환경 생성
        cfg = AntEnvCfg()
        cfg.scene = MySceneCfg(num_envs=num_envs, env_spacing=cfg.scene.env_spacing)
        cfg.sim.device = str(CONFIG.device)
        cfg.seed = 42
        
        self.isaac_env = ManagerBasedRLEnv(cfg=cfg)
        
        # 초기 리셋으로 차원 확인
        obs_dict, _ = self.isaac_env.reset()
        base_obs = obs_dict['policy']
        
        # 단일 환경 차원 계산
        if len(base_obs.shape) == 2:  # [num_envs, obs_dim]
            single_obs_dim = base_obs.shape[1]
        else:
            single_obs_dim = base_obs.shape[0]
        
        # 큐브/목표 위치 추가
        self.single_obs_dim = single_obs_dim + 6
        
        # Gymnasium 공간 정의 (단일 환경 기준)
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.single_obs_dim,), 
            dtype=np.float32
        )
        
        # 액션 공간
        action_space = self.isaac_env.action_space
        if len(action_space.shape) == 2:  # [num_envs, action_dim]
            single_action_dim = action_space.shape[1]
        else:
            single_action_dim = action_space.shape[0]
        
        self.action_space = Box(
            low=-1.0, 
            high=1.0, 
            shape=(single_action_dim,), 
            dtype=np.float32
        )
        
        # 환경 상태
        self.current_step = 0
        self.max_episode_steps = 500
        self.active_env_idx = 0  # 현재 사용 중인 환경 인덱스
        
        # 큐브/목표 위치 (모든 환경에 동일하게 적용)
        self.cube_pos = np.array(self.custom_cfg.get('cube_position', [2.0, 0.0, 0.1]), dtype=np.float32)
        self.goal_pos = np.array(self.custom_cfg.get('goal_position', [3.0, 0.0, 0.1]), dtype=np.float32)
        
        print(f"🏗️ VectorizedAntPushWrapper created:")
        print(f"  • Parallel environments: {num_envs}")
        print(f"  • Single obs dim: {self.single_obs_dim}")
        print(f"  • Single action dim: {single_action_dim}")
        print(f"  • Using environment rotation for diversity")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """환경 리셋 (병렬 환경에서 하나 선택)"""
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        
        # 전체 병렬 환경 리셋
        obs_dict, info = self.isaac_env.reset()
        
        # 사용할 환경 인덱스 선택 (라운드 로빈 또는 랜덤)
        self.active_env_idx = np.random.randint(0, self.num_envs)
        
        # 선택된 환경의 관측값 추출
        single_obs = self._extract_single_obs(obs_dict, self.active_env_idx)
        
        info_single = {
            'cube_position': self.cube_pos.tolist(),
            'goal_position': self.goal_pos.tolist(),
            'episode_step': self.current_step,
            'active_env_idx': self.active_env_idx,
            'total_parallel_envs': self.num_envs
        }
        
        return single_obs, info_single
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """환경 스텝 (병렬 환경 활용)"""
        self.current_step += 1
        
        # 액션을 병렬 환경 형태로 확장
        parallel_actions = self._expand_action_to_parallel(action)
        
        try:
            # 병렬 환경에서 스텝 실행
            obs_dict, rewards, terminated, truncated, info = self.isaac_env.step(parallel_actions)
            
            # 활성 환경의 결과만 추출
            single_obs = self._extract_single_obs(obs_dict, self.active_env_idx)
            single_reward = self._extract_single_value(rewards, self.active_env_idx)
            single_terminated = self._extract_single_value(terminated, self.active_env_idx)
            single_truncated = self._extract_single_value(truncated, self.active_env_idx)
            
            # 커스텀 보상 추가
            custom_reward = self._calculate_reward(single_obs)
            total_reward = float(single_reward) + custom_reward
            
            # 에피소드 길이 제한
            if self.current_step >= self.max_episode_steps:
                single_truncated = True
            
            info_single = {
                'episode_step': self.current_step,
                'active_env_idx': self.active_env_idx,
                'parallel_rewards_mean': float(rewards.mean()) if hasattr(rewards, 'mean') else single_reward,
                'max_steps_reached': self.current_step >= self.max_episode_steps
            }
            
            return single_obs, float(total_reward), bool(single_terminated), bool(single_truncated), info_single
            
        except Exception as e:
            print(f"⚠️ Vectorized step error: {e}")
            # 폴백
            dummy_obs = np.random.randn(self.single_obs_dim).astype(np.float32)
            return dummy_obs, 0.0, False, False, {'error': str(e)}
    
    def _extract_single_obs(self, obs_dict: Dict, env_idx: int) -> np.ndarray:
        """병렬 관측값에서 단일 환경 관측값 추출"""
        base_obs = obs_dict['policy']
        
        # Tensor를 numpy로 변환
        if hasattr(base_obs, 'cpu'):
            base_obs = base_obs.cpu().numpy()
        
        # 특정 환경의 관측값 추출
        if len(base_obs.shape) == 2:  # [num_envs, obs_dim]
            single_base_obs = base_obs[env_idx]
        else:
            single_base_obs = base_obs
        
        # 큐브/목표 위치 추가
        full_obs = np.concatenate([single_base_obs.flatten(), self.cube_pos, self.goal_pos])
        
        return full_obs.astype(np.float32)
    
    def _extract_single_value(self, tensor_values, env_idx: int):
        """병렬 텐서에서 단일 값 추출"""
        if hasattr(tensor_values, 'cpu'):
            values = tensor_values.cpu().numpy()
        elif isinstance(tensor_values, torch.Tensor):
            values = tensor_values.detach().numpy()
        else:
            values = np.array(tensor_values)
        
        # 스칼라인 경우
        if values.ndim == 0:
            return values.item()
        
        # 벡터인 경우
        if values.ndim == 1 and len(values) > env_idx:
            return values[env_idx]
        
        # 기본값
        return values.flatten()[0] if len(values.flatten()) > 0 else 0
    
    def _expand_action_to_parallel(self, action: np.ndarray) -> torch.Tensor:
        """단일 액션을 병렬 환경용으로 확장"""
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action).to(CONFIG.device)
        
        # 차원 정리
        if action.dim() == 1:
            action = action.unsqueeze(0)  # [action_dim] -> [1, action_dim]
        
        # 모든 환경에 동일한 액션 적용 (또는 다양화 가능)
        parallel_actions = action.repeat(self.num_envs, 1)  # [num_envs, action_dim]
        
        return parallel_actions
    
    def _calculate_reward(self, obs: np.ndarray) -> float:
        """보상 계산"""
        if len(obs) < 6:
            return 0.0
        
        cube_pos = obs[-6:-3]
        goal_pos = obs[-3:]
        
        distance = np.linalg.norm(cube_pos - goal_pos)
        distance_reward = -distance
        
        if distance < 0.3:
            goal_reward = 10.0
        else:
            goal_reward = 0.0
        
        return distance_reward + goal_reward + 0.1
    
    def close(self):
        """환경 종료"""
        if hasattr(self, 'isaac_env'):
            self.isaac_env.close()

# 🔄 환경 로테이션을 통한 다양성 증대
class RotatingVectorizedWrapper(VectorizedAntPushWrapper):
    """환경 로테이션으로 다양성을 증대하는 래퍼"""
    
    def __init__(self, custom_cfg: Dict[str, Any] = None, num_envs: int = 64, 
                 rotation_frequency: int = 10):
        super().__init__(custom_cfg, num_envs)
        self.rotation_frequency = rotation_frequency
        self.episode_count = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """리셋 시 환경 로테이션"""
        self.episode_count += 1
        
        # 주기적으로 다른 환경 사용
        if self.episode_count % self.rotation_frequency == 0:
            self.active_env_idx = (self.active_env_idx + 1) % self.num_envs
            print(f"🔄 Rotated to environment {self.active_env_idx}")
        
        return super().reset(seed, options)

# 🎯 MoE 에이전트와의 통합
class ParallelMoETraining:
    """병렬 환경을 활용한 MoE 학습"""
    
    def __init__(self, num_envs: int = 64, num_experts: int = 8):
        self.num_envs = num_envs
        self.num_experts = num_experts
        
        # 병렬 환경 생성
        self.env = RotatingVectorizedWrapper(num_envs=num_envs)
        
        # 각 전문가별로 다른 환경 인덱스 할당
        self.expert_env_mapping = {}
        for i in range(num_experts):
            self.expert_env_mapping[i] = list(range(i, num_envs, num_experts))
    
    def collect_parallel_experience(self, agent, instruction_emb, steps_per_env: int = 100):
        """병렬로 경험 수집"""
        all_experiences = []
        
        for expert_id in range(self.num_experts):
            # 해당 전문가가 사용할 환경들
            env_indices = self.expert_env_mapping[expert_id]
            
            for env_idx in env_indices[:4]:  # 각 전문가당 4개 환경 사용
                self.env.active_env_idx = env_idx
                
                # 에피소드 실행
                state, _ = self.env.reset()
                
                for step in range(steps_per_env):
                    action, cluster_id = agent.select_action(state, instruction_emb)
                    next_state, reward, done, truncated, info = self.env.step(action)
                    
                    all_experiences.append({
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'next_state': next_state,
                        'done': done or truncated,
                        'cluster_id': cluster_id,
                        'env_idx': env_idx
                    })
                    
                    state = next_state
                    
                    if done or truncated:
                        state, _ = self.env.reset()
        
        return all_experiences