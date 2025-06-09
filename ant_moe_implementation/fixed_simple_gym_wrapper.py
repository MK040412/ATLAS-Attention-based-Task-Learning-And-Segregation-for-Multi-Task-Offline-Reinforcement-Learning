import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg, MySceneCfg
from config import CONFIG

class SimpleAntPushWrapper(gym.Wrapper):
    """SB3 호환 Gym Wrapper - 텐서/numpy 변환 문제 해결"""
    def __init__(self, custom_cfg=None):
        # IsaacLab 환경 생성
        cfg = AntEnvCfg()
        cfg.scene = MySceneCfg(num_envs=1, env_spacing=cfg.scene.env_spacing)
        cfg.sim.device = str(CONFIG.device)
        cfg.seed = 42
        
        self.isaac_env = ManagerBasedRLEnv(cfg=cfg)
        
        # Gym 환경으로 래핑
        super().__init__(self.isaac_env)
        
        # 커스텀 설정
        c = custom_cfg or {}
        self.cube_pos = np.array(c.get('cube_position', [2.0, 0.0, 0.1]), dtype=np.float32)
        self.goal_pos = np.array(c.get('goal_position', [3.0, 0.0, 0.1]), dtype=np.float32)
        
        # 실제 관측값 크기 확인
        print("🔍 Checking actual observation dimensions...")
        try:
            dummy_obs_dict, _ = self.isaac_env.reset()
            dummy_base_obs = self._safe_tensor_to_numpy(dummy_obs_dict['policy'])
            
            if dummy_base_obs.ndim == 2:
                dummy_base_obs = dummy_base_obs[0]
            
            actual_base_dim = len(dummy_base_obs.flatten())
            print(f"  Actual base observation dimension: {actual_base_dim}")
            
        except Exception as e:
            print(f"  Warning: Could not determine actual obs dim: {e}")
            actual_base_dim = 60  # 기본값
        
        # 관측 공간 재정의
        total_obs_dim = actual_base_dim + 6  # +3 for cube, +3 for goal
        
        self.observation_space = Box(
            low=-10.0, high=10.0,
            shape=(total_obs_dim,),
            dtype=np.float32
        )
        
        # 액션 공간 재정의
        original_action_space = self.isaac_env.action_space
        if hasattr(original_action_space, 'shape'):
            if len(original_action_space.shape) == 2:
                action_dim = original_action_space.shape[1]
            else:
                action_dim = original_action_space.shape[0]
        else:
            action_dim = 8
        
        self.action_space = Box(
            low=-1.0, high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
        
        print(f"🏗️ SB3 Compatible Wrapper created")
        print(f"  Obs space: {self.observation_space.shape}")
        print(f"  Action space: {self.action_space.shape}")
    
    def _safe_tensor_to_numpy(self, data):
        """안전한 텐서 -> numpy 변환"""
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data, dtype=np.float32)
    
    def _safe_numpy_to_tensor(self, data, device=None):
        """안전한 numpy -> 텐서 변환"""
        if torch.is_tensor(data):
            return data
        else:
            tensor = torch.from_numpy(np.array(data, dtype=np.float32))
            if device:
                tensor = tensor.to(device)
            return tensor
    
    def reset(self, **kwargs):
        """리셋"""
        try:
            obs_dict, info = self.isaac_env.reset()
            obs = self._process_obs(obs_dict)
            return obs, info
        except Exception as e:
            print(f"⚠️ Reset error: {e}")
            dummy_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            return dummy_obs, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """환경 스텝 (Boolean Tensor 문제 수정)"""
        self.current_step += 1
    
        # 액션 전처리
        action = self._preprocess_action(action)
    
        try:
            if hasattr(self, 'isaac_env') and self.isaac_env is not None:
                # IsaacLab 환경 사용
                obs_dict, reward, terminated, truncated, info = self.isaac_env.step(action)
            
                # Tensor를 스칼라/numpy로 안전하게 변환
                if hasattr(reward, 'item'):
                    reward = reward.item()
                elif hasattr(reward, 'cpu'):
                    reward = reward.cpu().numpy().item()
            
                if hasattr(terminated, 'item'):
                    terminated = terminated.item()
                elif hasattr(terminated, 'cpu'):
                    terminated = terminated.cpu().numpy().item()
            
                if hasattr(truncated, 'item'):
                    truncated = truncated.item()
                elif hasattr(truncated, 'cpu'):
                    truncated = truncated.cpu().numpy().item()
            
                # 관측값 처리
                next_obs = self._process_isaac_obs(obs_dict)
            
            else:
                # 시뮬레이션 모드
                next_obs, reward, terminated, truncated = self._simulate_step(action)
                info = {}
            
        except Exception as e:
            print(f"⚠️ Environment step error: {e}")
            # 폴백
            next_obs = self._generate_simulation_obs()
            reward = 0.0
            terminated = False
            truncated = False
            info = {}
    
        # 에피소드 길이 제한
        if self.current_step >= self.max_episode_steps:
            truncated = True
    
        # 보상 계산
        custom_reward = self._calculate_reward(next_obs)
        total_reward = float(reward) + custom_reward
    
        self._current_obs = next_obs
    
        info.update({
            'episode_step': self.current_step,
            'max_steps_reached': self.current_step >= self.max_episode_steps
        })
    
        # 모든 반환값이 올바른 타입인지 확인
        return (
            next_obs.astype(np.float32), 
            float(total_reward), 
            bool(terminated), 
            bool(truncated), 
            info
        )
    
    def _process_obs(self, obs_dict):
        """관측값 처리"""
        try:
            base_obs = self._safe_tensor_to_numpy(obs_dict['policy'])
            
            # 다중 환경 처리
            if base_obs.ndim == 2:
                base_obs = base_obs[0]
            
            # 1차원으로 평탄화
            base_obs = base_obs.flatten()
            
            # cube/goal 위치 추가
            cube_pos = self.cube_pos
            goal_pos = self.goal_pos
            
            # 전체 관측값 구성
            full_obs = np.concatenate([base_obs, cube_pos, goal_pos])
            
            # 크기 조정
            expected_size = self.observation_space.shape[0]
            actual_size = len(full_obs)
            
            if actual_size != expected_size:
                if actual_size < expected_size:
                    padding = np.zeros(expected_size - actual_size, dtype=np.float32)
                    full_obs = np.concatenate([full_obs, padding])
                else:
                    full_obs = full_obs[:expected_size]
            
            # 범위 제한
            full_obs = np.clip(full_obs, -10.0, 10.0)
            
            return full_obs.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Obs processing error: {e}")
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
    
    def _compute_reward(self, obs):
        """보상 계산"""
        try:
            if len(obs) >= 6:
                cube_pos = obs[-6:-3]
                goal_pos = obs[-3:]
                
                distance = np.linalg.norm(cube_pos - goal_pos)
                reward = -distance
                
                # 추가 보상: ant가 cube에 가까이 있으면 보너스
                if len(obs) >= 9:  # ant 위치가 있다면
                    ant_pos = obs[:3]  # 첫 3개 차원을 ant 위치로 가정
                    ant_to_cube_dist = np.linalg.norm(ant_pos - cube_pos)
                    reward += -0.1 * ant_to_cube_dist  # ant가 cube에 가까울수록 보너스
                
                reward = np.clip(reward, -10.0, 10.0)
                return reward
            else:
                return 0.0
                
        except Exception as e:
            print(f"⚠️ Reward computation error: {e}")
            return 0.0
    
    def close(self):
        """환경 종료"""
        try:
            self.isaac_env.close()
            print("✅ Environment closed")
        except Exception as e:
            print(f"⚠️ Close error: {e}")