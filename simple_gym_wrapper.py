import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg, MySceneCfg
from config import CONFIG

class SimpleAntPushWrapper(gym.Wrapper):
    """가장 간단한 Gym Wrapper"""
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
        
        # 관측 공간 재정의 (policy obs + cube_pos + goal_pos)
        base_obs_space = self.isaac_env.observation_space['policy']
        base_dim = base_obs_space.shape[0]
        
        self.observation_space = Box(
            low=-np.inf, high=np.inf, 
            shape=(base_dim + 6,),  # +3 for cube, +3 for goal
            dtype=np.float32
        )
        
        # 액션 공간은 그대로 사용
        self.action_space = self.isaac_env.action_space
        
        print(f"🏗️ Simple Ant Push Wrapper created")
        print(f"  Obs space: {self.observation_space.shape}")
        print(f"  Action space: {self.action_space.shape}")
    
    def reset(self, **kwargs):
        """리셋"""
        obs_dict, info = self.isaac_env.reset()
        obs = self._process_obs(obs_dict)
        return obs, info
    
    def step(self, action):
        """스텝"""
        # action을 올바른 형태로 변환
        if isinstance(action, np.ndarray) and action.ndim == 1:
            action = action.reshape(1, -1)
        
        obs_dict, reward, terminated, truncated, info = self.isaac_env.step(action)
        obs = self._process_obs(obs_dict)
        
        # 커스텀 보상 계산
        custom_reward = self._compute_reward(obs)
        
        # 텐서를 스칼라로 변환
        if hasattr(terminated, 'item'):
            terminated = terminated.item()
        if hasattr(truncated, 'item'):
            truncated = truncated.item()
        
        return obs, float(custom_reward), bool(terminated), bool(truncated), info
    
    def _process_obs(self, obs_dict):
        """관측값 처리"""
        base_obs = obs_dict['policy']
        
        # 텐서를 numpy로 변환
        if hasattr(base_obs, 'cpu'):
            base_obs = base_obs.cpu().numpy()
        
        # 다중 환경 처리
        if base_obs.ndim == 2:
            base_obs = base_obs[0]  # 첫 번째 환경만 사용
        
        # 더미 cube/goal 위치 (실제로는 환경에서 가져와야 함)
        cube_pos = self.cube_pos
        goal_pos = self.goal_pos
        
        # 연결
        full_obs = np.concatenate([base_obs.flatten(), cube_pos, goal_pos])
        return full_obs.astype(np.float32)
    
    def _compute_reward(self, obs):
        """간단한 보상 함수"""
        # 마지막 6개 차원이 cube_pos(3) + goal_pos(3)
        cube_pos = obs[-6:-3]
        goal_pos = obs[-3:]
        
        # 거리 기반 보상
        distance = np.linalg.norm(cube_pos - goal_pos)
        reward = -distance  # 거리가 가까울수록 높은 보상
        
        return reward
    
    def close(self):
        """환경 종료"""
        self.isaac_env.close()