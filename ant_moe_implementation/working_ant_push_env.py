import numpy as np
import torch
from gymnasium.spaces import Box
from isaaclab.envs import ManagerBasedRLEnv
from config import CONFIG
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg, MySceneCfg

class WorkingAntPushEnv(ManagerBasedRLEnv):
    def __init__(self, custom_cfg=None, num_envs: int = 1):
        print(f"🏗️ Creating Working Ant Push Environment")
        
        cfg = AntEnvCfg()
        cfg.scene = MySceneCfg(num_envs=num_envs, env_spacing=cfg.scene.env_spacing)
        cfg.sim.device = str(CONFIG.device)
        super().__init__(cfg=cfg)

        c = custom_cfg or {}
        self.cube_position = np.array(c.get('cube_position', [2.0,0.0,0.1]), dtype=np.float32)
        self.goal_position = np.array(c.get('goal_position', [3.0,0.0,0.1]), dtype=np.float32)

        # 관측 공간 재정의
        base_shape = self.observation_space['policy'].shape
        base_dim = int(np.prod(base_shape))
        full_dim = base_dim + 6  # +cube_pos(3) +goal_pos(3)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(full_dim,), dtype=np.float32
        )
        print(f"✅ Observation space: {self.observation_space.shape}")

    def get_action_dim(self):
        if hasattr(self.action_space, 'shape'):
            return self.action_space.shape[1] if len(self.action_space.shape) == 2 else self.action_space.shape[0]
        return 8

    def _merge_obs(self, obs_dict):
        base = obs_dict['policy']
        if torch.is_tensor(base):
            base = base.detach().cpu().numpy()
        
        base = np.array(base, dtype=np.float32).flatten()
        cube_pos = self.cube_position.copy()
        goal_pos = self.goal_position.copy()
        
        return np.concatenate([base, cube_pos, goal_pos], axis=0).astype(np.float32)

    def reset(self):
        obs_dict, info = super().reset()
        return self._merge_obs(obs_dict), info

    def step(self, action):
        # 액션을 텐서로 변환
        if isinstance(action, np.ndarray):
            if action.ndim == 1:
                action = action.reshape(1, -1)
            action_tensor = torch.from_numpy(action).to(device=self.device, dtype=torch.float32)
        else:
            action_tensor = action
        
        try:
            obs_dict, _, term, trunc, info = super().step(action_tensor)
            obs = self._merge_obs(obs_dict)
            
            # 간단한 보상
            ant_pos = obs[:3]
            cube_pos = obs[-6:-3]
            goal_pos = obs[-3:]
            
            reward = -np.linalg.norm(cube_pos - goal_pos) - 0.1 * np.linalg.norm(ant_pos - cube_pos)
            done = bool(term or trunc)
            
            return obs, float(reward), done, info
        except:
            return np.zeros_like(self._merge_obs({'policy': np.zeros(60)})), -1.0, True, {}

    def close(self):
        try:
            super().close()
        except:
            pass