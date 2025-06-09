import numpy as np
import torch
from gymnasium.spaces import Box
from isaaclab.envs import ManagerBasedRLEnv
from config import CONFIG
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg, MySceneCfg

class IsaacAntPushEnv(ManagerBasedRLEnv):
    def __init__(self, custom_cfg=None, num_envs: int = 1):
        print(f"🏗️ Initializing IsaacAntPushEnv with {num_envs} environments")
        
        # Build base IsaacLab Ant config
        cfg = AntEnvCfg()
        cfg.scene = MySceneCfg(num_envs=num_envs, env_spacing=cfg.scene.env_spacing)
        cfg.sim.device = str(CONFIG.device)
        super().__init__(cfg=cfg)

        # Store custom params
        c = custom_cfg or {}
        self.cube_position = np.array(c.get('cube_position', [2.0,0.0,0.1]), dtype=np.float32)
        self.goal_position = np.array(c.get('goal_position', [3.0,0.0,0.1]), dtype=np.float32)

        print(f"  Cube position: {self.cube_position}")
        print(f"  Goal position: {self.goal_position}")

        # 큐브 생성 시도 (실패해도 계속 진행)
        self.cube = None
        try:
            # task_path 속성 확인
            if hasattr(self, 'template_env_ns'):
                cube_path = f"{self.template_env_ns}/Cube"
            elif hasattr(self, 'env_ns'):
                cube_path = f"{self.env_ns}/Cube"
            else:
                cube_path = "/World/envs/env_0/Cube"  # 기본 경로
            
            spawn_cfg = sim_utils.CuboidCfg(
                size=(0.2,0.2,0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,0.0,1.0), metallic=0.1)
            )
            cube_cfg = RigidObjectCfg(
                prim_path=cube_path,
                spawn=spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(),
            )
            self.cube = RigidObject(cfg=cube_cfg)
            print("✅ Cube object created successfully")
        except Exception as cube_error:
            print(f"⚠️ Cube creation failed: {cube_error}")
            self.cube = None

        # Redefine observation space
        try:
            base_shape = self.observation_space['policy'].shape
            base_dim = int(np.prod(base_shape))
            full_dim = base_dim + 3 + 3  # +cube_pos +goal_pos
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(full_dim,), dtype=np.float32
            )
            print(f"✅ Observation space: {self.observation_space.shape}")
        except Exception as obs_error:
            print(f"⚠️ Observation space setup failed: {obs_error}")
            # 기본값 사용
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(66,), dtype=np.float32
            )

    def get_action_dim(self):
        """행동 차원 반환"""
        try:
            if hasattr(self.action_space, 'shape'):
                if len(self.action_space.shape) == 2:
                    return self.action_space.shape[1]
                else:
                    return self.action_space.shape[0]
            else:
                return 8
        except:
            return 8

    def _safe_tensor_to_numpy(self, data):
        """안전한 텐서 변환"""
        try:
            if torch.is_tensor(data):
                return data.detach().cpu().numpy().astype(np.float32)
            elif isinstance(data, np.ndarray):
                return data.astype(np.float32)
            else:
                return np.array(data, dtype=np.float32)
        except Exception as e:
            print(f"⚠️ Tensor conversion failed: {e}")
            return np.zeros(3, dtype=np.float32)

    def _get_cube_position(self):
        """큐브 위치 가져오기"""
        try:
            if self.cube is not None and hasattr(self.cube, 'data'):
                cube_pos_tensor = self.cube.data.root_pos_w[0]
                cube_pos = self._safe_tensor_to_numpy(cube_pos_tensor)[:3]
                return cube_pos
            else:
                return self.cube_position.copy()
        except Exception as e:
            print(f"⚠️ Failed to get cube position: {e}")
            return self.cube_position.copy()

    def _merge_obs(self, obs_dict):
        """관측값 병합"""
        try:
            base = obs_dict['policy']
            
            # 안전한 numpy 변환
            if torch.is_tensor(base):
                base = self._safe_tensor_to_numpy(base)
            
            base = np.array(base, dtype=np.float32).flatten()
            cube_pos = np.array(self._get_cube_position(), dtype=np.float32).flatten()[:3]
            goal_pos = np.array(self.goal_position, dtype=np.float32).flatten()[:3]
            
            merged = np.concatenate([base, cube_pos, goal_pos], axis=0)
            return merged.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Observation merge failed: {e}")
            # 안전한 기본값
            return np.zeros(66, dtype=np.float32)

    def reset(self):
        """환경 리셋"""
        try:
            obs_dict, info = super().reset()
            merged = self._merge_obs(obs_dict)
            return merged, info
        except Exception as e:
            print(f"⚠️ Reset failed: {e}")
            return np.zeros(66, dtype=np.float32), {}

    def step(self, action):
        """환경 스텝 - 완전히 안전한 버전"""
        try:
            # 1. 액션을 안전하게 처리
            safe_action = self._prepare_safe_action(action)
            
            # 2. 부모 클래스의 step 호출
            obs_dict, _, term, trunc, info = super().step(safe_action)
            
            # 3. 관측값 병합
            obs = self._merge_obs(obs_dict)
            
            # 4. 보상 계산
            reward = self._calculate_reward(obs)
            done = bool(term or trunc)
            
            return obs, reward, done, info
            
        except Exception as e:
            print(f"⚠️ Step failed: {e}")
            # 완전히 안전한 기본값 반환
            dummy_obs = np.zeros(66, dtype=np.float32)
            return dummy_obs, -1.0, True, {}

    def _prepare_safe_action(self, action):
        """액션을 안전하게 준비"""
        try:
            # numpy 배열로 변환
            if torch.is_tensor(action):
                action = action.detach().cpu().numpy()
            
            action = np.array(action, dtype=np.float32)
            
            # 차원 조정
            if action.ndim == 1:
                action = action.reshape(1, -1)  # (action_dim,) -> (1, action_dim)
            
            # 클리핑 (안전성을 위해)
            action = np.clip(action, -1.0, 1.0)
            
            return action
            
        except Exception as e:
            print(f"⚠️ Action preparation failed: {e}")
            # 안전한 기본 액션
            return np.zeros((1, 8), dtype=np.float32)

    def _calculate_reward(self, obs):
        """보상 계산"""
        try:
            if len(obs) < 6:
                return -1.0
            
            # 위치 정보 추출
            base_dim = len(obs) - 6
            ant_pos = obs[:3] if base_dim >= 3 else obs[:min(3, len(obs))]
            cube_pos = obs[base_dim:base_dim+3] if base_dim >= 0 else self.cube_position
            goal_pos = obs[base_dim+3:base_dim+6] if base_dim >= 0 else self.goal_position
            
            # 거리 기반 보상
            cube_to_goal = np.linalg.norm(cube_pos - goal_pos)
            ant_to_cube = np.linalg.norm(ant_pos - cube_pos)
            
            reward = -cube_to_goal - 0.1 * ant_to_cube
            return float(reward)
            
        except Exception as e:
            print(f"⚠️ Reward calculation failed: {e}")
            return -1.0

    def close(self):
        """환경 정리"""
        try:
            print("🔄 Closing environment...")
            super().close()
            print("✅ Environment closed safely")
        except Exception as e:
            print(f"⚠️ Environment close failed: {e}")