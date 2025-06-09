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

        # Spawn cube as RigidObject using CuboidCfg
        try:
            spawn_cfg = sim_utils.CuboidCfg(
                size=(0.2,0.2,0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,0.0,1.0), metallic=0.1)
            )
            cube_cfg = RigidObjectCfg(
                prim_path=self.task_path + "/Cube",
                spawn=spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(),
            )
            # Create and initialize rigid object
            self.cube = RigidObject(cfg=cube_cfg)
            print("✓ Cube object created successfully")
        except Exception as cube_error:
            print(f"Warning: Cube creation failed: {cube_error}")
            self.cube = None

        # Redefine observation space: policy obs + cube_pos(3) + goal_pos(3)
        try:
            base_shape = self.observation_space['policy'].shape
            base_dim   = int(np.prod(base_shape))
            full_dim   = base_dim + 3 + 3
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(full_dim,), dtype=np.float32
            )
            print(f"✓ Observation space: {self.observation_space.shape}")
        except Exception as obs_error:
            print(f"Warning: Observation space setup failed: {obs_error}")
            # 기본 관측 공간 설정
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(252,), dtype=np.float32  # 추정값
            )

    def get_action_dim(self):
        """행동 차원 반환"""
        try:
            if hasattr(self.action_space, 'shape'):
                if len(self.action_space.shape) == 2:
                    return self.action_space.shape[1]  # (num_envs, action_dim)
                else:
                    return self.action_space.shape[0]  # (action_dim,)
            else:
                return 8  # Ant의 기본 행동 차원
        except Exception as e:
            print(f"Warning: Failed to get action dim: {e}")
            return 8

    def _safe_tensor_to_numpy(self, tensor):
        """텐서를 numpy 배열로 안전하게 변환"""
        try:
            if torch.is_tensor(tensor):
                return tensor.detach().cpu().numpy().astype(np.float32)
            elif isinstance(tensor, np.ndarray):
                return tensor.astype(np.float32)
            else:
                return np.array(tensor, dtype=np.float32)
        except Exception as e:
            print(f"Warning: Tensor conversion failed: {e}")
            return np.zeros(3, dtype=np.float32)

    def _get_cube_position(self):
        """큐브 위치를 안전하게 가져오기"""
        try:
            if self.cube is not None and hasattr(self.cube, 'data'):
                cube_pos_tensor = self.cube.data.root_pos_w[0]
                cube_pos = self._safe_tensor_to_numpy(cube_pos_tensor)[:3]
                return cube_pos
            else:
                # 큐브가 없으면 기본 위치 반환
                return self.cube_position.copy()
        except Exception as e:
            print(f"Warning: Failed to get cube position: {e}")
            return self.cube_position.copy()

    def _merge_obs(self, obs_dict):
        """관측값 병합"""
        try:
            base = obs_dict['policy']
            
            # 텐서를 numpy로 변환
            if torch.is_tensor(base):
                base = self._safe_tensor_to_numpy(base)
            
            # 큐브 위치 가져오기
            cube_pos = self._get_cube_position()
            
            # 모든 것을 numpy 배열로 변환
            base = np.array(base, dtype=np.float32)
            cube_pos = np.array(cube_pos, dtype=np.float32)
            goal_pos = np.array(self.goal_position, dtype=np.float32)
            
            # 1차원으로 평탄화
            if base.ndim > 1:
                base = base.flatten()
            if cube_pos.ndim > 1:
                cube_pos = cube_pos.flatten()[:3]
            if goal_pos.ndim > 1:
                goal_pos = goal_pos.flatten()[:3]
            
            merged = np.concatenate([base, cube_pos, goal_pos], axis=0)
            return merged.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Observation merge failed: {e}")
            # 안전한 기본값 반환
            dummy_obs = np.zeros(246, dtype=np.float32)  # 추정 크기
            return dummy_obs

    def reset(self):
        """환경 리셋"""
        try:
            obs_dict, info = super().reset()
            merged = self._merge_obs(obs_dict)
            return merged, info
        except Exception as e:
            print(f"Warning: Reset failed: {e}")
            # 안전한 기본값 반환
            dummy_obs = np.zeros(246, dtype=np.float32)
            return dummy_obs, {}

    def step(self, action):
        """환경 스텝"""
        try:
            # action을 적절한 형태로 변환
            if isinstance(action, np.ndarray):
                if action.ndim == 1:
                    # 단일 환경용 action을 다중 환경용으로 확장
                    action = action.reshape(1, -1)
            elif torch.is_tensor(action):
                action = self._safe_tensor_to_numpy(action)
                if action.ndim == 1:
                    action = action.reshape(1, -1)
            
            # 환경 스텝 실행
            obs_dict, _, term, trunc, info = super().step(action)
            
            # 관측값 병합
            obs = self._merge_obs(obs_dict)
            
            # 보상 계산
            reward = self._calculate_reward(obs)
            done = bool(term or trunc)
            
            return obs, reward, done, info
            
        except Exception as e:
            print(f"Warning: Step failed: {e}")
            # 안전한 기본값 반환
            dummy_obs = np.zeros(246, dtype=np.float32)
            return dummy_obs, -1.0, True, {}

    def _calculate_reward(self, obs):
        """보상 계산"""
        try:
            # 관측값에서 위치 정보 추출
            base_obs_dim = len(obs) - 6
            ant_base = obs[:3]  # 첫 3개 요소를 ant 위치로 가정
            cube_pos = obs[base_obs_dim:base_obs_dim+3]
            goal_pos = obs[base_obs_dim+3:base_obs_dim+6]
            
            # 거리 기반 보상
            r_push = -np.linalg.norm(cube_pos - goal_pos)
            r_near = -np.linalg.norm(ant_base - cube_pos)
            
            return float(r_push + r_near)
            
        except Exception as e:
            print(f"Warning: Reward calculation failed: {e}")
            return -1.0

    def close(self):
        """환경 정리"""
        try:
            print("🔄 Closing environment...")
            super().close()
            print("✓ Environment closed safely")
        except Exception as e:
            print(f"Warning: Environment close failed: {e}")