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
        # Build base IsaacLab Ant config
        cfg = AntEnvCfg()
        cfg.scene = MySceneCfg(num_envs=num_envs, env_spacing=cfg.scene.env_spacing)
        cfg.sim.device = str(CONFIG.device)
        super().__init__(cfg=cfg)

        # Store custom params
        c = custom_cfg or {}
        self.cube_position = np.array(c.get('cube_position', [2.0,0.0,0.1]), dtype=np.float32)
        self.goal_position = np.array(c.get('goal_position', [3.0,0.0,0.1]), dtype=np.float32)
        
        # 큐브 초기화 플래그
        self.cube_initialized = False
        self.cube = None

        # Redefine observation space: policy obs + cube_pos(3) + goal_pos(3)
        base_shape = self.observation_space['policy'].shape
        base_dim   = int(np.prod(base_shape))
        full_dim   = base_dim + 3 + 3
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(full_dim,), dtype=np.float32
        )
        
        # 액션 차원 저장
        self._action_dim = self.action_space.shape[1] if len(self.action_space.shape) == 2 else self.action_space.shape[0]

    def _setup_cube(self):
        """큐브 설정 (한 번만 실행)"""
        if self.cube_initialized:
            return
            
        try:
            print("🔧 Setting up cube...")
            
            # Spawn cube as RigidObject using CuboidCfg
            spawn_cfg = sim_utils.CuboidCfg(
                size=(0.2,0.2,0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,0.0,1.0), metallic=0.1)
            )
            
            cube_cfg = RigidObjectCfg(
                prim_path="/World/envs/env_0/Cube",  # 명시적 경로
                spawn=spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=self.cube_position,
                    rot=(1.0, 0.0, 0.0, 0.0)  # 쿼터니언
                ),
            )
            
            # Create and initialize rigid object
            self.cube = RigidObject(cfg=cube_cfg)
            self.cube_initialized = True
            print("✓ Cube setup completed")
            
        except Exception as e:
            print(f"Warning: Cube setup failed: {e}")
            self.cube = None
            self.cube_initialized = False

    def _tensor_to_numpy(self, tensor):
        """텐서를 numpy 배열로 안전하게 변환"""
        try:
            if torch.is_tensor(tensor):
                return tensor.detach().cpu().numpy()
            elif isinstance(tensor, np.ndarray):
                return tensor
            else:
                return np.array(tensor, dtype=np.float32)
        except Exception as e:
            print(f"Warning: Tensor conversion failed: {e}")
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def _get_cube_position(self):
        """큐브 위치 안전하게 가져오기"""
        try:
            if self.cube is None:
                return self.cube_position.copy()
            
            # 다양한 방법으로 큐브 위치 시도
            if hasattr(self.cube, 'data') and hasattr(self.cube.data, 'root_pos_w'):
                cube_pos_tensor = self.cube.data.root_pos_w[0]
                cube_pos = self._tensor_to_numpy(cube_pos_tensor)[:3]
                return cube_pos
            elif hasattr(self.cube, '_data') and hasattr(self.cube._data, 'root_pos_w'):
                cube_pos_tensor = self.cube._data.root_pos_w[0]
                cube_pos = self._tensor_to_numpy(cube_pos_tensor)[:3]
                return cube_pos
            else:
                # 기본 위치 반환
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
                base = self._tensor_to_numpy(base)
            
            # 큐브 위치 가져오기
            cube_pos = self._get_cube_position()
            
            # 모든 것을 numpy 배열로 변환
            base = np.array(base, dtype=np.float32).flatten()
            cube_pos = np.array(cube_pos, dtype=np.float32).flatten()[:3]
            goal_pos = np.array(self.goal_position, dtype=np.float32).flatten()[:3]
            
            return np.concatenate([base, cube_pos, goal_pos], axis=0)
            
        except Exception as e:
            print(f"Warning: Observation merge failed: {e}")
            # 안전한 기본값 반환
            base_dim = self.observation_space.shape[0] - 6
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

    def get_action_dim(self):
        """액션 차원 반환"""
        return self._action_dim

    def reset(self):
        """환경 리셋"""
        try:
            # 큐브 설정 (첫 번째 리셋에서만)
            if not self.cube_initialized:
                self._setup_cube()
            
            obs_dict, info = super().reset()
            merged = self._merge_obs(obs_dict)
            return merged, info
            
        except Exception as e:
            print(f"Warning: Reset failed: {e}")
            # 안전한 기본값 반환
            dummy_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
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
                action = self._tensor_to_numpy(action)
                if action.ndim == 1:
                    action = action.reshape(1, -1)
            
            # 부모 클래스의 step 호출
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
            dummy_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            return dummy_obs, -1.0, True, {}

    def _calculate_reward(self, obs):
        """보상 계산"""
        try:
            # 관측값에서 위치 정보 추출
            base_obs_dim = len(obs) - 6
            ant_base = obs[:3]  # 첫 3개는 ant 위치로 가정
            cube_pos = obs[base_obs_dim:base_obs_dim+3]
            goal_pos = obs[base_obs_dim+3:base_obs_dim+6]
            
            # 거리 기반 보상
            cube_to_goal_dist = np.linalg.norm(cube_pos - goal_pos)
            ant_to_cube_dist = np.linalg.norm(ant_base - cube_pos)
            
            # 보상 계산
            r_push = -cube_to_goal_dist  # 큐브를 목표에 가깝게
            r_near = -ant_to_cube_dist * 0.1  # 개미를 큐브에 가깝게 (작은 가중치)
            
            reward = r_push + r_near
            return float(reward)
            
        except Exception as e:
            print(f"Warning: Reward calculation failed: {e}")
            return -1.0

    def close(self):
        """환경 정리"""
        try:
            print("🔄 Closing environment...")
            
            # 큐브 정리
            if self.cube is not None:
                try:
                    # 큐브 객체 정리 시도
                    del self.cube
                except:
                    pass
                self.cube = None
                self.cube_initialized = False
            
            # 부모 클래스 정리
            super().close()
            print("✓ Environment closed safely")
            
        except Exception as e:
            print(f"Warning: Environment close failed: {e}")