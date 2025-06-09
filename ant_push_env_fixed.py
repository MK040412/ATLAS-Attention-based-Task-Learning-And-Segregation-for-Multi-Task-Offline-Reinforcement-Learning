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
        
        # Store custom params BEFORE calling super().__init__
        c = custom_cfg or {}
        self.cube_position = np.array(c.get('cube_position', [2.0,0.0,0.1]), dtype=np.float32)
        self.goal_position = np.array(c.get('goal_position', [3.0,0.0,0.1]), dtype=np.float32)
        self.cube = None  # 초기화
        
        super().__init__(cfg=cfg)
        
        # 관측 공간 재정의는 setup_scene 이후에
        self._setup_observation_space()

    def _setup_observation_space(self):
        """관측 공간 설정"""
        try:
            base_shape = self.observation_space['policy'].shape
            base_dim = int(np.prod(base_shape))
            full_dim = base_dim + 3 + 3  # base + cube_pos + goal_pos
            
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(full_dim,), dtype=np.float32
            )
            print(f"Observation space set: {full_dim} dimensions")
        except Exception as e:
            print(f"Warning: Failed to setup observation space: {e}")
            # 기본값 설정
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32
            )

    def setup_scene(self):
        """씬 설정 - 큐브 생성"""
        super().setup_scene()
        
        try:
            print("Setting up cube in scene...")
            
            # 큐브 스폰 설정
            spawn_cfg = sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.0, 1.0), 
                    metallic=0.1
                )
            )
            
            # 큐브 설정
            cube_cfg = RigidObjectCfg(
                prim_path=f"{self.scene.env_prim_paths[0]}/Cube",
                spawn=spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=self.cube_position,
                    rot=(1.0, 0.0, 0.0, 0.0),  # 쿼터니언 (w, x, y, z)
                ),
            )
            
            # 큐브 생성
            self.cube = RigidObject(cfg=cube_cfg)
            print(f"✓ Cube created at position: {self.cube_position}")
            
        except Exception as e:
            print(f"Warning: Failed to create cube: {e}")
            self.cube = None

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
            if self.cube is not None and hasattr(self.cube, 'data'):
                cube_pos_tensor = self.cube.data.root_pos_w[0]
                cube_pos = self._tensor_to_numpy(cube_pos_tensor)[:3]
                return np.array(cube_pos, dtype=np.float32)
            else:
                # 큐브가 없으면 초기 위치 반환
                return np.array(self.cube_position, dtype=np.float32)
        except Exception as e:
            print(f"Warning: Failed to get cube position: {e}")
            return np.array(self.cube_position, dtype=np.float32)

    def _merge_obs(self, obs_dict):
        """관측값 병합"""
        try:
            # 기본 관측값 가져오기
            base = obs_dict['policy']
            
            # 텐서를 numpy로 변환
            if torch.is_tensor(base):
                base = self._tensor_to_numpy(base)
            
            # 차원 확인 및 조정
            if base.ndim > 1:
                # 다중 환경인 경우 첫 번째 환경만 사용
                base = base[0] if base.shape[0] > 0 else base.flatten()
            
            # 1차원 배열로 만들기
            base = np.array(base, dtype=np.float32).flatten()
            
            # 큐브 위치 가져오기
            cube_pos = self._get_cube_position()
            
            # 목표 위치
            goal_pos = np.array(self.goal_position, dtype=np.float32)
            
            # 모든 배열이 1차원인지 확인
            print(f"Debug - Base shape: {base.shape}, Cube shape: {cube_pos.shape}, Goal shape: {goal_pos.shape}")
            
            # 연결
            merged = np.concatenate([base, cube_pos, goal_pos], axis=0)
            
            return merged.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Failed to merge observations: {e}")
            # 실패시 더미 관측값 반환
            dummy_size = 35  # 기본 크기
            return np.zeros(dummy_size, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """환경 리셋"""
        try:
            # 부모 클래스 리셋
            obs_dict, info = super().reset(seed=seed, options=options)
            
            # 큐브 위치 리셋 (가능한 경우)
            self._reset_cube_position()
            
            # 관측값 병합
            merged = self._merge_obs(obs_dict)
            
            return merged, info
            
        except Exception as e:
            print(f"Warning: Reset failed: {e}")
            # 실패시 더미 값 반환
            dummy_obs = np.zeros(35, dtype=np.float32)
            return dummy_obs, {}

    def _reset_cube_position(self):
        """큐브 위치 리셋"""
        try:
            if self.cube is not None and hasattr(self.cube, 'write_root_pose_to_sim'):
                # 위치와 회전 설정
                positions = torch.tensor([self.cube_position], device=self.device)
                orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)  # 쿼터니언
                
                self.cube.write_root_pose_to_sim(positions, orientations)
                print(f"✓ Cube position reset to: {self.cube_position}")
            else:
                print("Warning: Failed to reset cube position: cube not available")
        except Exception as e:
            print(f"Warning: Failed to reset cube position: {e}")

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
            
            # 부모 클래스 스텝 실행
            obs_dict, _, term, trunc, info = super().step(action)
            
            # 관측값 병합
            obs = self._merge_obs(obs_dict)
            
            # 보상 계산
            reward = self._calculate_reward(obs)
            
            done = bool(term or trunc)
            
            return obs, reward, done, info
            
        except Exception as e:
            print(f"Warning: Step failed: {e}")
            # 실패시 더미 값 반환
            dummy_obs = np.zeros(35, dtype=np.float32)
            return dummy_obs, 0.0, True, {}

    def _calculate_reward(self, obs):
        """보상 계산"""
        try:
            # 관측값에서 위치 정보 추출
            if len(obs) >= 6:
                # 마지막 6개 값이 cube_pos(3) + goal_pos(3)
                cube_pos = obs[-6:-3]
                goal_pos = obs[-3:]
                
                # 개미 위치는 첫 3개 값으로 가정
                ant_pos = obs[:3]
                
                # 보상 계산
                cube_to_goal_dist = np.linalg.norm(cube_pos - goal_pos)
                ant_to_cube_dist = np.linalg.norm(ant_pos - cube_pos)
                
                # 거리 기반 보상 (음수로 하여 거리가 가까울수록 높은 보상)
                r_push = -cube_to_goal_dist
                r_approach = -ant_to_cube_dist * 0.1  # 접근 보상은 작게
                
                reward = r_push + r_approach
                
                return float(reward)
            else:
                return 0.0
                
        except Exception as e:
            print(f"Warning: Reward calculation failed: {e}")
            return 0.0

    def close(self):
        """환경 정리"""
        try:
            if self.cube is not None:
                del self.cube
                self.cube = None
            super().close()
        except Exception as e:
            print(f"Warning: Close failed: {e}")
