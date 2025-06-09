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
        # Store custom params FIRST
        c = custom_cfg or {}
        self.cube_position = np.array(c.get('cube_position', [2.0, 0.0, 0.1]), dtype=np.float32)
        self.goal_position = np.array(c.get('goal_position', [3.0, 0.0, 0.1]), dtype=np.float32)
        self.cube = None
        self.cube_created = False
        
        # Build base IsaacLab Ant config
        cfg = AntEnvCfg()
        cfg.scene = MySceneCfg(num_envs=num_envs, env_spacing=cfg.scene.env_spacing)
        cfg.sim.device = str(CONFIG.device)
        
        # Initialize parent
        super().__init__(cfg=cfg)
        
        # Setup observation space after parent initialization
        self._setup_observation_space()

    def _setup_observation_space(self):
        """관측 공간 설정"""
        try:
            if hasattr(self, 'observation_space') and 'policy' in self.observation_space:
                base_shape = self.observation_space['policy'].shape
                base_dim = int(np.prod(base_shape))
            else:
                base_dim = 29  # 기본 ant 관측 차원
            
            full_dim = base_dim + 6  # base + cube_pos(3) + goal_pos(3)
            
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(full_dim,), dtype=np.float32
            )
            print(f"✓ Observation space: {full_dim} dimensions (base: {base_dim})")
            
        except Exception as e:
            print(f"Warning: Failed to setup observation space: {e}")
            # 안전한 기본값
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32
            )

    def setup_scene(self):
        """씬 설정"""
        super().setup_scene()
        self._create_cube()

    def _create_cube(self):
        """큐브 생성"""
        try:
            print("Creating cube object...")
            
            # 큐브 스폰 설정
            spawn_cfg = sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=100.0,
                    max_linear_velocity=100.0,
                    max_depenetration_velocity=10.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.0, 1.0),
                    metallic=0.2,
                    roughness=0.8
                )
            )
            
            # 큐브 설정 - 환경별로 다른 경로 사용
            cube_cfg = RigidObjectCfg(
                prim_path=f"{self.scene.env_prim_paths[0]}/Cube",
                spawn=spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=tuple(self.cube_position),
                    rot=(1.0, 0.0, 0.0, 0.0),  # w, x, y, z
                ),
            )
            
            # 큐브 생성
            self.cube = RigidObject(cfg=cube_cfg)
            self.cube_created = True
            print(f"✓ Cube created successfully at {self.cube_position}")
            
        except Exception as e:
            print(f"Warning: Failed to create cube: {e}")
            self.cube = None
            self.cube_created = False

    def _safe_tensor_convert(self, data):
        """안전한 텐서 변환"""
        try:
            if torch.is_tensor(data):
                return data.detach().cpu().numpy()
            elif isinstance(data, np.ndarray):
                return data
            else:
                return np.array(data, dtype=np.float32)
        except Exception as e:
            print(f"Warning: Tensor conversion failed: {e}")
            return np.zeros(3, dtype=np.float32)

    def _get_cube_position(self):
        """큐브 위치 안전하게 가져오기"""
        try:
            if self.cube_created and self.cube is not None:
                if hasattr(self.cube, 'data') and hasattr(self.cube.data, 'root_pos_w'):
                    cube_pos_tensor = self.cube.data.root_pos_w[0]
                    cube_pos = self._safe_tensor_convert(cube_pos_tensor)[:3]
                    return np.array(cube_pos, dtype=np.float32)
            
            # 큐브가 없거나 접근 실패시 초기 위치 반환
            return np.array(self.cube_position, dtype=np.float32)
            
        except Exception as e:
            print(f"Warning: Failed to get cube position: {e}")
            return np.array(self.cube_position, dtype=np.float32)

    def _get_ant_position(self):
        """개미 위치 안전하게 가져오기"""
        try:
            if hasattr(self.scene, 'robot') and hasattr(self.scene.robot, 'data'):
                ant_pos_tensor = self.scene.robot.data.root_pos_w[0]
                ant_pos = self._safe_tensor_convert(ant_pos_tensor)[:3]
                return np.array(ant_pos, dtype=np.float32)
            else:
                # 기본 관측값에서 추출 시도
                return np.zeros(3, dtype=np.float32)
        except Exception as e:
            print(f"Warning: Failed to get ant position: {e}")
            return np.zeros(3, dtype=np.float32)

    def _merge_obs(self, obs_dict):
        """관측값 병합 - 안전한 버전"""
        try:
            # 기본 관측값 가져오기
            if 'policy' in obs_dict:
                base = obs_dict['policy']
            else:
                # 대체 방법
                base = list(obs_dict.values())[0] if obs_dict else np.zeros(29)
            
            # 텐서를 numpy로 변환
            base = self._safe_tensor_convert(base)
            
            # 차원 조정
            if base.ndim > 1:
                base = base[0] if base.shape[0] > 0 else base.flatten()
            
            base = np.array(base, dtype=np.float32).flatten()
            
            # 큐브와 목표 위치
            cube_pos = self._get_cube_position()
            goal_pos = np.array(self.goal_position, dtype=np.float32)
            
            # 디버그 정보
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 1
            
            if self._debug_counter <= 3:  # 처음 3번만 출력
                print(f"Debug merge - Base: {base.shape}, Cube: {cube_pos.shape}, Goal: {goal_pos.shape}")
            
            # 연결
            merged = np.concatenate([base, cube_pos, goal_pos], axis=0)
            return merged.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Failed to merge observations: {e}")
            # 실패시 더미 관측값
            return np.zeros(35, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """환경 리셋"""
        try:
            # 부모 클래스 리셋
            if seed is not None:
                obs_dict, info = super().reset(seed=seed, options=options)
            else:
                obs_dict, info = super().reset()
            
            # 큐브 위치 리셋
            self._reset_cube_position()
            
            # 관측값 병합
            merged = self._merge_obs(obs_dict)
            
            return merged, info
            
        except Exception as e:
            print(f"Warning: Reset failed: {e}")
            # 실패시 더미 값
            dummy_obs = np.zeros(35, dtype=np.float32)
            return dummy_obs, {}

    def _reset_cube_position(self):
        """큐브 위치 리셋"""
        try:
            if self.cube_created and self.cube is not None:
                # 위치와 회전을 텐서로 변환
                pos_tensor = torch.tensor(
                    [self.cube_position], 
                    dtype=torch.float32, 
                    device=self.device
                )
                rot_tensor = torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]], 
                    dtype=torch.float32, 
                    device=self.device
                )
                
                # 큐브 위치 설정
                if hasattr(self.cube, 'write_root_pose_to_sim'):
                    self.cube.write_root_pose_to_sim(pos_tensor, rot_tensor)
                elif hasattr(self.cube, 'set_world_poses'):
                    self.cube.set_world_poses(pos_tensor, rot_tensor)
                
                print(f"✓ Cube reset to: {self.cube_position}")
            else:
                print("Warning: Cube not available for reset")
                
        except Exception as e:
            print(f"Warning: Failed to reset cube position: {e}")

    def step(self, action):
        """환경 스텝 - 안전한 버전"""
        try:
            # Action 전처리 - numpy 배열로 확실히 변환
            if torch.is_tensor(action):
                action = action.detach().cpu().numpy()
            
            action = np.array(action, dtype=np.float32)
            
            # 차원 조정
            if action.ndim == 1:
                action = action.reshape(1, -1)
            
            # 액션 범위 클리핑
            action = np.clip(action, -1.0, 1.0)
            
            # 부모 클래스 스텝 - 예외 처리 강화
            try:
                obs_dict, _, term, trunc, info = super().step(action)
            except Exception as step_error:
                print(f"Warning: Parent step failed: {step_error}")
                # 더미 관측값 생성
                obs_dict = {'policy': np.zeros(29, dtype=np.float32)}
                term, trunc = True, False
                info = {}
            
            # 관측값 병합
            obs = self._merge_obs(obs_dict)
            
            # 보상 계산
            reward = self._calculate_reward(obs)
            
            done = bool(term or trunc)
            
            return obs, reward, done, info
            
        except Exception as e:
            print(f"Warning: Step completely failed: {e}")
            # 완전 실패시 안전한 더미 값 반환
            dummy_obs = np.zeros(35, dtype=np.float32)
            return dummy_obs, 0.0, True, {}

    def _calculate_reward(self, obs):
        """보상 계산"""
        try:
            if len(obs) >= 6:
                # 위치 정보 추출
                cube_pos = obs[-6:-3]
                goal_pos = obs[-3:]
                
                # 개미 위치 (첫 3개 또는 실제 위치)
                ant_pos = self._get_ant_position()
                if np.allclose(ant_pos, 0):  # 실패시 관측값에서 추출
                    ant_pos = obs[:3]
                
                # 거리 계산
                cube_to_goal = np.linalg.norm(cube_pos - goal_pos)
                ant_to_cube = np.linalg.norm(ant_pos - cube_pos)
                
                # 보상 설계
                push_reward = -cube_to_goal  # 큐브를 목표에 가깝게
                approach_reward = -ant_to_cube * 0.1  # 개미가 큐브에 접근
                
                # 성공 보너스
                success_bonus = 10.0 if cube_to_goal < 0.3 else 0.0
                
                total_reward = push_reward + approach_reward + success_bonus
                
                return float(total_reward)
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
            self.cube_created = False
            super().close()
        except Exception as e:
            print(f"Warning: Close failed: {e}")

    def get_info(self):
        """환경 정보 반환"""
        try:
            cube_pos = self._get_cube_position()
            ant_pos = self._get_ant_position()
            
            return {
                'cube_position': cube_pos.tolist(),
                'goal_position': self.goal_position.tolist(),
                'ant_position': ant_pos.tolist(),
                'cube_created': self.cube_created,
                'distance_to_goal': float(np.linalg.norm(cube_pos - self.goal_position)),
                'distance_to_cube': float(np.linalg.norm(ant_pos - cube_pos))
            }
        except Exception as e:
            return {
                'error': str(e),
                'cube_created': self.cube_created
            }
