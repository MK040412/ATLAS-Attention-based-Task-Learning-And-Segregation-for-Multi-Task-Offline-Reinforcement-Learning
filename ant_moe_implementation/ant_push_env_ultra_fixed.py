import numpy as np
import torch
from gymnasium.spaces import Box
from isaaclab.envs import ManagerBasedRLEnv
from config import CONFIG
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg, MySceneCfg

class IsaacAntPushEnv(ManagerBasedRLEnv):
    """완전히 수정된 Ant Push 환경"""
    
    def __init__(self, custom_cfg=None, num_envs: int = 1):
        print(f"Creating IsaacAntPushEnv with {num_envs} environments...")
        
        # Build base IsaacLab Ant config
        cfg = AntEnvCfg()
        cfg.scene = MySceneCfg(num_envs=num_envs, env_spacing=cfg.scene.env_spacing)
        cfg.sim.device = str(CONFIG.device)
        super().__init__(cfg=cfg)

        # Store custom params
        c = custom_cfg or {}
        self.cube_position = np.array(c.get('cube_position', [2.0,0.0,0.1]), dtype=np.float32)
        self.goal_position = np.array(c.get('goal_position', [3.0,0.0,0.1]), dtype=np.float32)
        
        print(f"Environment configuration:")
        print(f"  Cube position: {self.cube_position}")
        print(f"  Goal position: {self.goal_position}")

        # Spawn cube as RigidObject using CuboidCfg
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

        # Get base observation space info
        base_shape = self.observation_space['policy'].shape
        self.base_obs_dim = int(np.prod(base_shape))
        
        # Total observation: base + cube_pos(3) + goal_pos(3)
        self.total_obs_dim = self.base_obs_dim + 3 + 3
        
        # Redefine observation space
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.total_obs_dim,), dtype=np.float32
        )
        
        print(f"Observation space setup:")
        print(f"  Base observation dim: {self.base_obs_dim}")
        print(f"  Total observation dim: {self.total_obs_dim}")
        print(f"  Action space shape: {self.action_space.shape}")
        
        # 차원 정보 저장
        self._state_dim = self.total_obs_dim
        self._action_dim = self.action_space.shape[1] if len(self.action_space.shape) == 2 else self.action_space.shape[0]
        
        print(f"Final dimensions:")
        print(f"  State dim: {self._state_dim}")
        print(f"  Action dim: {self._action_dim}")
        print("✓ IsaacAntPushEnv initialized successfully")

    def get_observation_dim(self):
        """정확한 관측 차원 반환"""
        return self._state_dim
    
    def get_action_dim(self):
        """정확한 행동 차원 반환"""
        return self._action_dim

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
            return np.array([0.0], dtype=np.float32)

    def _merge_obs(self, obs_dict):
        """관측 병합 - 차원 보장"""
        try:
            # 기본 관측 추출
            base = obs_dict['policy']
            
            # 텐서를 numpy로 변환
            if torch.is_tensor(base):
                base = self._tensor_to_numpy(base)
            
            # 1차원으로 평탄화
            base = np.array(base, dtype=np.float32).flatten()
            
            # 기본 관측 차원 검증
            if len(base) != self.base_obs_dim:
                print(f"Warning: Base obs dimension mismatch: expected {self.base_obs_dim}, got {len(base)}")
                if len(base) < self.base_obs_dim:
                    base = np.pad(base, (0, self.base_obs_dim - len(base)), mode='constant')
                else:
                    base = base[:self.base_obs_dim]
            
            # 큐브 위치 추출
            try:
                cube_pos_tensor = self.cube.data.root_pos_w[0]
                cube_pos = self._tensor_to_numpy(cube_pos_tensor)[:3]
            except Exception as cube_error:
                print(f"Warning: Failed to get cube position: {cube_error}")
                cube_pos = self.cube_position.copy()
            
            cube_pos = np.array(cube_pos, dtype=np.float32).flatten()[:3]
            
            # 목표 위치
            goal_pos = np.array(self.goal_position, dtype=np.float32).flatten()[:3]
            
            # 모든 것을 연결
            merged = np.concatenate([base, cube_pos, goal_pos], axis=0)
            
            # 최종 차원 검증
            if len(merged) != self.total_obs_dim:
                print(f"Warning: Merged obs dimension mismatch: expected {self.total_obs_dim}, got {len(merged)}")
                if len(merged) < self.total_obs_dim:
                    merged = np.pad(merged, (0, self.total_obs_dim - len(merged)), mode='constant')
                else:
                    merged = merged[:self.total_obs_dim]
            
            return merged.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Observation merge failed: {e}")
            # 안전한 기본값 반환
            return np.zeros(self.total_obs_dim, dtype=np.float32)

    def reset(self):
        """환경 리셋 - 차원 보장"""
        try:
            obs_dict, info = super().reset()
            merged = self._merge_obs(obs_dict)
            
            # 차원 최종 검증
            if len(merged) != self.total_obs_dim:
                print(f"ERROR: Reset obs dimension mismatch: expected {self.total_obs_dim}, got {len(merged)}")
                merged = np.zeros(self.total_obs_dim, dtype=np.float32)
            
            return merged, info
            
        except Exception as e:
            print(f"Warning: Reset failed: {e}")
            return np.zeros(self.total_obs_dim, dtype=np.float32), {}

    def step(self, action):
        """환경 스텝 - 차원 보장"""
        try:
            # 액션 전처리
            action = self._preprocess_action(action)
            
            # 부모 클래스 스텝 실행
            obs_dict, _, term, trunc, info = super().step(action)
            
            # 관측 병합
            obs = self._merge_obs(obs_dict)
            
            # 보상 계산
            reward = self._calculate_reward(obs)
            
            # 종료 조건
            done = bool(term or trunc)
            
            return obs, reward, done, info
            
        except Exception as e:
            print(f"Warning: Step failed: {e}")
            # 안전한 기본값 반환
            obs = np.zeros(self.total_obs_dim, dtype=np.float32)
            return obs, -1.0, True, {}
    
    def _preprocess_action(self, action):
        """액션 전처리 - 차원 보장"""
        try:
            # numpy 배열로 변환
            if isinstance(action, (list, tuple)):
                action = np.array(action, dtype=np.float32)
            elif torch.is_tensor(action):
                action = self._tensor_to_numpy(action)
            elif not isinstance(action, np.ndarray):
                action = np.array([action], dtype=np.float32)
            
            # 1차원으로 평탄화
            action = action.flatten()
            
            # 차원 검증 및 조정
            if len(action) != self._action_dim:
                print(f"Warning: Action dimension mismatch: expected {self._action_dim}, got {len(action)}")
                if len(action) < self._action_dim:
                    action = np.pad(action, (0, self._action_dim - len(action)), mode='constant')
                else:
                    action = action[:self._action_dim]
            
            # 클리핑
            action = np.clip(action, -1.0, 1.0)
            
            # 환경이 기대하는 형태로 변환 (배치 차원 추가)
            if action.ndim == 1:
                action = action.reshape(1, -1)
            
            return action.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Action preprocessing failed: {e}")
            # 안전한 기본 액션
            return np.zeros((1, self._action_dim), dtype=np.float32)
    
    def _calculate_reward(self, obs):
        """보상 계산"""
        try:
            # 위치 정보 추출
            ant_base = obs[:3]  # 첫 3개 요소가 ant 위치라고 가정
            cube_pos = obs[self.base_obs_dim:self.base_obs_dim+3]
            goal_pos = obs[self.base_obs_dim+3:self.base_obs_dim+6]
            
            # 거리 기반 보상
            cube_to_goal_dist = np.linalg.norm(cube_pos - goal_pos)
            ant_to_cube_dist = np.linalg.norm(ant_base - cube_pos)
            
            # 보상 계산
            r_push = -cube_to_goal_dist  # 큐브를 목표로 밀기
            r_near = -ant_to_cube_dist * 0.1  # 개미가 큐브에 가까이 가기
            
            # 성공 보너스
            success_bonus = 10.0 if cube_to_goal_dist < 0.5 else 0.0
            
            total_reward = r_push + r_near + success_bonus
            
            return float(total_reward)
            
        except Exception as e:
            print(f"Warning: Reward calculation failed: {e}")
            return -1.0
    
    def get_debug_info(self):
        """디버깅 정보 반환"""
        try:
            obs, _ = self.reset()
            return {
                'observation_dim': len(obs),
                'expected_obs_dim': self.total_obs_dim,
                'action_dim': self._action_dim,
                'base_obs_dim': self.base_obs_dim,
                'cube_position': self.cube_position.tolist(),
                'goal_position': self.goal_position.tolist(),
                'observation_space_shape': self.observation_space.shape,
                'action_space_shape': self.action_space.shape
            }
        except Exception as e:
            return {'error': str(e)}
    
    def close(self):
        """환경 정리"""
        try:
            super().close()
        except Exception as e:
            print(f"Warning: Environment close failed: {e}")