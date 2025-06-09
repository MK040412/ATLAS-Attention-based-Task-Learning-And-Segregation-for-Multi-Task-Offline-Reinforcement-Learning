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
        
        # 시드 설정 (경고 제거)
        cfg.seed = 42
        
        super().__init__(cfg=cfg)

        # Store custom params
        c = custom_cfg or {}
        self.cube_position = np.array(c.get('cube_position', [2.0,0.0,0.1]), dtype=np.float32)
        self.goal_position = np.array(c.get('goal_position', [3.0,0.0,0.1]), dtype=np.float32)
        self.num_envs = num_envs

        # Spawn cube as RigidObject using CuboidCfg
        spawn_cfg = sim_utils.CuboidCfg(
            size=(0.2,0.2,0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,0.0,1.0), metallic=0.1)
        )
        cube_cfg = RigidObjectCfg(
            prim_path=self.template_env_ns + "/Cube",
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(self.cube_position[0], self.cube_position[1], self.cube_position[2])
            ),
        )
        # Create and initialize rigid object
        self.cube = RigidObject(cfg=cube_cfg)

        # Redefine observation space: policy obs + cube_pos(3) + goal_pos(3)
        base_shape = self.observation_space['policy'].shape
        base_dim = int(np.prod(base_shape))
        full_dim = base_dim + 3 + 3
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(full_dim,), dtype=np.float32
        )
        
        print(f"🏗️ Environment initialized:")
        print(f"  Num envs: {num_envs}")
        print(f"  Base obs dim: {base_dim}")
        print(f"  Full obs dim: {full_dim}")
        print(f"  Action space: {self.action_space.shape}")

    def _tensor_to_numpy(self, tensor):
        """텐서를 numpy 배열로 안전하게 변환"""
        if torch.is_tensor(tensor):
            return tensor.detach().cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            return np.array(tensor, dtype=np.float32)

    def _merge_obs(self, obs_dict):
        """관측값 병합 - 다중 환경 지원"""
        base = obs_dict['policy']
        
        # 텐서를 numpy로 변환
        if torch.is_tensor(base):
            base = self._tensor_to_numpy(base)
        
        # 다중 환경 처리
        if base.ndim == 2:  # (num_envs, obs_dim)
            base = base[0]  # 첫 번째 환경만 사용
        
        # cube world position from RigidObject data
        try:
            cube_pos_tensor = self.cube.data.root_pos_w
            if cube_pos_tensor.ndim == 2:  # (num_envs, 3)
                cube_pos_tensor = cube_pos_tensor[0]  # 첫 번째 환경
            cube_pos = self._tensor_to_numpy(cube_pos_tensor)[:3]
        except Exception as e:
            print(f"Warning: Could not get cube position: {e}")
            cube_pos = self.cube_position.copy()
        
        # 모든 것을 numpy 배열로 변환
        base = np.array(base, dtype=np.float32).flatten()
        cube_pos = np.array(cube_pos, dtype=np.float32)
        goal_pos = np.array(self.goal_position, dtype=np.float32)
        
        merged = np.concatenate([base, cube_pos, goal_pos], axis=0)
        return merged

    def reset(self, seed=None, options=None):
        """환경 리셋 - 개선된 버전"""
        try:
            # 부모 클래스 리셋
            obs_dict, info = super().reset(seed=seed, options=options)
            
            # 큐브 위치 초기화
            if hasattr(self.cube, 'write_root_pose_to_sim'):
                cube_pose = torch.zeros((self.num_envs, 7), device=self.device)
                cube_pose[:, :3] = torch.tensor(self.cube_position, device=self.device)
                cube_pose[:, 3] = 1.0  # quaternion w
                self.cube.write_root_pose_to_sim(cube_pose)
            
            merged = self._merge_obs(obs_dict)
            
            print(f"🔄 Environment reset - State shape: {merged.shape}")
            return merged, info
            
        except Exception as e:
            print(f"❌ Reset error: {e}")
            # 더미 관측값 반환
            dummy_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            return dummy_obs, {}

    def step(self, action):
        """환경 스텝 - 수정된 버전"""
        try:
            # action 차원 및 타입 검증
            if isinstance(action, (list, tuple)):
                action = np.array(action, dtype=np.float32)
            elif torch.is_tensor(action):
                action = self._tensor_to_numpy(action)
            
            # action을 올바른 형태로 변환
            if action.ndim == 1:
                # 단일 환경용 action을 다중 환경용으로 확장
                action = action.reshape(1, -1)
            
            # action 차원 검증
            expected_action_dim = self.action_space.shape[-1]
            if action.shape[-1] != expected_action_dim:
                print(f"Warning: Action dim mismatch. Expected: {expected_action_dim}, Got: {action.shape[-1]}")
                # 차원 맞추기
                if action.shape[-1] < expected_action_dim:
                    padding = np.zeros((action.shape[0], expected_action_dim - action.shape[-1]))
                    action = np.concatenate([action, padding], axis=-1)
                else:
                    action = action[:, :expected_action_dim]
            
            # 부모 클래스 step 호출
            obs_dict, base_reward, terminated, truncated, info = super().step(action)
            
            # 관측값 병합
            obs = self._merge_obs(obs_dict)
            
            # 커스텀 보상 계산
            reward = self._compute_reward(obs)
            
            # done 플래그 처리 (텐서 문제 해결)
            if torch.is_tensor(terminated):
                terminated = terminated.any().item()
            if torch.is_tensor(truncated):
                truncated = truncated.any().item()
            
            done = bool(terminated or truncated)
            
            return obs, float(reward), done, info
            
        except Exception as e:
            print(f"❌ Step error: {e}")
            # 더미 값 반환
            dummy_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            return dummy_obs, 0.0, True, {}

    def _compute_reward(self, obs):
        """보상 계산"""
        try:
            # 관측값에서 위치 정보 추출
            base_obs_dim = len(obs) - 6
            ant_pos = obs[:3]  # 첫 3개 차원이 ant 위치라고 가정
            cube_pos = obs[base_obs_dim:base_obs_dim+3]
            goal_pos = obs[base_obs_dim+3:base_obs_dim+6]
            
            # 거리 기반 보상
            cube_to_goal_dist = np.linalg.norm(cube_pos - goal_pos)
            ant_to_cube_dist = np.linalg.norm(ant_pos - cube_pos)
            
            # 보상 함수 (음수 거리 + 보너스)
            r_push = -cube_to_goal_dist
            r_approach = -ant_to_cube_dist * 0.1  # 접근 보상은 작게
            
            # 성공 보너스
            success_bonus = 10.0 if cube_to_goal_dist < 0.3 else 0.0
            
            total_reward = r_push + r_approach + success_bonus
            
            return total_reward
            
        except Exception as e:
            print(f"Warning: Reward computation failed: {e}")
            return 0.0

    def close(self):
        """환경 종료"""
        try:
            super().close()
            print("✅ Environment closed safely")
        except Exception as e:
            print(f"Warning: Environment close error: {e}")