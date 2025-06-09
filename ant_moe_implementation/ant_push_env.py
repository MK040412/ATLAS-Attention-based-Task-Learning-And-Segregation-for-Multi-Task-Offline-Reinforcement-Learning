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
        print(f"🏗️ Creating Ant Push Environment with Blue Cube")
        
        # Build base IsaacLab Ant config
        cfg = AntEnvCfg()
        cfg.scene = MySceneCfg(num_envs=num_envs, env_spacing=cfg.scene.env_spacing)
        cfg.sim.device = str(CONFIG.device)
        super().__init__(cfg=cfg)

        # Store custom params
        c = custom_cfg or {}
        self.cube_position = np.array(c.get('cube_position', [2.0, 0.0, 0.1]), dtype=np.float32)
        self.goal_position = np.array(c.get('goal_position', [3.0, 0.0, 0.1]), dtype=np.float32)
        
        print(f"  Cube position: {self.cube_position}")
        print(f"  Goal position: {self.goal_position}")

        # --- BLUE CUBE CREATION ---
        try:
            # Spawn cube as RigidObject using CuboidCfg
            spawn_cfg = sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),  # 20cm cube
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.0, 1.0),  # Blue color
                    metallic=0.1
                )
            )
            
            cube_cfg = RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube",  # Proper path pattern
                spawn=spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(*self.cube_position,),  # Initial position
                ),
            )
            
            # Create and initialize rigid object
            self.cube = RigidObject(cfg=cube_cfg)
            print("✅ Blue cube created successfully")
            
        except Exception as e:
            print(f"Warning: Cube creation failed: {e}")
            self.cube = None

        # Redefine observation space: policy obs + cube_pos(3) + goal_pos(3)
        base_shape = self.observation_space['policy'].shape
        base_dim = int(np.prod(base_shape))
        full_dim = base_dim + 3 + 3  # +cube_pos +goal_pos
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(full_dim,), dtype=np.float32
        )
        print(f"✅ Observation space: {self.observation_space.shape}")

    def _tensor_to_numpy(self, tensor):
        """텐서를 numpy 배열로 안전하게 변환"""
        if torch.is_tensor(tensor):
            return tensor.detach().cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            return np.array(tensor)

    def _prepare_action(self, action):
        """액션을 환경에 맞는 형태로 변환"""
        # numpy array로 변환
        if torch.is_tensor(action):
            action = self._tensor_to_numpy(action)
        elif not isinstance(action, np.ndarray):
            action = np.array(action)
        
        # 차원 확인 및 조정
        if action.ndim == 1:
            # 단일 환경용 action을 다중 환경용으로 확장
            action = action.reshape(1, -1)
        
        # torch tensor로 변환 (Isaac Lab이 기대하는 형태)
        action_tensor = torch.FloatTensor(action).to(CONFIG.device)
        return action_tensor

    def _get_cube_position(self):
        """큐브의 현재 위치 가져오기"""
        if self.cube is not None:
            try:
                cube_pos_tensor = self.cube.data.root_pos_w[0]
                cube_pos = self._tensor_to_numpy(cube_pos_tensor)[:3]
                return np.array(cube_pos, dtype=np.float32)
            except:
                pass
        # 큐브가 없거나 오류 시 초기 위치 반환
        return self.cube_position.copy()

    def _merge_obs(self, obs_dict):
        """관측값 병합: [policy_obs, cube_pos, goal_pos]"""
        base = obs_dict['policy']
        
        # 텐서를 numpy로 변환
        if torch.is_tensor(base):
            base = self._tensor_to_numpy(base)
        
        # 큐브 위치 가져오기
        cube_pos = self._get_cube_position()
        
        # 모든 것을 numpy 배열로 변환
        base = np.array(base, dtype=np.float32).flatten()
        cube_pos = np.array(cube_pos, dtype=np.float32)
        goal_pos = np.array(self.goal_position, dtype=np.float32)
        
        return np.concatenate([base, cube_pos, goal_pos], axis=0)

    def reset(self):
        obs_dict, info = super().reset()
        merged = self._merge_obs(obs_dict)
        print(f"🔄 Environment reset - State shape: {merged.shape}")
        return merged, info

    def step(self, action):
        try:
            # 액션을 적절한 형태로 변환
            action_tensor = self._prepare_action(action)
            
            # 환경 스텝 실행
            obs_dict, _, term, trunc, info = super().step(action_tensor)
            
        except Exception as e:
            print(f"Environment step error: {e}")
            # 더미 값 반환
            obs_dict = {'policy': np.zeros(self.observation_space.shape[0] - 6)}
            term = True
            trunc = False
            info = {}
        
        obs = self._merge_obs(obs_dict)
        
        # 보상 계산
        ant_base = obs[:3]  # Ant의 기본 위치
        cube_pos = obs[-6:-3]  # 큐브 위치
        goal_pos = obs[-3:]   # 목표 위치
        
        # 보상: 큐브를 목표로 밀기 + 개미가 큐브에 가까이 가기
        r_push = -np.linalg.norm(cube_pos - goal_pos)  # 큐브-목표 거리
        r_near = -0.1 * np.linalg.norm(ant_base - cube_pos)  # 개미-큐브 거리
        reward = float(r_push + r_near)
        
        done = bool(term or trunc)
        
        return obs, reward, done, info

    def get_action_dim(self):
        """액션 차원 반환"""
        if hasattr(self.action_space, 'shape'):
            return self.action_space.shape[1] if len(self.action_space.shape) == 2 else self.action_space.shape[0]
        return 8  # 기본값

    def close(self):
        """환경 정리"""
        try:
            super().close()
            print("✅ Environment closed safely")
        except Exception as e:
            print(f"Warning during close: {e}")
