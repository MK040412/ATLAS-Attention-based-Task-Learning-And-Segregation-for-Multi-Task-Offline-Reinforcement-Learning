import numpy as np
import torch
from gymnasium.spaces import Box
from isaaclab.envs import ManagerBasedRLEnv
from config import CONFIG
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg, MySceneCfg
from glip_cube_detector import GLIPCubeDetector
from improved_reward_system import ImprovedRewardSystem

class IsaacAntPushEnvImproved(ManagerBasedRLEnv):
    """개선된 보상 시스템을 가진 Ant Push 환경"""
    
    def __init__(self, custom_cfg=None, num_envs: int = 1):
        print(f"🏗️ Creating Improved Ant Push Environment")
        
        # Build base IsaacLab Ant config
        cfg = AntEnvCfg()
        cfg.scene = MySceneCfg(num_envs=num_envs, env_spacing=cfg.scene.env_spacing)
        cfg.sim.device = str(CONFIG.device)
        super().__init__(cfg=cfg)

        # Store custom params
        c = custom_cfg or {}
        self.cube_position = np.array(c.get('cube_position', [2.0, 0.0, 0.1]), dtype=np.float32)
        self.goal_position = np.array(c.get('goal_position', [3.0, 0.0, 0.1]), dtype=np.float32)
        self.success_distance = c.get('success_distance', 0.3)
        self.max_steps = c.get('max_steps', 500)
        
        print(f"  Cube position: {self.cube_position}")
        print(f"  Goal position: {self.goal_position}")
        print(f"  Success distance: {self.success_distance}")
        print(f"  Max steps: {self.max_steps}")

        # 개선된 보상 시스템
        self.reward_system = ImprovedRewardSystem()
        
        # 스텝 카운터
        self.current_step = 0

        # --- BLUE CUBE CREATION ---
        try:
            spawn_cfg = sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.0, 1.0),  # Blue color
                    metallic=0.1
                )
            )
            
            cube_cfg = RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube",
                spawn=spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(*self.cube_position,),
                ),
            )
            
            self.cube = RigidObject(cfg=cube_cfg)
            print("✅ Blue cube created successfully")
            
        except Exception as e:
            print(f"Warning: Cube creation failed: {e}")
            self.cube = None

        # --- GLIP DETECTOR ---
        try:
            self.glip_detector = GLIPCubeDetector(
                model_name="IDEA-Research/grounding-dino-tiny",
                confidence_threshold=0.3,
                device=CONFIG.device
            )
            
            # 카메라 내재 파라미터 (Isaac Lab 기본값)
            self.camera_intrinsics = {
                'fx': 554.3, 'fy': 554.3,
                'cx': 320.0, 'cy': 240.0,
                'width': 640, 'height': 480
            }
            print(f"  Camera intrinsics: fx={self.camera_intrinsics['fx']}, fy={self.camera_intrinsics['fy']}, cx={self.camera_intrinsics['cx']}, cy={self.camera_intrinsics['cy']}")
            
        except Exception as e:
            print(f"Warning: GLIP detector initialization failed: {e}")
            self.glip_detector = None

        # Redefine observation space: policy obs + cube_pos(3) + goal_pos(3)
        base_shape = self.observation_space['policy'].shape
        base_dim = int(np.prod(base_shape))
        full_dim = base_dim + 3 + 3
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(full_dim,), dtype=np.float32
        )
        print(f"✅ Observation space: ({full_dim},)")

    def _tensor_to_numpy(self, tensor):
        """텐서를 numpy 배열로 안전하게 변환"""
        if torch.is_tensor(tensor):
            return tensor.detach().cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            return np.array(tensor)

    def _get_cube_position(self):
        """실제 큐브 위치 가져오기"""
        if self.cube is not None:
            try:
                cube_pos_tensor = self.cube.data.root_pos_w[0]
                cube_pos = self._tensor_to_numpy(cube_pos_tensor)[:3]
                return np.array(cube_pos, dtype=np.float32)
            except:
                pass
        return np.array(self.cube_position, dtype=np.float32)

    def _get_ant_position(self):
        """Ant의 현재 위치 가져오기"""
        try:
            ant_pos_tensor = self.scene["robot"].data.root_pos_w[0]
            ant_pos = self._tensor_to_numpy(ant_pos_tensor)[:3]
            return np.array(ant_pos, dtype=np.float32)
        except:
            return np.array([0.5, 0.0, 0.0], dtype=np.float32)

    def get_camera_obs(self):
        """카메라 관측 데이터 가져오기 (더미 구현)"""
        # 실제 구현에서는 Isaac Lab의 카메라 센서를 사용
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.random.rand(480, 640).astype(np.float32)
        return rgb, depth

    def _get_glip_cube_position(self):
        """GLIP으로 감지된 큐브 위치 가져오기"""
        if self.glip_detector is None:
            return self._get_cube_position()
        
        try:
            rgb, depth = self.get_camera_obs()
            
            # GLIP으로 3D 위치 감지
            world_pos = self.glip_detector.get_best_detection(
                rgb, depth, self.camera_intrinsics
            )
            
            if world_pos is not None:
                # 카메라 좌표계에서 월드 좌표계로 변환
                ant_pos = self._get_ant_position()
                world_x = world_pos[0] + ant_pos[0]
                world_y = world_pos[1] + ant_pos[1] 
                world_z = 0.1  # 큐브는 지면에서 0.1m 높이
                
                return np.array([world_x, world_y, world_z], dtype=np.float32)
        except Exception as e:
            pass
        
        # GLIP 감지 실패 시 실제 큐브 위치 반환
        return self._get_cube_position()

    def _prepare_action(self, action):
        """액션을 환경에 맞게 준비"""
        if isinstance(action, np.ndarray):
            if action.ndim == 1:
                action = action.reshape(1, -1)
            action_tensor = torch.FloatTensor(action).to(self.device)
        elif torch.is_tensor(action):
            if action.dim() == 1:
                action = action.unsqueeze(0)
            action_tensor = action.to(self.device)
        else:
            action_tensor = torch.FloatTensor([[action]]).to(self.device)
        
        return action_tensor

    def _merge_obs(self, obs_dict):
        """관측값 병합: [policy_obs, glip_cube_pos, goal_pos]"""
        base = obs_dict['policy']
        
        # 텐서를 numpy로 변환
        if torch.is_tensor(base):
            base = self._tensor_to_numpy(base)
        
        # GLIP으로 큐브 위치 감지
        glip_cube_pos = self._get_glip_cube_position()
        
        # 모든 것을 numpy 배열로 변환
        base = np.array(base, dtype=np.float32).flatten()
        glip_cube_pos = np.array(glip_cube_pos, dtype=np.float32)
        goal_pos = np.array(self.goal_position, dtype=np.float32)
        
        return np.concatenate([base, glip_cube_pos, goal_pos], axis=0)

    def reset(self):
        """환경 리셋"""
        obs_dict, info = super().reset()
        merged = self._merge_obs(obs_dict)
        
        # 보상 시스템 리셋
        self.reward_system.reset()
        self.current_step = 0
        
        print(f"🔄 Environment reset - State shape: {merged.shape}")
        return merged, info

    def step(self, action):
        """환경 스텝 실행"""
        self.current_step += 1
        
        try:
            action_tensor = self._prepare_action(action)
            obs_dict, _, term, trunc, info = super().step(action_tensor)
            
        except Exception as e:
            print(f"Environment step error: {e}")
            obs_dict = {'policy': np.zeros(self.observation_space.shape[0] - 6)}
            term = True
            trunc = False
            info = {}
        
        obs = self._merge_obs(obs_dict)
        
        # 위치 정보 추출
        base_obs_dim = len(obs) - 6
        ant_pos = self._get_ant_position()
        glip_cube_pos = obs[base_obs_dim:base_obs_dim+3]
        goal_pos = obs[base_obs_dim+3:base_obs_dim+6]
        actual_cube_pos = self._get_cube_position()
        
        # 개선된 보상 계산
        reward, reward_info = self.reward_system.compute_reward(
            ant_pos=ant_pos,
            cube_pos=actual_cube_pos,  # 실제 큐브 위치 사용
            goal_pos=goal_pos,
            action=action if isinstance(action, np.ndarray) else np.array(action)
        )
        
        # 성공 조건 확인
        cube_to_goal_dist = np.linalg.norm(actual_cube_pos - goal_pos)
        success = cube_to_goal_dist < self.success_distance
        
        # 종료 조건
        done = bool(term or trunc or success or self.current_step >= self.max_steps)
        
        # 추가 정보
        info.update({
            'success': success,
            'cube_to_goal_distance': cube_to_goal_dist,
            'ant_to_cube_distance': np.linalg.norm(ant_pos - actual_cube_pos),
            'steps': self.current_step,
            'reward_info': reward_info,
            'glip_detection_error': np.linalg.norm(glip_cube_pos - actual_cube_pos)
        })
        
        return obs, reward, done, info

    def get_action_dim(self):
        """액션 차원 반환"""
        if hasattr(self.action_space, 'shape'):
            return self.action_space.shape[1] if len(self.action_space.shape) == 2 else self.action_space.shape[0]
        return 8

    def update_task_config(self, task_config):
        """태스크 설정 업데이트 (커리큘럼용)"""
        self.cube_position = np.array(task_config['cube_position'], dtype=np.float32)
        self.goal_position = np.array(task_config['goal_position'], dtype=np.float32)
        self.success_distance = task_config.get('success_distance', 0.3)
        self.max_steps = task_config.get('max_steps', 500)
        
        print(f"📝 Task config updated:")
        print(f"  Cube: {self.cube_position} -> Goal: {self.goal_position}")
        print(f"  Success distance: {self.success_distance}, Max steps: {self.max_steps}")

    def close(self):
        """환경 정리"""
        try:
            super().close()
            print("✅ Environment closed safely")
        except Exception as e:
            print(f"Warning during close: {e}")