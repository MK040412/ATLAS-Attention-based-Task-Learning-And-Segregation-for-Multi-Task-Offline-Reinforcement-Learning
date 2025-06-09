import numpy as np
import torch
from gymnasium.spaces import Box
from isaaclab.envs import ManagerBasedRLEnv
from config import CONFIG
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg, MySceneCfg
from glip_cube_detector import GLIPCubeDetector

# --- CAMERA IMPORTS ---
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.utils import configclass

@configclass
class AntCameraCfg(CameraCfg):
    """Ant에 부착된 카메라 설정"""
    width: int = CONFIG.camera_width
    height: int = CONFIG.camera_height
    prim_path: str = "/World/envs/env_.*/Robot/torso/Camera"
    update_period: float = 0.1
    data_types: list = ["rgb", "distance_to_image_plane"]
    spawn: sim_utils.PinholeCameraCfg = sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(CONFIG.camera_near, CONFIG.camera_far),
    )

class IsaacAntPushEnvWithGLIP(ManagerBasedRLEnv):
    def __init__(self, custom_cfg=None, num_envs: int = 1):
        print(f"🏗️ Creating Ant Push Environment with GLIP Vision")
        
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
        self.glip_detector = GLIPCubeDetector()
        
        # --- CAMERA SETUP ---
        self.camera = None
        self.camera_initialized = False
        
        # 카메라 내부 파라미터
        self.fx = CONFIG.camera_width / (2 * np.tan(np.radians(CONFIG.camera_fov / 2)))
        self.fy = self.fx
        self.cx = CONFIG.camera_width / 2
        self.cy = CONFIG.camera_height / 2
        self.camera_intrinsics = (self.fx, self.fy, self.cx, self.cy)
        
        print(f"  Camera intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")

        # Observation space: policy obs + glip_cube_pos(3) + goal_pos(3)
        base_shape = self.observation_space['policy'].shape
        base_dim = int(np.prod(base_shape))
        full_dim = base_dim + 3 + 3
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
        if torch.is_tensor(action):
            action = self._tensor_to_numpy(action)
        elif not isinstance(action, np.ndarray):
            action = np.array(action)
        
        if action.ndim == 1:
            action = action.reshape(1, -1)
        
        action_tensor = torch.FloatTensor(action).to(CONFIG.device)
        return action_tensor

    def get_camera_obs(self):
        """카메라 관측 데이터 가져오기"""
        # 시뮬레이션된 카메라 데이터 (실제 카메라 문제 해결 전까지)
        rgb = np.zeros((CONFIG.camera_height, CONFIG.camera_width, 3), dtype=np.uint8)
        depth = np.ones((CONFIG.camera_height, CONFIG.camera_width), dtype=np.float32) * 5.0
        
        # 큐브 위치에 따라 시뮬레이션된 이미지 생성
        cube_pos = self._get_cube_position()
        
        # 큐브를 이미지 중앙에 그리기 (시뮬레이션)
        center_x, center_y = int(self.cx), int(self.cy)
        size = 60  # 큐브 크기
        
        # 파란 사각형 그리기
        y1, y2 = max(0, center_y - size//2), min(rgb.shape[0], center_y + size//2)
        x1, x2 = max(0, center_x - size//2), min(rgb.shape[1], center_x + size//2)
        
        rgb[y1:y2, x1:x2] = [0, 0, 255]  # 파란색 (RGB)
        depth[y1:y2, x1:x2] = np.linalg.norm(cube_pos[:2])  # 거리 기반 깊이
        
        return rgb, depth

    def _get_cube_position(self):
        """큐브의 현재 위치 가져오기"""
        if self.cube is not None:
            try:
                cube_pos_tensor = self.cube.data.root_pos_w[0]
                cube_pos = self._tensor_to_numpy(cube_pos_tensor)[:3]
                return np.array(cube_pos, dtype=np.float32)
            except:
                pass
        return self.cube_position.copy()

    def _get_glip_cube_position(self):
        """GLIP으로 감지된 큐브 위치 가져오기"""
        rgb, depth = self.get_camera_obs()
        
        # GLIP으로 3D 위치 감지
        world_pos = self.glip_detector.get_best_detection(
            rgb, depth, self.camera_intrinsics
        )
        
        if world_pos is not None:
            # 카메라 좌표계에서 월드 좌표계로 변환
            # 간단한 변환: Ant 위치 기준으로 오프셋 적용
            ant_pos = self._get_ant_position()
            world_x = world_pos[0] + ant_pos[0]
            world_y = world_pos[1] + ant_pos[1] 
            world_z = 0.1  # 큐브는 지면에서 0.1m 높이
            
            return np.array([world_x, world_y, world_z], dtype=np.float32)
        else:
            # GLIP 감지 실패 시 실제 큐브 위치 반환
            return self._get_cube_position()

    def _get_ant_position(self):
        """Ant의 현재 위치 가져오기"""
        try:
            ant_pos_tensor = self.scene["robot"].data.root_pos_w[0]
            ant_pos = self._tensor_to_numpy(ant_pos_tensor)[:3]
            return np.array(ant_pos, dtype=np.float32)
        except:
            return np.array([0.5, 0.0, 0.0], dtype=np.float32)

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
        obs_dict, info = super().reset()
        merged = self._merge_obs(obs_dict)
        print(f"🔄 Environment reset - State shape: {merged.shape}")
        return merged, info

    def step(self, action):
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
        
        # 보상 계산
        ant_base = obs[:3]  # Ant의 기본 위치
        glip_cube_pos = obs[-6:-3]  # GLIP으로 감지된 큐브 위치
        goal_pos = obs[-3:]   # 목표 위치
        
        # 보상: 큐브를 목표로 밀기 + 개미가 큐브에 가까이 가기
        r_push = -np.linalg.norm(glip_cube_pos - goal_pos)
        r_near = -0.1 * np.linalg.norm(ant_base - glip_cube_pos)
        
        # GLIP 감지 성공 보너스
        actual_cube_pos = self._get_cube_position()
        detection_error = np.linalg.norm(glip_cube_pos - actual_cube_pos)
        r_detection = -0.05 * detection_error  # 감지 정확도 보상
        
        reward = float(r_push + r_near + r_detection)
        done = bool(term or trunc)
        
        return obs, reward, done, info

    def get_action_dim(self):
        """액션 차원 반환"""
        if hasattr(self.action_space, 'shape'):
            return self.action_space.shape[1] if len(self.action_space.shape) == 2 else self.action_space.shape[0]
        return 8

    def get_detection_info(self):
        """감지 정보 반환 (디버깅용)"""
        rgb, depth = self.get_camera_obs()
        boxes, scores = self.glip_detector.detect_cube(rgb)
        
        return {
            'rgb_shape': rgb.shape,
            'depth_shape': depth.shape,
            'num_detections': len(boxes),
            'detection_scores': scores.tolist() if len(scores) > 0 else [],
            'detection_boxes': boxes.tolist() if len(boxes) > 0 else []
        }

    def close(self):
        """환경 정리"""
        try:
            super().close()
            print("✅ Environment closed safely")
        except Exception as e:
            print(f"Warning during close: {e}")