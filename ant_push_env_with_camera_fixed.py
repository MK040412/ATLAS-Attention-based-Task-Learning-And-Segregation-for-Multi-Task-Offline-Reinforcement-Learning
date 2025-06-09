import numpy as np
import torch
from gymnasium.spaces import Box
from isaaclab.envs import ManagerBasedRLEnv
from config import CONFIG
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg, MySceneCfg

# --- NEW CAMERA IMPORTS ---
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

class IsaacAntPushEnvWithCamera(ManagerBasedRLEnv):
    def __init__(self, custom_cfg=None, num_envs: int = 1):
        print(f"🏗️ Creating Ant Push Environment with Blue Cube + Camera")
        
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

        # --- NEW CAMERA SETUP ---
        self.camera = None
        self.camera_initialized = False
        
        # 카메라 내부 파라미터 계산
        self.fx = CONFIG.camera_width / (2 * np.tan(np.radians(CONFIG.camera_fov / 2)))
        self.fy = self.fx  # 정사각형 픽셀 가정
        self.cx = CONFIG.camera_width / 2
        self.cy = CONFIG.camera_height / 2
        
        print(f"  Camera intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")

        # Redefine observation space: policy obs + vision_cube_pos(3) + goal_pos(3)
        base_shape = self.observation_space['policy'].shape
        base_dim = int(np.prod(base_shape))
        full_dim = base_dim + 3 + 3  # +vision_cube_pos +goal_pos
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(full_dim,), dtype=np.float32
        )
        print(f"✅ Observation space: {self.observation_space.shape}")

    def _setup_camera_after_reset(self):
        """리셋 후 카메라 설정"""
        if not self.camera_initialized:
            try:
                camera_cfg = AntCameraCfg()
                self.camera = Camera(cfg=camera_cfg)
                self.camera_initialized = True
                print("✅ Camera initialized after reset")
            except Exception as e:
                print(f"Warning: Camera initialization failed: {e}")
                self.camera = None

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
        # 더미 데이터 반환 (카메라 문제 해결 전까지)
        rgb = np.zeros((CONFIG.camera_height, CONFIG.camera_width, 3), dtype=np.uint8)
        depth = np.ones((CONFIG.camera_height, CONFIG.camera_width), dtype=np.float32) * 5.0
        
        # 간단한 시뮬레이션: 파란 사각형 그리기
        # 큐브 위치에 따라 이미지에 파란 사각형 그리기
        cube_pos = self._get_cube_position()
        
        # 큐브가 카메라 시야에 있다고 가정하고 중앙에 파란 사각형 그리기
        center_x, center_y = int(self.cx), int(self.cy)
        size = 50  # 사각형 크기
        
        rgb[center_y-size//2:center_y+size//2, center_x-size//2:center_x+size//2] = [0, 0, 255]  # 파란색
        depth[center_y-size//2:center_y+size//2, center_x-size//2:center_x+size//2] = 2.0  # 2미터 거리
        
        return rgb, depth

    def _detect_cube_with_simple_vision(self, rgb, depth):
        """간단한 색상 기반 큐브 감지"""
        try:
            import cv2
            
            # RGB를 BGR로 변환 (OpenCV용)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            
            # 파란색 범위 정의
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            
            # 파란색 마스크 생성
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # 가장 큰 컨투어 선택
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 바운딩 박스 계산
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # 중심점 계산
                u = x + w // 2
                v = y + h // 2
                
                # 깊이 값 가져오기
                d = depth[v, u] if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1] else 2.0
                
                # 3D 좌표 계산 (카메라 좌표계에서 월드 좌표계로 변환)
                X = (u - self.cx) * d / self.fx
                Y = (v - self.cy) * d / self.fy
                Z = d
                
                # 간단한 변환: 카메라가 Ant 위에 있다고 가정
                # 실제로는 더 복잡한 변환이 필요하지만, 여기서는 근사치 사용
                world_x = X + 0.5  # Ant의 기본 x 위치
                world_y = Y + 0.0  # Ant의 기본 y 위치
                world_z = 0.1      # 큐브의 z 위치
                
                return np.array([world_x, world_y, world_z], dtype=np.float32)
            
        except Exception as e:
            print(f"Simple vision detection error: {e}")
        
        # 감지 실패 시 실제 큐브 위치 반환
        return self._get_cube_position()

    def _get_cube_position(self):
        """큐브의 현재 위치 가져오기 (물리 시뮬레이션에서)"""
        if self.cube is not None:
            try:
                cube_pos_tensor = self.cube.data.root_pos_w[0]
                cube_pos = self._tensor_to_numpy(cube_pos_tensor)[:3]
                return np.array(cube_pos, dtype=np.float32)
            except:
                pass
        return self.cube_position.copy()

    def _get_vision_cube_position(self):
        """비전으로 감지된 큐브 위치 가져오기"""
        rgb, depth = self.get_camera_obs()
        vision_cube_pos = self._detect_cube_with_simple_vision(rgb, depth)
        return vision_cube_pos

    def _merge_obs(self, obs_dict):
        """관측값 병합: [policy_obs, vision_cube_pos, goal_pos]"""
        base = obs_dict['policy']
        
        # 텐서를 numpy로 변환
        if torch.is_tensor(base):
            base = self._tensor_to_numpy(base)
        
        # 비전으로 큐브 위치 감지
        vision_cube_pos = self._get_vision_cube_position()
        
        # 모든 것을 numpy 배열로 변환
        base = np.array(base, dtype=np.float32).flatten()
        vision_cube_pos = np.array(vision_cube_pos, dtype=np.float32)
        goal_pos = np.array(self.goal_position, dtype=np.float32)
        
        return np.concatenate([base, vision_cube_pos, goal_pos], axis=0)

    def reset(self):
        obs_dict, info = super().reset()
        
        # 카메라 설정 (첫 리셋 후)
        self._setup_camera_after_reset()
        
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
        vision_cube_pos = obs[-6:-3]  # 비전으로 감지된 큐브 위치
        goal_pos = obs[-3:]   # 목표 위치
        
        # 보상: 큐브를 목표로 밀기 + 개미가 큐브에 가까이 가기
        r_push = -np.linalg.norm(vision_cube_pos - goal_pos)
        r_near = -0.1 * np.linalg.norm(ant_base - vision_cube_pos)
        reward = float(r_push + r_near)
        
        done = bool(term or trunc)
        
        return obs, reward, done, info

    def get_action_dim(self):
        """액션 차원 반환"""
        if hasattr(self.action_space, 'shape'):
            return self.action_space.shape[1] if len(self.action_space.shape) == 2 else self.action_space.shape[0]
        return 8

    def close(self):
        """환경 정리"""
        try:
            super().close()
            print("✅ Environment closed safely")
        except Exception as e:
            print(f"Warning during close: {e}")