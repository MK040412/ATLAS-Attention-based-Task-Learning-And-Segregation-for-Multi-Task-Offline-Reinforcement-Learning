import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

class CubeObject:
    def __init__(self, prim_path: str, initial_position: np.ndarray = None, device: str = "cuda:0"):
        self.prim_path = prim_path
        self.device = device
        
        if initial_position is None:
            initial_position = np.array([2.0, 0.0, 0.1])
        self.initial_position = initial_position
        
        # 큐브 스폰 설정
        spawn_cfg = sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=100.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0), 
                metallic=0.1,
                roughness=0.5
            )
        )
        
        # RigidObject 설정
        self.cube_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(self.initial_position),
                rot=(1.0, 0.0, 0.0, 0.0),  # quaternion (w, x, y, z)
            ),
        )
        
        self.cube = None
    
    def create_cube(self):
        """큐브 객체 생성"""
        self.cube = RigidObject(cfg=self.cube_cfg)
        return self.cube
    
    def get_position(self):
        """큐브의 현재 위치 반환"""
        if self.cube is None:
            return self.initial_position
        
        try:
            # RigidObject에서 위치 데이터 가져오기
            pos_data = self.cube.data.root_pos_w
            if hasattr(pos_data, 'cpu'):
                pos_data = pos_data.cpu().numpy()
            return np.array(pos_data[0], dtype=np.float32)
        except (AttributeError, IndexError):
            return self.initial_position
    
    def set_position(self, position: np.ndarray):
        """큐브 위치 설정"""
        if self.cube is not None:
            try:
                pos_tensor = torch.tensor(position, dtype=torch.float32, device=self.device)
                # 위치 설정 로직 (Isaac Lab API에 따라 조정 필요)
                self.cube.write_root_pose_to_sim(
                    root_pose=torch.cat([pos_tensor, torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)])
                )
            except AttributeError:
                pass
    
    def reset_position(self):
        """큐브를 초기 위치로 리셋"""
        self.set_position(self.initial_position)