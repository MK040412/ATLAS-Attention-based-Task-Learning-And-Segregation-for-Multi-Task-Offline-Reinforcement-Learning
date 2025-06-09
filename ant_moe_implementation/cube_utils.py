import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
import numpy as np

def create_cube(scene, prim_path, position, friction):
    """
    Spawn a dynamic cube prim via RigidObject.
    scene is unused; RigidObject spawns itself.
    """
    spawn_cfg = sim_utils.CuboidCfg(
        size=(0.2, 0.2, 0.2),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.1)
    )
    init_state = RigidObjectCfg.InitialStateCfg()
    init_state.pos = np.array(position, dtype=np.float32)
    cube_cfg = RigidObjectCfg(
        prim_path=prim_path,
        spawn=spawn_cfg,
        init_state=init_state
    )
    cube = RigidObject(cfg=cube_cfg)
    return cube
