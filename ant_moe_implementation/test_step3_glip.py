import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        from ant_push_env_with_glip import IsaacAntPushEnvWithGLIP
        from config import CONFIG
        
        print("🧪 Step 3 Test: GLIP Vision Integration")
        
        # 환경 생성
        env = IsaacAntPushEnvWithGLIP(
            custom_cfg={
                'cube_position': [2.0, 0.0, 0.1],
                'goal_position': [3.0, 0.0, 0.1]
            },
            num_envs=1
        )
        
        # 초기 리셋
        state, info = env.reset()
        print(f"✓ Initial state shape: {state.shape}")
        
        # 상태 분석
        base_obs_dim = len(state) - 6
        ant_pos = state[:3]
        glip_cube_pos = state[base_obs_dim:base_obs_dim+3]
        goal_pos = state[base_obs_dim+3:base_obs_dim+6]
        
        print(f"📊 State Analysis:")
        print(f"  Total state dim: {len(state)}")
        print(f"  Base obs dim: {base_obs_dim}")
        print(f"  Ant position: {ant_pos}")
        print(f"  GLIP cube position: {glip_cube_pos}")
        print(f"  Goal position: {goal_pos}")
        
        # GLIP 감지 테스트
        print(f"\n🔍 GLIP Detection Test:")
        detection_info = env.get_detection_info()
        print(f"  RGB shape: {detection_info['rgb_shape']}")
        print(f"  Depth shape: {detection_info['depth_shape']}")
        print(f"  Number of detections: {detection_info['num_detections']}")
        print(f"  Detection scores: {detection_info['detection_scores']}")
        
        # 몇 스텝 실행하여 GLIP vs 실제 위치 비교
        print(f"\n🎯 GLIP vs Actual Position Comparison:")
        for i in range(5):
            action = torch.randn(env.get_action_dim()) * 0.1
            next_state, reward, done, info = env.step(action.numpy())
            
            # 위치 비교
            glip_pos = next_state[base_obs_dim:base_obs_dim+3]
            actual_pos = env._get_cube_position()
            ant_pos = env._get_ant_position()
            
            detection_error = np.linalg.norm(glip_pos - actual_pos)
            
            print(f"Step {i+1}:")
            print(f"  Reward: {reward:.3f}")
            print(f"  Ant pos: {ant_pos}")
            print(f"  GLIP cube pos: {glip_pos}")
            print(f"  Actual cube pos: {actual_pos}")
            print(f"  Detection error: {detection_error:.3f}")
            print(f"  Distance to goal: {np.linalg.norm(glip_pos - goal_pos):.3f}")
            
            if done:
                state, _ = env.reset()
            else:
                state = next_state
        
        # GLIP 모델 상태 확인
        print(f"\n🤖 GLIP Model Status:")
        print(f"  Model loaded: {env.glip_detector.model_loaded}")
        print(f"  Confidence threshold: {env.glip_detector.confidence_threshold}")
        print(f"  Device: {env.glip_detector.device}")
        
        print("✅ Step 3 test passed!")
        env.close()
        
    except Exception as e:
        print(f"❌ Step 3 test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sim.close()

if __name__ == "__main__":
    main()