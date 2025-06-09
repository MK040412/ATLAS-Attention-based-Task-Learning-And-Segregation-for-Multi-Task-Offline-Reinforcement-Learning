import torch
from isaaclab.app import AppLauncher

class SimulationManager:
    """시뮬레이션 컨텍스트 관리"""
    
    def __init__(self):
        self.app = None
        self.sim = None
        self.is_initialized = False
        
    def initialize(self, args):
        """시뮬레이션 초기화"""
        try:
            if not self.is_initialized:
                print("🚀 Initializing simulation context...")
                self.app = AppLauncher(vars(args))
                self.sim = self.app.app
                self.is_initialized = True
                print("✓ Simulation context initialized")
            else:
                print("⚠️ Simulation context already initialized")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize simulation: {e}")
            return False
    
    def cleanup_stage(self):
        """스테이지 정리 - Isaac Sim 초기화 후에만 사용"""
        try:
            if self.is_initialized:
                # Isaac Sim이 초기화된 후에만 omni 모듈 import
                try:
                    import omni.usd
                    # 스테이지 정리
                    stage = omni.usd.get_context().get_stage()
                    if stage:
                        # 환경 관련 프림들 삭제
                        env_prims = ["/World/envs", "/World/ant_push_cube"]
                        for prim_path in env_prims:
                            if stage.GetPrimAtPath(prim_path):
                                stage.RemovePrim(prim_path)
                                print(f"  Removed prim: {prim_path}")
                except ImportError:
                    print("  Omni modules not available, skipping stage cleanup")
                
                # 가비지 컬렉션
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print("✓ Stage cleaned up")
        except Exception as e:
            print(f"Warning: Stage cleanup failed: {e}")
    
    def close(self):
        """시뮬레이션 종료"""
        try:
            if self.is_initialized and self.sim:
                print("🔄 Closing simulation...")
                self.cleanup_stage()
                self.sim.close()
                self.sim = None
                self.app = None
                self.is_initialized = False
                print("✓ Simulation closed")
        except Exception as e:
            print(f"Warning: Simulation close failed: {e}")
    
    def reset_for_new_task(self):
        """새 태스크를 위한 리셋"""
        try:
            if self.is_initialized:
                print("🔄 Resetting simulation for new task...")
                self.cleanup_stage()
                # 짧은 대기 시간
                import time
                time.sleep(0.5)
                print("✓ Simulation reset for new task")
        except Exception as e:
            print(f"Warning: Task reset failed: {e}")

# 전역 시뮬레이션 매니저
_sim_manager = None

def get_simulation_manager():
    """전역 시뮬레이션 매니저 반환"""
    global _sim_manager
    if _sim_manager is None:
        _sim_manager = SimulationManager()
    return _sim_manager