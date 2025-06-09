# SB3MoEAgent 클래스에 추가할 메서드들

def set_dpmm_params(self, tau_birth=None, tau_merge=None):
    """DPMM 파라미터 설정"""
    if hasattr(self, 'dpmm'):
        if tau_birth is not None:
            self.dpmm.tau_birth = tau_birth
        if tau_merge is not None:
            if hasattr(self.dpmm, 'tau_merge_kl'):
                self.dpmm.tau_merge_kl = tau_merge
            elif hasattr(self.dpmm, 'cluster_threshold'):
                self.dpmm.cluster_threshold = tau_merge
    print(f"  🔧 DPMM params set: tau_birth={tau_birth}, tau_merge={tau_merge}")

def get_info(self):
    """에이전트 정보 반환"""
    info = {
        'num_experts': len(getattr(self, 'experts', [])),
        'total_buffer_size': sum(len(getattr(expert, 'replay_buffer', [])) for expert in getattr(self, 'experts', [])),
        'dpmm': {}
    }
    
    if hasattr(self, 'dpmm') and hasattr(self.dpmm, 'get_cluster_info'):
        info['dpmm'] = self.dpmm.get_cluster_info()
    
    return info