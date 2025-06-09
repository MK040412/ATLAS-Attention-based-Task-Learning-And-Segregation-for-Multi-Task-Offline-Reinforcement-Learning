class IntegratedKLDPMMMoESAC:
    def __init__(self, state_dim, action_dim, instr_dim, latent_dim=32,
                 tau_birth=1e-4, tau_merge_kl=0.1, device='cpu'):
        self.device = torch.device(device)
        
        # KL-Divergence 기반 DPMM
        self.dpmm = KLDivergenceDPMM(
            latent_dim=latent_dim,
            tau_birth=tau_birth,
            tau_merge_kl=tau_merge_kl
        )
        
        # Fusion Encoder
        self.fusion_encoder = FusionEncoder(
            instr_dim, state_dim, latent_dim
        ).to(self.device)
        
        # 전문가들
        self.experts = []
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        print(f"🤖 Integrated KL-DPMM-MoE-SAC Agent initialized")
        print(f"  Using KL divergence for cluster merging")
    
    def merge_clusters(self, verbose=False):
        """KL divergence 기반 클러스터 병합"""
        old_count = len(self.dpmm.clusters)
        self.dpmm.merge(verbose=verbose)
        new_count = len(self.dpmm.clusters)
        
        if new_count < old_count:
            print(f"  🔗 KL-Merge: {old_count} → {new_count} clusters")
            return True
        return False