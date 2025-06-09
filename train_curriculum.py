import argparse
import torch
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Ant MoE Agent Curriculum")
    parser.add_argument("--tasks",    type=str, default="tasks.yaml")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--num_envs", type=int, default=1)  # 기본값을 1로 변경
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    import yaml
    import numpy as np
    from ant_push_env import IsaacAntPushEnv
    from config import CONFIG
    from language_processor import LanguageProcessor
    from dpmm_ood_detector import DPMMOODDetector
    from moe_sac_agent import InstructionFusion, MoESACAgent

    # 전역 변수들
    tasks = yaml.safe_load(open(args.tasks))
    lang = LanguageProcessor(device=CONFIG.device)
    dpmm = DPMMOODDetector(latent_dim=CONFIG.latent_dim)
    fusion = None
    experts = []

    def init_expert(state_dim, action_dim):
        """새로운 전문가 초기화"""
        print(f"  Creating new expert {len(experts)}")
        expert = MoESACAgent(
            state_dim=state_dim, 
            action_dim=action_dim, 
            instr_dim=CONFIG.instr_emb_dim,
            num_experts=4,
            hidden_dim=CONFIG.hidden_dim
        )
        experts.append(expert)
        return expert

    class FusionEncoder(torch.nn.Module):
        def __init__(self, instr_dim, state_dim, latent_dim):
            super().__init__()
            print(f"InstructionFusion init: instr_dim={instr_dim}, state_dim={state_dim}, hidden_dim={latent_dim}")
            self.instr_proj = torch.nn.Linear(instr_dim, latent_dim//2)
            self.state_proj = torch.nn.Linear(state_dim, latent_dim//2)
            self.fusion = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, latent_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(latent_dim, latent_dim)
            )
        
        def forward(self, instr_emb, state_obs):
            # 디버깅 출력 (첫 번째 스텝에서만)
            if not hasattr(self, '_debug_printed'):
                print(f"  instr_emb shape: {instr_emb.shape}")
                print(f"  st shape: {state_obs.shape}")
                self._debug_printed = True
            
            # 차원 정리
            if instr_emb.dim() == 3:
                instr_emb = instr_emb.squeeze(0).mean(dim=0)  # (1, seq, dim) -> (dim)
            elif instr_emb.dim() == 2:
                instr_emb = instr_emb.mean(dim=0)  # (seq, dim) -> (dim)
            
            if state_obs.dim() == 2:
                state_obs = state_obs.squeeze(0)  # (1, dim) -> (dim)
            
            # 배치 차원 추가
            if instr_emb.dim() == 1:
                instr_emb = instr_emb.unsqueeze(0)
            if state_obs.dim() == 1:
                state_obs = state_obs.unsqueeze(0)
            
            instr_proj = self.instr_proj(instr_emb)
            state_proj = self.state_proj(state_obs)
            combined = torch.cat([instr_proj, state_proj], dim=-1)
            z = self.fusion(combined).squeeze(0)
            
            if not hasattr(self, '_debug_printed_z'):
                print(f"  z shape: {z.shape}")
                self._debug_printed_z = True
                
            return z

    # 메인 학습 루프
    for t in tasks:
        print(f"\n{'='*50}")
        print(f"Task: {t['name']}")
        print(f"Instruction: {t['instruction']}")
        print(f"{'='*50}")
        
        env = IsaacAntPushEnv(custom_cfg=t['env_cfg'], num_envs=args.num_envs)
        
        # 초기 리셋으로 상태 벡터 얻기
        state, _ = env.reset()
        
        # 다중 환경인 경우 첫 번째 환경만 사용
        if isinstance(state, np.ndarray) and state.ndim > 1:
            state = state[0]  # 첫 번째 환경의 상태만 사용
        
        state_dim = state.shape[0]
        
        # 액션 차원 계산
        if len(env.action_space.shape) == 2:
            action_dim = env.action_space.shape[1]  # (num_envs, action_dim)
        else:
            action_dim = env.action_space.shape[0]  # (action_dim,)
        
        print(f"State dim: {state_dim}, Action dim: {action_dim}")
        print(f"Action space shape: {env.action_space.shape}")
        print(f"State shape: {state.shape}")
        
        # Fusion 모델 초기화 (한 번만)
        if fusion is None:
            fusion = FusionEncoder(CONFIG.instr_emb_dim, state_dim, CONFIG.latent_dim).to(CONFIG.device)
        
        # 지시사항 임베딩
        instr_emb = lang.encode(t['instruction'])
        
        # 에피소드 루프
        for episode in range(t['episodes']):
            print(f"\n=== Episode {episode + 1}/{t['episodes']} ===")
            done = False
            state_cur = state.copy() if hasattr(state, 'copy') else state
            step_count = 0
            max_steps = 1000
            episode_reward = 0
            episode_losses = []
            
            while not done and step_count < max_steps:
                if step_count == 0:
                    print(f"Step {step_count}:")
                
                # 상태를 텐서로 변환
                st = torch.tensor(state_cur, device=CONFIG.device, dtype=torch.float32)
                if st.dim() == 1:
                    st = st.unsqueeze(0)
                
                # 융합된 특징 추출
                z = fusion(instr_emb, st).squeeze(0) if fusion(instr_emb, st).dim() > 1 else fusion(instr_emb, st)
                
                # 클러스터 할당
                k = dpmm.assign(z.detach().cpu().numpy())
                if step_count == 0:
                    print(f"  Assigned to expert {k}")
                
                # 전문가가 없으면 생성
                if k >= len(experts):
                    init_expert(state_dim, action_dim)
                
                # 전문가로부터 행동 선택
                exp = experts[k]
                action = exp.select_action(state_cur, instr_emb.squeeze().detach().cpu().numpy())
                
                if step_count == 0:
                    print(f"  Generated action shape: {action.shape}")
                    print(f"  Action values: {action}")
                
                # 환경에서 스텝 실행
                next_state, reward, done, info = env.step(action)
                
                # 다중 환경인 경우 첫 번째 환경만 사용
                if isinstance(next_state, np.ndarray) and next_state.ndim > 1:
                    next_state = next_state[0]
                if isinstance(reward, (list, np.ndarray)) and len(reward) > 1:
                    reward = reward[0]
                if isinstance(done, (list, np.ndarray)) and len(done) > 1:
                    done = done[0]
                
                episode_reward += reward
                
                # 위치 정보 추출 (디버깅용)
                base_obs_dim = len(next_state) - 6
                try:
                    ant_pos_data = env.scene["robot"].data.root_pos_w[0]
                    ant_pos = env._tensor_to_numpy(ant_pos_data)[:3]
                except:
                    ant_pos = next_state[:3]
                
                cube_pos = next_state[base_obs_dim:base_obs_dim+3]
                goal_pos = next_state[base_obs_dim+3:base_obs_dim+6]
                
                # 매 100스텝마다 상세 정보 출력
                if step_count % 100 == 0:
                    print(f"  Step {step_count}:")
                    print(f"    Reward: {reward:.3f}, Total: {episode_reward:.3f}")
                    print(f"    Ant pos: {ant_pos}")
                    print(f"    Cube pos: {cube_pos}")
                    print(f"    Goal pos: {goal_pos}")
                    print(f"    Distance to goal: {np.linalg.norm(cube_pos - goal_pos):.3f}")
                
                state_cur = next_state
                step_count += 1
            
            print(f"Episode {episode + 1} Summary:")
            print(f"  Steps: {step_count}, Total Reward: {episode_reward:.3f}")
            print(f"  Average Reward: {episode_reward/step_count:.3f}")
            
            # 클러스터 병합
            dpmm.merge()
            
            # 다음 에피소드를 위한 리셋
            state, _ = env.reset()
            if isinstance(state, np.ndarray) and state.ndim > 1:
                state = state[0]
        
        # 클러스터 정보 출력
        cluster_info = dpmm.get_cluster_info()
        print(f"\nTask '{t['name']}' completed!")
        print(f"Total clusters discovered: {cluster_info['num_clusters']}")
        print(f"Cluster counts: {cluster_info['cluster_counts']}")
        
        env.close()

    sim.close()
    print("\nTraining completed!")

if __name__=="__main__":
    main()
