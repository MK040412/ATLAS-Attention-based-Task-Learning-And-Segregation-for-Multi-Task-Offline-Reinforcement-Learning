# train.py
import torch
from omni.isaac.lab.app import AppLauncher
from ant_push_env import IsaacAntPushEnv
from moe_sac_agent import FusionEncoder, assign_to_cluster, merge_clusters, experts
from transformers import AutoTokenizer, AutoModel
import numpy as np

def main():
    # 1. Initialize simulation
    app = AppLauncher({"headless": True})
    # 2. Create environment
    env = IsaacAntPushEnv()
    state_dim = env.observation_space['obs'].shape[0] + 6  # example
    action_dim = env.action_space.shape[0]
    # 3. Text encoder
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_encoder = AutoModel.from_pretrained(model_name).eval()
    # 4. Fusion encoder
    fusion = FusionEncoder(instr_emb_dim=384, state_dim=state_dim, latent_dim=32)
    fusion = fusion

    num_episodes = 1000
    max_steps = 200

    for ep in range(num_episodes):
        obs = env.reset()
        # sample instruction (fixed for now)
        instruction = "Push the cube to the target"
        # encode instruction once
        inputs = tokenizer(instruction, return_tensors="pt")
        with torch.no_grad():
            outputs = text_encoder(**inputs)
            instr_emb = outputs.pooler_output.squeeze(0)
        for t in range(max_steps):
            # prepare state_obs vector
            state_obs = torch.tensor(np.concatenate([obs['obs'], obs['cube_pos'], obs['goal_pos']]), dtype=torch.float32).unsqueeze(0)
            z = fusion(instr_emb.unsqueeze(0), state_obs).squeeze(0)
            # clustering and expert selection
            idx = assign_to_cluster(z, state_dim, action_dim)
            expert = experts[idx]
            # sample action
            s_tensor = torch.tensor(obs['obs'], dtype=torch.float32).unsqueeze(0)
            a, _ = expert.sample_action(s_tensor)
            action = a.squeeze(0).numpy()
            # step env
            obs, reward, done, _ = env.step(action)
            # store and update
            expert.replay_buffer.add(obs['obs'], action, reward, obs['obs'], done)
            expert.update()
            if done:
                break
        # merge check
        merge_clusters()

    app.close()

if __name__ == "__main__":
    main()
