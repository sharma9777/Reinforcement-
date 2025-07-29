# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import gymnasium as gym
import time
import os

# ================================================================
#  CELL 1: Model Classes
# ================================================================
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=4, embed_dim=256):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, embed_dim) # 64 * 7 * 7 = 3136
    def forward(self, x):
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

class QNetwork(nn.Module):
    def __init__(self, embed_dim=256, action_dim=4):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(embed_dim, 512)
        self.fc2 = nn.Linear(512, action_dim)
    def forward(self, z):
        x = F.relu(self.fc1(z))
        return self.fc2(x)

class ForwardModel(nn.Module):
    def __init__(self, embed_dim=256, action_dim=4):
        super(ForwardModel, self).__init__()
        self.fc1 = nn.Linear(embed_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, embed_dim)
    def forward(self, z, a_oh):
        za = torch.cat([z, a_oh], dim=1)
        x = F.relu(self.fc1(za))
        return self.fc2(x)

# ================================================================
#  CELL 2: Agent Class and Buffers
# ================================================================
class RunningStats:
    def __init__(self): self.n = 0; self.mean = 0.0; self.M2 = 0.0
    def update(self, values):
        for x in values:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            self.M2   += delta * (x - self.mean)
    @property
    def std(self): return 1.0 if self.n < 2 else (self.M2 / (self.n - 1)) ** 0.5

class ReplayBuffer:
    def __init__(self, capacity: int): self.buffer = deque(maxlen=capacity)
    def add(self, s, a, r, s_next, done): self.buffer.append((s, a, r, s_next, done))
    def sample(self, bs: int):
        s, a, r, s2, d = zip(*random.sample(self.buffer, bs))
        return (np.array(s), np.array(a), np.array(r), np.array(s2), np.array(d))
    def __len__(self): return len(self.buffer)

class MaxMinExplorerAgent:
    def __init__(self, action_dim, device, encoder,
                 N=3, M=5,
                 alpha=1e-4, alpha_g=1e-4,
                 gamma=0.99, epsilon=0.05,
                 eta_Q=1.0, lambda_int=1.0,
                 beta_min=0.05, dis_min=0.05,
                 tau=0.005,
                 buffer_capacity=1_000_000,
                 batch_size=32, tau_pred=32,
                 embed_dim=256):

        self.device, self.N, self.M = device, N, M
        self.action_dim = action_dim
        self.gamma, self.epsilon = gamma, epsilon
        self.eta_Q, self.lambda_int = eta_Q, lambda_int
        self.beta_min, self.dis_min = beta_min, dis_min
        self.tau, self.batch_size, self.tau_pred = tau, batch_size, tau_pred
        self.eps = 1e-8

        self.encoder = encoder.to(device)
        self.critics = [QNetwork(embed_dim, action_dim).to(device) for _ in range(N)]
        self.target_critics = [QNetwork(embed_dim, action_dim).to(device) for _ in range(N)]
        for tgt, src in zip(self.target_critics, self.critics):
            tgt.load_state_dict(src.state_dict())
        self.ensemble_models = [ForwardModel(embed_dim, action_dim).to(device) for _ in range(M)]

        all_critic_params = sum((list(net.parameters()) for net in self.critics), [])
        all_model_params = sum((list(m.parameters()) for m in self.ensemble_models), [])
        self.opt_critic  = torch.optim.Adam(all_critic_params, lr=alpha)
        self.opt_models  = torch.optim.Adam(all_model_params, lr=alpha_g)
        self.opt_encoder = torch.optim.Adam(self.encoder.parameters(), lr=alpha)

        self.replay = ReplayBuffer(buffer_capacity)
        self.beta_stats, self.dis_stats = RunningStats(), RunningStats()

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(int(self.action_dim))

        with torch.no_grad():
            s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            z   = self.encoder(s_t)
            q_vals = torch.stack([c(z) for c in self.critics]).squeeze(1)

            z_exp  = z.repeat(self.action_dim, 1)
            a_ohs  = F.one_hot(torch.arange(int(self.action_dim), device=self.device),
                                 num_classes=int(self.action_dim)).float()
            preds  = torch.stack([m(z_exp, a_ohs) for m in self.ensemble_models])

            q_np   = q_vals.cpu().numpy()
            raw_b  = self.eta_Q * np.var(q_np, axis=0)
            raw_d  = torch.std(preds, dim=0).mean(dim=1).cpu().numpy()

            self.beta_stats.update(raw_b); self.dis_stats.update(raw_d)

            q_min = q_np.min(axis=0)
            scores = []
            for rb, rd, q in zip(raw_b, raw_d, q_min):
                beta = max(rb / (self.beta_stats.std + self.eps), self.beta_min)
                dis  = max(rd / (self.dis_stats.std + self.eps),  self.dis_min)
                scores.append(q + self.lambda_int * beta * dis)

            return int(np.argmax(scores))

    def store_transition(self, s, a, r, s2, done):
        self.replay.add(s, a, r, s2, done)

    def learn(self):
        if len(self.replay) < self.batch_size: return
        s, a, r, s2, d = self.replay.sample(self.batch_size)
        s    = torch.as_tensor(s,  dtype=torch.float32, device=self.device)
        s2   = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        r    = torch.as_tensor(r,  dtype=torch.float32, device=self.device)
        done = torch.as_tensor(d,  dtype=torch.float32, device=self.device)
        a    = torch.as_tensor(a,  dtype=torch.long,    device=self.device)

        with torch.no_grad():
            z2 = self.encoder(s2)
            proposer_idx = random.randrange(self.N)
            q_proposals = self.critics[proposer_idx](z2)
            a_star = q_proposals.argmax(dim=1)

            q_vals_all_targets = torch.stack([qc(z2) for qc in self.target_critics])
            a_star_expanded = a_star.view(1, -1, 1).expand(self.N, -1, -1)
            q_vals_for_astar = q_vals_all_targets.gather(2, a_star_expanded).squeeze(-1)

            q_min_target = q_vals_for_astar.min(dim=0).values
            y = r + self.gamma * q_min_target * (1 - done)

        z      = self.encoder(s)
        a_oh   = F.one_hot(a, num_classes=int(self.action_dim)).float()
        pred_all_q = self.critics[proposer_idx](z)
        pred   = pred_all_q.gather(1, a.unsqueeze(1))
        loss_q = F.mse_loss(pred, y.unsqueeze(1))

        self.opt_encoder.zero_grad(); self.opt_critic.zero_grad()
        loss_q.backward(); self.opt_critic.step(); self.opt_encoder.step()

        if self.tau_pred and len(self.replay) > self.tau_pred:
            idxs   = np.random.choice(self.batch_size, self.tau_pred, replace=False)
            z_sel  = z[idxs].detach()
            with torch.no_grad(): z2_sel = self.encoder(s2[idxs])
            a_sel = a_oh[idxs]

            loss_g = sum(F.mse_loss(m(z_sel, a_sel), z2_sel) for m in self.ensemble_models) / self.M
            self.opt_models.zero_grad(); loss_g.backward(); self.opt_models.step()

        for tgt, src in zip(self.target_critics, self.critics):
            for tp, p in zip(tgt.parameters(), src.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

# ================================================================
#  CELL 3: Training Logic
# ================================================================
class SweepConfig:
    env_name = "ALE/Breakout-v5" # CORRECTED environment name
    lambda_ints_to_test = [0.0, 0.05, 0.1, 0.2]
    etas_to_test        = [1.0, 0.5]
    total_steps  = 1_000_000
    buffer_size  = 200_000
    warmup_steps = 5_000
    render_mode = None
    seed        = 42
    embed_dim   = 256
    batch_size  = 32
    checkpoint_path = 'breakout_checkpoint.pth' # Path for saving progress

def run_experiment(config: SweepConfig):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    make_kwargs = dict(render_mode=config.render_mode)
    if config.env_name.startswith("ALE/"):
        make_kwargs["frameskip"] = 1

    env = gym.make(config.env_name, **make_kwargs)
    print(f"Using env ID: {config.env_name}")
    print("Applying AtariPreprocessing + FrameStack …")

    env = gym.wrappers.AtariPreprocessing(env, frame_skip=1, screen_size=84, grayscale_obs=True, scale_obs=False, noop_max=30)
    env = gym.wrappers.FrameStack(env, 4)

    encoder = CNNEncoder(in_channels=4, embed_dim=config.embed_dim)
    action_dim  = env.action_space.n

    agent = MaxMinExplorerAgent(action_dim=action_dim, device=device, encoder=encoder, buffer_capacity=config.buffer_size,
        batch_size=config.batch_size, embed_dim=config.embed_dim, lambda_int=config.lambda_int, eta_Q=config.eta_Q)

    reward_history = deque(maxlen=100)
    start_step = 1

    # --- NEW: LOAD CHECKPOINT IF IT EXISTS ---
    if os.path.exists(config.checkpoint_path):
        print(f"[INFO] Checkpoint found at {config.checkpoint_path}. Loading...")
        checkpoint = torch.load(config.checkpoint_path)
        agent.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        for i in range(len(agent.critics)):
            agent.critics[i].load_state_dict(checkpoint['critics_state_dict'][i])
            agent.target_critics[i].load_state_dict(checkpoint['target_critics_state_dict'][i])
        for i in range(len(agent.ensemble_models)):
            agent.ensemble_models[i].load_state_dict(checkpoint['ensemble_models_state_dict'][i])
        agent.opt_encoder.load_state_dict(checkpoint['opt_encoder_state_dict'])
        agent.opt_critic.load_state_dict(checkpoint['opt_critic_state_dict'])
        agent.opt_models.load_state_dict(checkpoint['opt_models_state_dict'])
        agent.replay.buffer = checkpoint['replay_buffer']
        start_step = checkpoint['step'] + 1
        episode_count = checkpoint['episode_count']
        print(f"[INFO] Resuming training from step {start_step}")
    else:
        print(f"[INFO] No checkpoint found. Starting from scratch.")
        episode_count = 1
        # --- Warm‑up if starting from scratch ---
        print(f"[INFO] Warming up replay buffer for {config.warmup_steps} steps …")
        state, _ = env.reset()
        for _ in range(config.warmup_steps):
            a = env.action_space.sample()
            s2, r, term, trunc, _ = env.step(a)
            agent.store_transition(state, a, r, s2, term or trunc)
            state, _ = env.reset() if term or trunc else (s2, None)

    print(f"[INFO] Starting training from step {start_step} to {config.total_steps:,} steps …")
    state, _ = env.reset()
    episode_reward = 0

    for step in range(start_step, config.total_steps + 1):
        a = agent.choose_action(state)
        s2, r, term, trunc, _ = env.step(a)
        done = term or trunc

        agent.store_transition(state, a, r, s2, done)
        agent.learn()

        state = s2
        episode_reward += r

        if step % 25_000 == 0:
            avg100 = np.mean(reward_history) if reward_history else 0.0
            print(f"Step {step:7d}/{config.total_steps:,} | Episodes {episode_count:5d} | Avg100 {avg100:5.1f}")

            # --- NEW: SAVE CHECKPOINT ---
            print(f"[INFO] Saving checkpoint at step {step}...")
            checkpoint = {
                'step': step,
                'episode_count': episode_count,
                'encoder_state_dict': agent.encoder.state_dict(),
                'critics_state_dict': [c.state_dict() for c in agent.critics],
                'target_critics_state_dict': [tc.state_dict() for tc in agent.target_critics],
                'ensemble_models_state_dict': [m.state_dict() for m in agent.ensemble_models],
                'opt_encoder_state_dict': agent.opt_encoder.state_dict(),
                'opt_critic_state_dict': agent.opt_critic.state_dict(),
                'opt_models_state_dict': agent.opt_models.state_dict(),
                'replay_buffer': agent.replay.buffer
            }
            torch.save(checkpoint, config.checkpoint_path)

        if done:
            reward_history.append(episode_reward)
            state, _      = env.reset()
            episode_reward = 0
            episode_count += 1

    return np.mean(reward_history) if reward_history else -999.0

if __name__ == "__main__":
    results = {}
    cfg = SweepConfig()
    # For a single run, you might want to simplify the sweep
    # For now, let's keep it to show how it works
    for lam in cfg.lambda_ints_to_test:
        for eta in cfg.etas_to_test:
            run_key = f"lambda={lam}_eta={eta}"
            print("\n" + "=" * 46)
            print(f"🚀 STARTING RUN: {run_key}")
            print("=" * 46 + "\n")

            cfg.lambda_int = lam
            cfg.eta_Q = eta

            t0 = time.time()
            score = run_experiment(cfg)
            t1 = time.time()

            results[run_key] = score
            print(f"\n--- FINAL AVG SCORE for {run_key}: {score:.2f} (elapsed {(t1 - t0)/60:.1f} min) ---")

    print("\n\n🏆 SWEEP COMPLETE 🏆")
    best = max(results, key=results.get)
    print(f"Best run: {best}  |  Avg100 = {results[best]:.2f}\n")
    print("Full results:")
    for k, v in sorted(results.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {k}: {v:.2f}")