"""RSSM の smoke test (Phase 3).

実行:
    uv run python scripts/smoke_rssm.py

何をするか:
    1. 5 ランダムエピソードを DMC walker_walk から収集
    2. ConvEncoder + RSSM を組み合わせ:
         embed = encoder(obs)
         priors, posts = rssm.observe(embed, action, init_state)
         loss = KL(post || prior) + recon_loss
    3. 1000 step Adam で学習
    4. KL が下がることを確認 (初期 ~10, 最終 ~3 程度)
    5. 学習曲線を results/rssm_smoke.png に保存

期待出力:
    Step    0 | kl=12.34 | recon=0.082
    Step  100 | kl=4.56  | recon=0.045
    ...
    Step  900 | kl=2.10  | recon=0.018
    Saved curve to: results/rssm_smoke.png
    OK
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from dreamer.env import DMCEnv
from dreamer.replay import ReplayBuffer
from dreamer.encoder import ConvEncoder
from dreamer.decoder import ConvDecoder
from dreamer.rssm import RSSM


def collect_random_episode(env: DMCEnv, max_steps: int = 1000) -> dict:
    obs_list, action_list, reward_list, discount_list = [], [], [], []
    obs = env.reset()
    obs_list.append(obs)
    for _ in range(max_steps):
        a = np.random.uniform(-1, 1, env.action_dim).astype(np.float32)
        obs, r, done, _ = env.step(a)
        obs_list.append(obs)
        action_list.append(a)
        reward_list.append(r)
        discount_list.append(0.0 if done else 1.0)
        if done:
            break
    return {
        "obs": np.stack(obs_list).astype(np.uint8),
        "action": np.stack(action_list).astype(np.float32),
        "reward": np.array(reward_list, dtype=np.float32),
        "discount": np.array(discount_list, dtype=np.float32),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # 1. 経験収集
    print("collecting 5 random episodes...")
    env = DMCEnv("walker", "walk", seed=0)
    buf = ReplayBuffer(
        capacity=100_000,
        obs_shape=env.obs_shape,
        action_dim=env.action_dim,
    )
    for i in range(5):
        ep = collect_random_episode(env)
        buf.add_episode(ep)
        print(f"  episode {i}: length={len(ep['action'])}")

    # 2. モデル
    enc = ConvEncoder(depth=32).to(device)
    rssm = RSSM(
        action_dim=env.action_dim,
        embed_dim=enc.embed_dim,
        deter_dim=200,
        stoch_dim=30,
        hidden_dim=200,
        min_std=0.1,
    ).to(device)
    dec = ConvDecoder(feat_dim=rssm.feat_dim, depth=32).to(device)

    params = list(enc.parameters()) + list(rssm.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=6e-4)

    # 3. 学習
    print("\ntraining encoder + rssm + decoder...")
    kl_log, recon_log, total_log = [], [], []

    B = 16
    T = 20
    for step in range(1000):
        batch = buf.sample(batch_size=B, seq_len=T)
        obs_seq = torch.from_numpy(batch["obs"][:, :T]).to(device)  # (B, T, 3, 64, 64)
        action_seq = torch.from_numpy(batch["action"]).to(device)   # (B, T, A)

        # encode → observe
        embed = enc(obs_seq)
        init = rssm.initial_state(batch_size=B, device=torch.device(device))
        priors, posts = rssm.observe(embed, action_seq, init)

        # 再構成
        feat = rssm.get_feat(posts)  # (B, T, deter+stoch)
        recon = dec(feat)
        recon_loss = F.mse_loss(recon, obs_seq)

        # KL (free nats=0 で純粋な下がり方を見る)
        kl = rssm.kl_loss(posts, priors, free_nats=0.0)

        loss = recon_loss + kl

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 100.0)
        opt.step()

        kl_log.append(kl.item())
        recon_log.append(recon_loss.item())
        total_log.append(loss.item())

        if step % 100 == 0:
            print(
                f"  Step {step:>4d} | kl={kl.item():.3f} | "
                f"recon={recon_loss.item():.4f}"
            )

    print(
        f"  Step  999 | kl={kl_log[-1]:.3f} | recon={recon_log[-1]:.4f}"
    )

    # 4. 学習曲線
    Path("results").mkdir(exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(kl_log)
        axes[0].set_title("KL(post || prior)")
        axes[0].set_xlabel("step")
        axes[1].plot(recon_log)
        axes[1].set_title("Recon MSE")
        axes[1].set_xlabel("step")
        plt.tight_layout()
        plt.savefig("results/rssm_smoke.png", dpi=100)
        print("Saved curve to: results/rssm_smoke.png")
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        print(f"KL log (every 100 step): {kl_log[::100]}")

    print(f"\nKL: {kl_log[0]:.3f} → {kl_log[-1]:.3f} (ratio {kl_log[-1]/kl_log[0]:.3f})")
    print(f"Recon: {recon_log[0]:.4f} → {recon_log[-1]:.4f}")
    if kl_log[-1] < kl_log[0] and recon_log[-1] < recon_log[0]:
        print("OK")
    else:
        print("WARNING: did not decrease.")


if __name__ == "__main__":
    main()
