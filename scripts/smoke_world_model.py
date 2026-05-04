"""WorldModel の smoke test (Phase 4).

実行:
    uv run python scripts/smoke_world_model.py

何をするか:
    1. 5 ランダムエピソードを DMC walker_walk から収集
    2. WorldModel (Encoder + RSSM + Decoder + RewardHead) を構築
    3. 1000 step 学習
    4. 各損失成分の学習曲線を results/wm_smoke.png に保存
    5. 再構成画像 vs 元画像の比較を results/wm_recon.png に保存

期待出力:
    Step    0 | total=400.0 | recon=350.0 | reward=0.5 | kl=12.0
    Step  100 | total= 80.0 | recon= 65.0 | reward=0.3 | kl=5.0
    ...
    Step  900 | total= 30.0 | recon= 25.0 | reward=0.1 | kl=3.0  (free_nats=3 でクランプ)
    Saved curve to: results/wm_smoke.png
    Saved recon to: results/wm_recon.png
    OK
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from dreamer.env import DMCEnv
from dreamer.replay import ReplayBuffer
from dreamer.world_model import WorldModel


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


def to_tensor_batch(np_batch: dict, device: str) -> dict:
    return {k: torch.from_numpy(v).to(device) for k, v in np_batch.items()}


def save_recon_grid(obs: np.ndarray, recon: np.ndarray, path: str):
    import imageio

    def to_img(x):
        x = np.clip(x + 0.5, 0, 1)
        x = (x * 255).astype(np.uint8)
        x = np.transpose(x, (0, 2, 3, 1))
        return x

    obs_imgs = to_img(obs)
    recon_imgs = to_img(recon)
    pair = np.concatenate([obs_imgs, recon_imgs], axis=1)
    grid = np.concatenate(list(pair), axis=1)
    imageio.imwrite(path, grid)


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

    # 2. WorldModel
    wm = WorldModel(
        action_dim=env.action_dim,
        cnn_depth=32,
        rssm_deter=200,
        rssm_stoch=30,
        rssm_hidden=200,
        min_std=0.1,
        num_units=400,
        num_layers=3,
    ).to(device)
    opt = torch.optim.Adam(wm.parameters(), lr=6e-4)

    # 3. 学習
    print("\ntraining world model...")
    logs = {"total": [], "recon": [], "reward": [], "kl": []}

    B, T = 16, 20
    for step in range(1000):
        np_batch = buf.sample(batch_size=B, seq_len=T)
        batch = to_tensor_batch(np_batch, device)

        metrics, _ = wm.update(
            batch, opt, free_nats=3.0, kl_scale=1.0, grad_clip=100.0
        )

        logs["total"].append(metrics["wm/total"])
        logs["recon"].append(metrics["wm/recon"])
        logs["reward"].append(metrics["wm/reward"])
        logs["kl"].append(metrics["wm/kl"])

        if step % 100 == 0:
            print(
                f"  Step {step:>4d} | "
                f"total={metrics['wm/total']:>7.2f} | "
                f"recon={metrics['wm/recon']:>7.2f} | "
                f"reward={metrics['wm/reward']:>5.2f} | "
                f"kl={metrics['wm/kl']:>5.2f}"
            )

    print(
        f"  Step  999 | "
        f"total={logs['total'][-1]:>7.2f} | "
        f"recon={logs['recon'][-1]:>7.2f} | "
        f"reward={logs['reward'][-1]:>5.2f} | "
        f"kl={logs['kl'][-1]:>5.2f}"
    )

    # 4. 学習曲線
    Path("results").mkdir(exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, key in zip(axes.flat, ["total", "recon", "reward", "kl"]):
            ax.plot(logs[key])
            ax.set_title(f"WM {key}")
            ax.set_xlabel("step")
        plt.tight_layout()
        plt.savefig("results/wm_smoke.png", dpi=100)
        print("\nSaved curve to: results/wm_smoke.png")
    except ImportError:
        print("matplotlib not installed; skipping plot.")

    # 5. 再構成画像の確認
    wm.eval()
    with torch.no_grad():
        np_batch = buf.sample(batch_size=8, seq_len=5)
        batch = to_tensor_batch(np_batch, device)
        T_eval = batch["action"].shape[1]
        obs_seq = batch["obs"][:, :T_eval]
        priors, posts, embed = wm.observe(obs_seq, batch["action"])
        feat = wm.rssm.get_feat(posts)
        recon = wm.decoder(feat)

        # 各サンプルの t=2 を取る
        obs_np = obs_seq[:, 2].cpu().numpy()
        recon_np = recon[:, 2].cpu().numpy()

    save_recon_grid(obs_np, recon_np, "results/wm_recon.png")
    print("Saved recon to: results/wm_recon.png")

    print(f"\ntotal: {logs['total'][0]:.2f} → {logs['total'][-1]:.2f}")
    print(f"recon: {logs['recon'][0]:.2f} → {logs['recon'][-1]:.2f}")
    print(f"reward: {logs['reward'][0]:.4f} → {logs['reward'][-1]:.4f}")
    print(f"kl: {logs['kl'][0]:.2f} → {logs['kl'][-1]:.2f}")
    if logs["recon"][-1] < logs["recon"][0] * 0.5:
        print("OK")
    else:
        print("WARNING: recon did not decrease enough.")


if __name__ == "__main__":
    main()
