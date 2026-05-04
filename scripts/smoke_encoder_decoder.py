"""Encoder + Decoder の smoke test (Phase 2).

実行:
    uv run python scripts/smoke_encoder_decoder.py

何をするか:
    1. 5 ランダムエピソードを DMC walker_walk から収集
    2. ReplayBuffer に格納
    3. random batch を取り出して Encoder + Decoder を 1000 step 学習
    4. 初期と最終の MSE loss を比較
    5. 再構成画像 (元 vs 復元) を results/recon_smoke.png に保存

期待出力:
    Step    0 | loss=0.083
    Step  100 | loss=0.054
    ...
    Step  900 | loss=0.012
    Saved comparison image to: results/recon_smoke.png
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


def save_recon_grid(obs: np.ndarray, recon: np.ndarray, path: str):
    """並列の比較画像を保存.

    obs:   (N, 3, 64, 64) float in [-0.5, 0.5]
    recon: (N, 3, 64, 64) float
    """
    import imageio

    # [-0.5, 0.5] → [0, 255] uint8
    def to_img(x):
        x = np.clip(x + 0.5, 0, 1)
        x = (x * 255).astype(np.uint8)
        x = np.transpose(x, (0, 2, 3, 1))  # NCHW → NHWC
        return x

    obs_imgs = to_img(obs)      # (N, 64, 64, 3)
    recon_imgs = to_img(recon)
    # 縦並び: 上が元、下が復元
    pair = np.concatenate([obs_imgs, recon_imgs], axis=1)  # (N, 128, 64, 3)
    grid = np.concatenate(list(pair), axis=1)              # (128, 64*N, 3)
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

    # 2. モデル
    enc = ConvEncoder(depth=32).to(device)
    dec = ConvDecoder(feat_dim=enc.embed_dim, depth=32).to(device)
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=6e-4)

    # 3. 学習
    print("\ntraining encoder + decoder...")
    losses = []
    for step in range(1000):
        batch = buf.sample(batch_size=16, seq_len=10)
        obs = torch.from_numpy(batch["obs"][:, :-1]).to(device)  # (B, T, 3, 64, 64)

        embed = enc(obs)
        recon = dec(embed)
        loss = F.mse_loss(recon, obs)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        if step % 100 == 0:
            print(f"  Step {step:>4d} | loss={loss.item():.4f}")

    print(f"  Step  999 | loss={losses[-1]:.4f}")

    # 4. 可視化
    enc.eval()
    dec.eval()
    with torch.no_grad():
        batch = buf.sample(batch_size=8, seq_len=1)
        obs = torch.from_numpy(batch["obs"][:, :1]).to(device)  # (8, 1, 3, 64, 64)
        embed = enc(obs)
        recon = dec(embed)

        obs_np = obs.cpu().numpy()[:, 0]      # (8, 3, 64, 64)
        recon_np = recon.cpu().numpy()[:, 0]

    Path("results").mkdir(exist_ok=True)
    save_recon_grid(obs_np, recon_np, "results/recon_smoke.png")
    print("\nSaved comparison image to: results/recon_smoke.png")
    print(f"  initial loss: {losses[0]:.4f}")
    print(f"  final loss:   {losses[-1]:.4f}")
    print(f"  ratio: {losses[-1] / losses[0]:.3f}")
    if losses[-1] < losses[0] * 0.5:
        print("OK")
    else:
        print("WARNING: loss did not decrease enough.")


if __name__ == "__main__":
    main()
