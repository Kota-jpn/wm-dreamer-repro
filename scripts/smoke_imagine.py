"""Imagination rollout の smoke test (Phase 5).

実行:
    uv run python scripts/smoke_imagine.py

何をするか:
    1. 5 ランダムエピソードを DMC walker_walk から収集
    2. WorldModel を 1000 step 学習 (Phase 4 と同じ)
    3. 学習済み WM で imagine_rollout を 15 step 実行
    4. 想像 rollout の各ステップを decoder で復号
    5. 元軌跡 vs 想像軌跡を並べて results/imagine_rollout.gif に保存

期待出力:
    pretraining world model 1000 steps...
    Step    0 | recon=350.0 | kl=12.0
    ...
    Step  900 | recon= 25.0 | kl= 3.0
    running imagination rollout (horizon=15)...
    saved: results/imagine_rollout.gif
    saved: results/imagine_grid.png
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


def to_uint8(x: np.ndarray) -> np.ndarray:
    """[-0.5, 0.5] CHW → [0, 255] HWC uint8."""
    x = np.clip(x + 0.5, 0, 1)
    x = (x * 255).astype(np.uint8)
    return np.transpose(x, (1, 2, 0))  # CHW → HWC


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

    # 2. WM 学習 (Phase 4 と同じ)
    wm = WorldModel(action_dim=env.action_dim).to(device)
    opt = torch.optim.Adam(wm.parameters(), lr=6e-4)

    print("\npretraining world model 1000 steps...")
    for step in range(1000):
        np_batch = buf.sample(batch_size=16, seq_len=20)
        batch = to_tensor_batch(np_batch, device)
        metrics, _ = wm.update(
            batch, opt, free_nats=3.0, kl_scale=1.0, grad_clip=100.0
        )
        if step % 100 == 0:
            print(
                f"  Step {step:>4d} | "
                f"recon={metrics['wm/recon']:>7.2f} | "
                f"kl={metrics['wm/kl']:>5.2f}"
            )

    # 3. 想像 rollout
    print("\nrunning imagination rollout (horizon=15)...")
    wm.eval()

    # ランダム方策 (Phase 5 用. Phase 6 で actor に置き換え)
    def random_policy(feat: torch.Tensor) -> torch.Tensor:
        return torch.randn(feat.shape[0], env.action_dim, device=feat.device) * 0.5

    with torch.no_grad():
        # 開始点: 1 episode の最初の数フレーム posterior を使う
        np_batch = buf.sample(batch_size=4, seq_len=5)
        batch = to_tensor_batch(np_batch, device)
        T = batch["action"].shape[1]
        obs_seq = batch["obs"][:, :T]  # (4, 5, 3, 64, 64)
        _, posts, _ = wm.observe(obs_seq, batch["action"])

        # 最後の posterior を起点にする (B=4, ...)
        init_state = {k: v[:, -1] for k, v in posts.items()}

        # imagine
        out = wm.imagine_rollout(init_state, random_policy, horizon=15)

        # decoder で復号 → 画像化
        imag_feat = out["feat"]  # (4, 15, F)
        imag_recon = wm.decoder(imag_feat)  # (4, 15, 3, 64, 64)
        imag_recon_np = imag_recon.cpu().numpy()

        # 比較用: 同じ batch の "実際の続き" は無いので、起点の前のフレームを並べる
        actual_np = obs_seq.cpu().numpy()  # (4, 5, 3, 64, 64)

    Path("results").mkdir(exist_ok=True)

    # 4a. GIF: imagine 系列 (sample 0)
    try:
        import imageio
        frames = []
        for h in range(15):
            frame = to_uint8(imag_recon_np[0, h])
            frames.append(frame)
        imageio.mimwrite("results/imagine_rollout.gif", frames, duration=0.1)
        print("saved: results/imagine_rollout.gif")
    except Exception as e:
        print(f"GIF save failed: {e}")

    # 4b. 横並び PNG: 元 5 frame + 想像 15 frame
    try:
        import imageio
        rows = []
        for b in range(4):
            actual_strip = np.concatenate(
                [to_uint8(actual_np[b, t]) for t in range(5)], axis=1
            )
            imag_strip = np.concatenate(
                [to_uint8(imag_recon_np[b, h]) for h in range(15)], axis=1
            )
            # 縦に並べる: 上が元 5 frame、下が想像 15 frame (列数が違うので別行)
            sep = np.zeros((4, max(actual_strip.shape[1], imag_strip.shape[1]), 3),
                           dtype=np.uint8)
            row = np.concatenate(
                [
                    np.pad(actual_strip,
                           ((0, 0), (0, max(0, imag_strip.shape[1] - actual_strip.shape[1])), (0, 0))),
                    sep,
                    np.pad(imag_strip,
                           ((0, 0), (0, max(0, actual_strip.shape[1] - imag_strip.shape[1])), (0, 0))),
                ],
                axis=0,
            )
            rows.append(row)
        grid = np.concatenate(rows, axis=0)
        imageio.imwrite("results/imagine_grid.png", grid)
        print("saved: results/imagine_grid.png  (上=元 5 frame, 下=想像 15 frame)")
    except Exception as e:
        print(f"PNG save failed: {e}")

    # 5. 数値チェック
    final_norm = out["states"]["deter"][:, -1].norm(dim=-1).max().item()
    has_nan = (
        torch.isnan(out["feat"]).any().item()
        or torch.isnan(out["reward"]).any().item()
    )
    print(f"\nfinal step deter max norm: {final_norm:.2f}")
    print(f"NaN in trajectory: {has_nan}")
    print(f"reward range: [{out['reward'].min():.3f}, {out['reward'].max():.3f}]")

    if not has_nan and final_norm < 100.0:
        print("OK")
    else:
        print("WARNING: imagination unstable.")


if __name__ == "__main__":
    main()
