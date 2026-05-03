"""ランダム方策で N エピソード収集して buffer に格納する (Phase 1 結合テスト).

実行:
    uv run python scripts/collect_random.py

期待出力 (例):
    episode 0: length=500, total_r=12.34
    episode 1: length=500, total_r=8.21
    ...
    buffer total steps: 2500
    sampled batch shapes: {'obs': (4, 21, 3, 64, 64), 'action': (4, 20, 6), ...}
    OK

このスクリプトは env.py + replay.py の結合動作確認.
完成すれば Phase 1 完了.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from dreamer.env import DMCEnv
from dreamer.replay import ReplayBuffer


def collect_random_episode(env: DMCEnv, max_steps: int = 1000) -> dict:
    """ランダム方策で 1 episode 走らせて episode dict を返す."""
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
    env = DMCEnv("walker", "walk", seed=0)
    buf = ReplayBuffer(
        capacity=100_000,
        obs_shape=env.obs_shape,
        action_dim=env.action_dim,
    )

    for i in range(5):
        ep = collect_random_episode(env)
        buf.add_episode(ep)
        print(
            f"episode {i}: length={len(ep['action'])}, "
            f"total_r={ep['reward'].sum():.2f}"
        )

    print(f"buffer total steps: {len(buf)}")

    batch = buf.sample(batch_size=4, seq_len=20)
    shapes = {k: v.shape for k, v in batch.items()}
    print(f"sampled batch shapes: {shapes}")

    print("OK")


if __name__ == "__main__":
    main()
