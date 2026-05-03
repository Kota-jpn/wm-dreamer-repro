"""DMCEnv の動作確認スクリプト (Phase 1).

実行:
    uv run python scripts/check_env_wrapper.py

期待出力:
    obs: (64, 64, 3) uint8
    action_dim: 6
    obs after step: (64, 64, 3) uint8
    total reward: <数十程度の数字>
    OK
"""

import sys
from pathlib import Path

# src/ をパスに追加 (uv プロジェクトでは pyproject.toml で解決される想定だが念のため)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from dreamer.env import DMCEnv


def main():
    env = DMCEnv("walker", "walk", seed=0)

    # reset
    obs = env.reset()
    print(f"obs: {obs.shape} {obs.dtype}")
    assert obs.shape == (64, 64, 3), f"expected (64,64,3), got {obs.shape}"
    assert obs.dtype == np.uint8, f"expected uint8, got {obs.dtype}"

    print(f"action_dim: {env.action_dim}")
    assert env.action_dim == 6, f"walker should have 6, got {env.action_dim}"

    # step
    a = np.random.uniform(-1, 1, env.action_dim).astype(np.float32)
    obs, r, done, info = env.step(a)
    print(f"obs after step: {obs.shape} {obs.dtype}")
    print(f"first reward: {r:.4f}, done: {done}")
    assert isinstance(r, (int, float, np.floating)), f"reward type: {type(r)}"
    assert isinstance(done, bool), f"done type: {type(done)}"
    assert isinstance(info, dict), f"info type: {type(info)}"

    # 100 step rollout
    total = 0.0
    for _ in range(100):
        a = np.random.uniform(-1, 1, env.action_dim).astype(np.float32)
        obs, r, done, _ = env.step(a)
        total += r
        if done:
            break
    print(f"total reward (100 random steps): {total:.4f}")

    print("OK")


if __name__ == "__main__":
    main()
