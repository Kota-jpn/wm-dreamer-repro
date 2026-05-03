"""ReplayBuffer のテスト (Phase 1).

実行:
    uv run pytest tests/test_replay.py -v

これら全てを通すのが Phase 1 完了条件.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from dreamer.replay import ReplayBuffer


# ─────────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────────

OBS_SHAPE = (64, 64, 3)
ACTION_DIM = 6


def make_episode(length: int) -> dict:
    """ダミーのエピソード."""
    return {
        "obs": np.random.randint(
            0, 256, (length + 1, *OBS_SHAPE), dtype=np.uint8
        ),
        "action": np.random.uniform(
            -1, 1, (length, ACTION_DIM)
        ).astype(np.float32),
        "reward": np.random.randn(length).astype(np.float32),
        "discount": np.ones(length, dtype=np.float32),
    }


@pytest.fixture
def buf() -> ReplayBuffer:
    return ReplayBuffer(
        capacity=10_000,
        obs_shape=OBS_SHAPE,
        action_dim=ACTION_DIM,
    )


# ─────────────────────────────────────────────────────────────────────
# add_episode
# ─────────────────────────────────────────────────────────────────────


def test_add_and_len(buf):
    """1 episode を入れたら len(buf) == episode 長."""
    ep = make_episode(50)
    buf.add_episode(ep)
    assert len(buf) == 50
    assert buf.num_episodes() == 1


def test_capacity_eviction():
    """capacity 超過で古い episode が消える."""
    buf = ReplayBuffer(capacity=120, obs_shape=OBS_SHAPE, action_dim=ACTION_DIM)
    buf.add_episode(make_episode(50))
    buf.add_episode(make_episode(50))
    buf.add_episode(make_episode(50))  # 合計 150 > 120 → 最初が消える
    assert buf.num_episodes() == 2
    assert len(buf) == 100


# ─────────────────────────────────────────────────────────────────────
# sample
# ─────────────────────────────────────────────────────────────────────


def test_sample_shapes(buf):
    """サンプルの形状が正しい."""
    buf.add_episode(make_episode(100))
    buf.add_episode(make_episode(80))

    batch = buf.sample(batch_size=4, seq_len=20)

    # obs は (B, T+1, C, H, W) (CHW 変換後)
    assert batch["obs"].shape == (4, 21, 3, 64, 64), f"got {batch['obs'].shape}"
    assert batch["action"].shape == (4, 20, 6)
    assert batch["reward"].shape == (4, 20)
    assert batch["discount"].shape == (4, 20)


def test_sample_dtype(buf):
    """dtype が float32 (obs を含めて)."""
    buf.add_episode(make_episode(100))
    batch = buf.sample(batch_size=4, seq_len=20)
    assert batch["obs"].dtype == np.float32
    assert batch["action"].dtype == np.float32
    assert batch["reward"].dtype == np.float32
    assert batch["discount"].dtype == np.float32


def test_sample_range(buf):
    """obs が [-0.5, 0.5] にスケールされている."""
    buf.add_episode(make_episode(100))
    batch = buf.sample(batch_size=4, seq_len=20)
    assert batch["obs"].min() >= -0.5 - 1e-5
    assert batch["obs"].max() <= 0.5 + 1e-5


def test_sample_randomness(buf):
    """異なる呼び出しで異なるバッチが返る (高確率で)."""
    buf.add_episode(make_episode(200))
    b1 = buf.sample(batch_size=4, seq_len=20)
    b2 = buf.sample(batch_size=4, seq_len=20)
    # 完全一致する確率は天文学的に低い
    assert not np.array_equal(b1["action"], b2["action"])
