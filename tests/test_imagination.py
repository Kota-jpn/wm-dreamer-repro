"""Imagination rollout のテスト (Phase 5).

実行:
    uv run pytest tests/test_imagination.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from dreamer.world_model import WorldModel


# ─────────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────────

A_DIM = 6  # walker
H_DIM = 200
Z_DIM = 30


def make_wm() -> WorldModel:
    return WorldModel(
        action_dim=A_DIM,
        cnn_depth=32,
        rssm_deter=H_DIM,
        rssm_stoch=Z_DIM,
        rssm_hidden=200,
        min_std=0.1,
        num_units=400,
        num_layers=3,
    )


def make_init_state(B: int = 4) -> dict:
    """ランダムな初期 state dict を作る."""
    return {
        "deter": torch.randn(B, H_DIM) * 0.1,
        "stoch": torch.randn(B, Z_DIM) * 0.1,
        "mean": torch.zeros(B, Z_DIM),
        "std": torch.ones(B, Z_DIM),
    }


def random_policy(feat: torch.Tensor) -> torch.Tensor:
    """テスト用ランダム方策. feat (B, F) → action (B, A)."""
    B = feat.shape[0]
    return torch.randn(B, A_DIM) * 0.3


# ─────────────────────────────────────────────────────────────────────
# 形状テスト
# ─────────────────────────────────────────────────────────────────────


def test_imagine_rollout_shape():
    """imagine_rollout の出力 dict の形状."""
    wm = make_wm()
    init = make_init_state(B=4)
    out = wm.imagine_rollout(init, random_policy, horizon=15)

    assert "states" in out
    assert "feat" in out
    assert "action" in out
    assert "reward" in out

    assert out["feat"].shape == (4, 15, H_DIM + Z_DIM)
    assert out["action"].shape == (4, 15, A_DIM)
    assert out["reward"].shape == (4, 15)
    assert out["states"]["deter"].shape == (4, 15, H_DIM)
    assert out["states"]["stoch"].shape == (4, 15, Z_DIM)


def test_imagine_horizon_variants():
    """horizon を変えても動く."""
    wm = make_wm()
    init = make_init_state(B=2)
    for H in (1, 5, 30):
        out = wm.imagine_rollout(init, random_policy, horizon=H)
        assert out["feat"].shape == (2, H, H_DIM + Z_DIM)


# ─────────────────────────────────────────────────────────────────────
# 数値安定性
# ─────────────────────────────────────────────────────────────────────


def test_imagine_no_nan():
    """15 step rollout で NaN/Inf が出ない."""
    wm = make_wm()
    init = make_init_state(B=8)
    out = wm.imagine_rollout(init, random_policy, horizon=15)

    for key in ("feat", "action", "reward"):
        t = out[key]
        assert not torch.isnan(t).any(), f"{key} has NaN"
        assert not torch.isinf(t).any(), f"{key} has Inf"
    for k in ("deter", "stoch"):
        t = out["states"][k]
        assert not torch.isnan(t).any(), f"states.{k} has NaN"


def test_imagine_long_horizon_stable():
    """50 step rollout でも暴走しない (norm が ∞ にならない)."""
    wm = make_wm()
    init = make_init_state(B=4)
    out = wm.imagine_rollout(init, random_policy, horizon=50)
    # final step の deter のノルムが 1000 以下程度
    final_deter_norm = out["states"]["deter"][:, -1].norm(dim=-1).max().item()
    assert final_deter_norm < 1000.0, f"deter exploded: {final_deter_norm}"


# ─────────────────────────────────────────────────────────────────────
# 勾配伝播 (Phase 6 で actor 学習に使うために重要)
# ─────────────────────────────────────────────────────────────────────


def test_imagine_grad_to_policy():
    """policy_fn のパラメータに勾配が流れる (Phase 6 actor 学習の前提)."""
    wm = make_wm()

    # 学習可能な policy (= 簡易 actor)
    policy_net = torch.nn.Linear(H_DIM + Z_DIM, A_DIM)

    def learnable_policy(feat: torch.Tensor) -> torch.Tensor:
        return torch.tanh(policy_net(feat))

    init = make_init_state(B=4)
    out = wm.imagine_rollout(init, learnable_policy, horizon=10)

    # 報酬の合計を最大化したい
    loss = -out["reward"].mean()
    loss.backward()

    assert policy_net.weight.grad is not None
    assert policy_net.weight.grad.abs().sum() > 0, "no gradient flowed to policy"


def test_imagine_grad_to_world_model():
    """WM のパラメータにも勾配が流れる (rssm img_step, reward_head)."""
    wm = make_wm()
    init = make_init_state(B=4)
    out = wm.imagine_rollout(init, random_policy, horizon=10)

    out["reward"].sum().backward()

    # rssm の img_step を通る weight に勾配
    assert wm.rssm.fc_input.weight.grad is not None
    assert wm.rssm.fc_input.weight.grad.abs().sum() > 0
    # reward_head にも
    first_linear = next(m for m in wm.reward_head.mlp if isinstance(m, torch.nn.Linear))
    assert first_linear.weight.grad is not None


# ─────────────────────────────────────────────────────────────────────
# flatten_state_for_imagine helper
# ─────────────────────────────────────────────────────────────────────


def test_flatten_state_for_imagine():
    """(B, T, ...) → (B*T, ...)."""
    wm = make_wm()
    state = {
        "deter": torch.randn(4, 5, H_DIM),
        "stoch": torch.randn(4, 5, Z_DIM),
        "mean": torch.randn(4, 5, Z_DIM),
        "std": torch.ones(4, 5, Z_DIM),
    }
    flat = wm.flatten_state_for_imagine(state)
    assert flat["deter"].shape == (20, H_DIM)
    assert flat["stoch"].shape == (20, Z_DIM)


def test_imagine_with_flattened_post():
    """observe → flatten → imagine の流れが動く (Phase 6 と同じパターン)."""
    wm = make_wm()
    B, T = 2, 4

    # 1. observe
    obs = torch.randn(B, T, 3, 64, 64) * 0.3
    action = torch.randn(B, T, A_DIM) * 0.3
    _, posts, _ = wm.observe(obs, action)

    # 2. flatten posts (B, T, ...) → (B*T, ...)
    init = wm.flatten_state_for_imagine(posts)
    assert init["deter"].shape == (B * T, H_DIM)

    # 3. imagine
    out = wm.imagine_rollout(init, random_policy, horizon=15)
    assert out["feat"].shape == (B * T, 15, H_DIM + Z_DIM)
