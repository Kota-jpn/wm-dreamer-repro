"""Agent のテスト (Phase 7).

実行:
    uv run pytest tests/test_agent.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from dreamer.agent import Agent


# ─────────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────────

A_DIM = 6


def make_agent() -> Agent:
    return Agent(
        action_dim=A_DIM,
        cnn_depth=32,
        rssm_deter=200,
        rssm_stoch=30,
        rssm_hidden=200,
        min_std=0.1,
        num_units=400,
        num_layers=3,
        device="cpu",
    )


def make_obs() -> np.ndarray:
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


def make_batch(B: int = 4, T: int = 8) -> dict:
    return {
        "obs": torch.randn(B, T + 1, 3, 64, 64) * 0.3,
        "action": torch.randn(B, T, A_DIM) * 0.3,
        "reward": torch.randn(B, T),
        "discount": torch.ones(B, T),
    }


# ─────────────────────────────────────────────────────────────────────
# __init__ + reset
# ─────────────────────────────────────────────────────────────────────


def test_init():
    """Agent が wm, ac, 3 つの optimizer を持つ."""
    agent = make_agent()
    assert hasattr(agent, "wm")
    assert hasattr(agent, "ac")
    assert hasattr(agent, "wm_opt")
    assert hasattr(agent, "actor_opt")
    assert hasattr(agent, "critic_opt")


def test_reset_clears_state():
    """reset で _state, _prev_action が None に戻る."""
    agent = make_agent()
    obs = make_obs()
    agent.act(obs, training=True)  # _state が埋まる
    assert agent._state is not None
    agent.reset()
    assert agent._state is None
    assert agent._prev_action is None


# ─────────────────────────────────────────────────────────────────────
# act
# ─────────────────────────────────────────────────────────────────────


def test_act_shape():
    """act が numpy (A,) を返す."""
    agent = make_agent()
    obs = make_obs()
    a = agent.act(obs, training=True)
    assert isinstance(a, np.ndarray)
    assert a.shape == (A_DIM,)
    assert a.dtype == np.float32 or a.dtype == np.float64


def test_act_range():
    """action が [-1, 1] (clip + tanh)."""
    agent = make_agent()
    obs = make_obs()
    for _ in range(5):
        a = agent.act(obs, training=True)
        assert (a >= -1).all() and (a <= 1).all()


def test_act_eval_deterministic():
    """training=False は同じ入力に対して deterministic (seed 固定時)."""
    agent = make_agent()
    obs = make_obs()
    
    agent.reset()
    torch.manual_seed(0)
    a1 = agent.act(obs, training=False)
    
    agent.reset()
    torch.manual_seed(0)
    a2 = agent.act(obs, training=False)
    
    # 同じ初期状態 + 同じ obs + 同じ seed なら同じ action
    np.testing.assert_allclose(a1, a2, atol=1e-5)


def test_act_state_persistence():
    """連続 act で _state が更新される (zero ではなくなる)."""
    agent = make_agent()
    agent.reset()
    obs = make_obs()
    agent.act(obs, training=True)
    state_after_1 = agent._state["deter"].clone()
    agent.act(obs, training=True)
    state_after_2 = agent._state["deter"].clone()
    # 2 つの state は異なる (RSSM が進んでいる)
    assert not torch.allclose(state_after_1, state_after_2)


def test_act_training_vs_eval_differ():
    """training=True と False で action が異なる (探索ノイズが効いている)."""
    agent = make_agent()
    agent.expl_noise = 0.5  # ノイズ大きく
    obs = make_obs()
    agent.reset()
    a_train = agent.act(obs, training=True)
    agent.reset()
    a_eval = agent.act(obs, training=False)
    # 高確率で異なる
    assert not np.allclose(a_train, a_eval, atol=1e-3)


# ─────────────────────────────────────────────────────────────────────
# learn
# ─────────────────────────────────────────────────────────────────────


def test_learn_returns_metrics():
    """learn が WM と AC の metrics をマージして返す."""
    agent = make_agent()
    batch = make_batch(B=2, T=5)
    metrics = agent.learn(batch)

    # WM 由来
    assert "wm/total" in metrics
    assert "wm/recon" in metrics
    assert "wm/reward" in metrics
    assert "wm/kl" in metrics
    # AC 由来
    assert "ac/actor" in metrics
    assert "ac/critic" in metrics


def test_learn_changes_params():
    """learn でパラメータが変化."""
    torch.manual_seed(0)
    agent = make_agent()
    batch = make_batch(B=2, T=5)

    # WM, actor, critic それぞれの first param をスナップショット
    wm_before = agent.wm.encoder.layers[0].weight.detach().clone()
    actor_first = [m for m in agent.ac.actor.mlp if isinstance(m, torch.nn.Linear)][0]
    actor_before = actor_first.weight.detach().clone()
    critic_first = [m for m in agent.ac.critic.mlp if isinstance(m, torch.nn.Linear)][0]
    critic_before = critic_first.weight.detach().clone()

    agent.learn(batch)

    assert not torch.allclose(wm_before, agent.wm.encoder.layers[0].weight)
    assert not torch.allclose(actor_before, actor_first.weight)
    assert not torch.allclose(critic_before, critic_first.weight)


# ─────────────────────────────────────────────────────────────────────
# 統合: act + learn の交互呼び出し (env interaction simulation)
# ─────────────────────────────────────────────────────────────────────


def test_act_then_learn_no_crash():
    """env step を simulate しながら learn を呼んで crash しない."""
    agent = make_agent()
    agent.reset()
    obs = make_obs()
    for _ in range(5):
        a = agent.act(obs, training=True)
        # 次の obs (random)
        obs = make_obs()
    # buffer 相当
    batch = make_batch(B=2, T=5)
    metrics = agent.learn(batch)
    assert "wm/total" in metrics


def test_save_load(tmp_path):
    """save → load で state が一致."""
    agent = make_agent()
    obs = make_obs()
    agent.act(obs, training=False)  # 適当に動かす

    # 保存
    ckpt_path = str(tmp_path / "agent.pt")
    agent.save(ckpt_path)

    # 新しい agent に load
    agent2 = make_agent()
    agent2.load(ckpt_path)

    # 同じ obs に対して同じ action が出る
    agent.reset()
    agent2.reset()
    
    torch.manual_seed(0)
    a1 = agent.act(obs, training=False)
    
    torch.manual_seed(0)
    a2 = agent2.act(obs, training=False)
    
    np.testing.assert_allclose(a1, a2, atol=1e-5)
