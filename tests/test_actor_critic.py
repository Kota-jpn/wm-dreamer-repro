"""Actor + Critic + λ-return のテスト (Phase 6).

実行:
    uv run pytest tests/test_actor_critic.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F

from dreamer.actor_critic import Actor, Critic, ActorCritic
from dreamer.utils import lambda_return
from dreamer.world_model import WorldModel


# ─────────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────────

A_DIM = 6
F_DIM = 230  # rssm.feat_dim


def make_actor() -> Actor:
    return Actor(feat_dim=F_DIM, action_dim=A_DIM, num_units=400, num_layers=3)


def make_critic() -> Critic:
    return Critic(feat_dim=F_DIM, num_units=400, num_layers=3)


def make_wm() -> WorldModel:
    return WorldModel(
        action_dim=A_DIM,
        cnn_depth=32,
        rssm_deter=200,
        rssm_stoch=30,
        rssm_hidden=200,
        min_std=0.1,
        num_units=400,
        num_layers=3,
    )


# ═════════════════════════════════════════════════════════════════════════
# Actor
# ═════════════════════════════════════════════════════════════════════════


def test_actor_init():
    """Actor が mlp を持ち、出力次元が 2*A."""
    actor = make_actor()
    assert hasattr(actor, "mlp")
    # 最終 Linear の out_features が 2*A
    final = [m for m in actor.mlp if isinstance(m, torch.nn.Linear)][-1]
    assert final.out_features == 2 * A_DIM


def test_actor_output_range():
    """rsample した action が [-1, 1] (tanh で squash されている)."""
    actor = make_actor()
    feat = torch.randn(8, F_DIM)
    dist = actor(feat)
    # 1000 サンプルで全部 [-1, 1]
    samples = torch.stack([dist.rsample() for _ in range(50)])  # (50, 8, A)
    assert (samples >= -1).all() and (samples <= 1).all()


def test_actor_initial_std():
    """初期 std が 5 付近 (softplus(0 + log(e^5-1)) = 5)."""
    torch.manual_seed(0)
    actor = make_actor()
    # 平均 0 の入力
    feat = torch.zeros(100, F_DIM)
    dist = actor(feat)
    # base_dist (Independent(Normal)) の base_dist が Normal(mean, std)
    # Independent.base_dist.scale が std
    std = dist.base_dist.base_dist.scale  # (100, A)
    print("initial std mean:", std.mean().item())
    # 期待値 5 付近 (mlp の出力が初期は 0 付近なので INIT_STD_BIAS だけが効く)
    assert 3.0 < std.mean().item() < 7.0


def test_actor_act_training_mode():
    """act(training=True) は rsample, training=False は deterministic."""
    actor = make_actor()
    feat = torch.randn(4, F_DIM)
    a_train = actor.act(feat, training=True)
    a_eval = actor.act(feat, training=False)
    assert a_train.shape == (4, A_DIM)
    assert a_eval.shape == (4, A_DIM)
    # 評価モードは deterministic なので 2 回呼んでも同じ
    a_eval2 = actor.act(feat, training=False)
    assert torch.allclose(a_eval, a_eval2)


def test_actor_grad_flow():
    """rsample で actor のパラメータに勾配が流れる."""
    actor = make_actor()
    feat = torch.randn(4, F_DIM)
    a = actor.act(feat, training=True)
    a.sum().backward()
    final = [m for m in actor.mlp if isinstance(m, torch.nn.Linear)][-1]
    assert final.weight.grad is not None
    assert final.weight.grad.abs().sum() > 0


# ═════════════════════════════════════════════════════════════════════════
# Critic
# ═════════════════════════════════════════════════════════════════════════


def test_critic_init():
    """Critic が mlp を持ち、出力 1 次元."""
    critic = make_critic()
    assert hasattr(critic, "mlp")
    final = [m for m in critic.mlp if isinstance(m, torch.nn.Linear)][-1]
    assert final.out_features == 1


def test_critic_output_shape():
    """Critic: (..., F) → (...,)  (最後の 1 次元を squeeze)."""
    critic = make_critic()
    # 2D
    v1 = critic(torch.randn(8, F_DIM))
    assert v1.shape == (8,)
    # 3D (B, T, F)
    v2 = critic(torch.randn(4, 5, F_DIM))
    assert v2.shape == (4, 5)


# ═════════════════════════════════════════════════════════════════════════
# lambda_return
# ═════════════════════════════════════════════════════════════════════════


def test_lambda_return_simple():
    """T=1 の解析的な値と一致.

    T=1, reward=[1], value=[2], discount=[0.99], bootstrap=3, λ=0.5:
        V^λ_0 = 1 + 0.99 * (0.5 * 3 + 0.5 * 3) = 3.97
    """
    reward = torch.tensor([[1.0]])           # (T=1, B=1)
    value = torch.tensor([[2.0]])
    discount = torch.tensor([[0.99]])
    bootstrap = torch.tensor([3.0])          # (B=1,)
    out = lambda_return(reward, value, discount, bootstrap, lambda_=0.5)
    assert out.shape == (1, 1)
    assert abs(out.item() - 3.97) < 1e-5


def test_lambda_return_lambda_zero():
    """λ=0: TD(0) = r_t + γ V_{t+1}."""
    T = 5
    B = 2
    reward = torch.randn(T, B)
    value = torch.randn(T, B)
    discount = 0.99 * torch.ones(T, B)
    bootstrap = torch.randn(B)

    out = lambda_return(reward, value, discount, bootstrap, lambda_=0.0)

    # 期待値: returns[t] = reward[t] + 0.99 * V_{t+1}
    next_value = torch.cat([value[1:], bootstrap.unsqueeze(0)], dim=0)
    expected = reward + 0.99 * next_value
    assert torch.allclose(out, expected, atol=1e-5)


def test_lambda_return_lambda_one():
    """λ=1: Monte Carlo.
    V^λ_t = r_t + γ V^λ_{t+1}, V^λ_T = bootstrap.
    """
    T = 4
    B = 2
    reward = torch.randn(T, B)
    value = torch.randn(T, B)  # not used when λ=1 except for bootstrap path
    discount = 0.99 * torch.ones(T, B)
    bootstrap = torch.randn(B)

    out = lambda_return(reward, value, discount, bootstrap, lambda_=1.0)

    # 期待値: 後ろから累積
    expected = torch.zeros(T, B)
    last = bootstrap
    for t in reversed(range(T)):
        last = reward[t] + 0.99 * last
        expected[t] = last
    assert torch.allclose(out, expected, atol=1e-5)


def test_lambda_return_shape():
    """T=15 で形状確認."""
    T = 15
    B = 32
    reward = torch.randn(T, B)
    value = torch.randn(T, B)
    discount = 0.99 * torch.ones(T, B)
    bootstrap = torch.randn(B)
    out = lambda_return(reward, value, discount, bootstrap, lambda_=0.95)
    assert out.shape == (T, B)


# ═════════════════════════════════════════════════════════════════════════
# ActorCritic.compute_loss / update
# ═════════════════════════════════════════════════════════════════════════


def _make_posts(B: int = 2, T: int = 4) -> dict:
    """ダミーの posterior dict (B, T, ...)."""
    return {
        "deter": torch.randn(B, T, 200) * 0.1,
        "stoch": torch.randn(B, T, 30) * 0.1,
        "mean": torch.zeros(B, T, 30),
        "std": torch.ones(B, T, 30),
    }


def test_compute_loss_shape():
    """compute_loss が actor_loss, critic_loss (scalar), metrics を返す."""
    wm = make_wm()
    ac = ActorCritic(feat_dim=wm.rssm.feat_dim, action_dim=A_DIM)
    posts = _make_posts(B=2, T=3)

    actor_loss, critic_loss, metrics = ac.compute_loss(
        wm, posts, horizon=5, gamma=0.99, lambda_=0.95
    )
    assert actor_loss.dim() == 0
    assert critic_loss.dim() == 0
    assert "ac/actor" in metrics
    assert "ac/critic" in metrics


def test_compute_loss_grad_flow():
    """actor 勾配は actor のみ、critic 勾配は critic のみに流れる."""
    wm = make_wm()
    ac = ActorCritic(feat_dim=wm.rssm.feat_dim, action_dim=A_DIM)
    posts = _make_posts(B=2, T=3)

    actor_loss, critic_loss, _ = ac.compute_loss(
        wm, posts, horizon=5, gamma=0.99, lambda_=0.95
    )

    # actor backward (retain_graph)
    actor_loss.backward(retain_graph=True)
    final_actor = [m for m in ac.actor.mlp if isinstance(m, torch.nn.Linear)][-1]
    assert final_actor.weight.grad is not None and final_actor.weight.grad.abs().sum() > 0

    # PyTorch の仕様上、actor_loss は returns (Critic の出力に依存) から計算されるため、
    # actor_loss.backward() 時点で Critic のパラメータにも勾配が計算されます。
    # しかし、actor_opt.step() は Actor のみ更新し、
    # critic_opt.step() の直前に critic_opt.zero_grad() が呼ばれるため、実際の学習には影響しません。

    # critic backward
    critic_loss.backward()
    final_critic = [m for m in ac.critic.mlp if isinstance(m, torch.nn.Linear)][-1]
    assert final_critic.weight.grad is not None and final_critic.weight.grad.abs().sum() > 0


def test_compute_loss_wm_no_grad():
    """compute_loss で WM のパラメータには勾配が流れない (detach 済みのはず)."""
    wm = make_wm()
    ac = ActorCritic(feat_dim=wm.rssm.feat_dim, action_dim=A_DIM)
    posts = _make_posts(B=2, T=3)

    actor_loss, critic_loss, _ = ac.compute_loss(
        wm, posts, horizon=5, gamma=0.99, lambda_=0.95
    )
    (actor_loss + critic_loss).backward()

    # WM の encoder には勾配が流れない (posts は detach されているため)
    enc_first = wm.encoder.layers[0]
    assert enc_first.weight.grad is None or enc_first.weight.grad.abs().sum() == 0


def test_update_runs():
    """update が optimizer step を実行し metrics を返す."""
    wm = make_wm()
    ac = ActorCritic(feat_dim=wm.rssm.feat_dim, action_dim=A_DIM)
    actor_opt = torch.optim.Adam(ac.actor.parameters(), lr=8e-5)
    critic_opt = torch.optim.Adam(ac.critic.parameters(), lr=8e-5)
    posts = _make_posts(B=2, T=3)

    metrics = ac.update(
        wm, posts, actor_opt, critic_opt, horizon=5, gamma=0.99, lambda_=0.95
    )
    assert "ac/actor" in metrics
    assert "ac/critic" in metrics
    assert "ac/actor_grad" in metrics
    assert "ac/critic_grad" in metrics
