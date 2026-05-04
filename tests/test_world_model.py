"""WorldModel のテスト (Phase 4).

実行:
    uv run pytest tests/test_world_model.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from dreamer.world_model import WorldModel, RewardHead


# ─────────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────────

A_DIM = 6  # walker


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


def make_batch(B: int = 4, T: int = 8, device: str = "cpu") -> dict:
    """ReplayBuffer.sample が返すような batch (torch tensor 化済み)."""
    return {
        "obs": torch.randn(B, T + 1, 3, 64, 64, device=device) * 0.3,
        "action": torch.randn(B, T, A_DIM, device=device) * 0.3,
        "reward": torch.randn(B, T, device=device),
        "discount": torch.ones(B, T, device=device),
    }


# ─────────────────────────────────────────────────────────────────────
# RewardHead
# ─────────────────────────────────────────────────────────────────────


def test_reward_head_shape():
    """RewardHead: (B, T, F) → (B, T)."""
    head = RewardHead(feat_dim=230, num_units=400, num_layers=3)
    feat = torch.randn(4, 8, 230)
    out = head(feat)
    assert out.shape == (4, 8), f"got {out.shape}"


def test_reward_head_param_count():
    """RewardHead は num_layers + 1 個の Linear を持つ."""
    head = RewardHead(feat_dim=230, num_units=400, num_layers=3)
    linears = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 4, f"expected 4 Linear (3 hidden + 1 out), got {len(linears)}"


# ─────────────────────────────────────────────────────────────────────
# WorldModel: __init__
# ─────────────────────────────────────────────────────────────────────


def test_init():
    """WorldModel が encoder, rssm, decoder, reward_head を持つ."""
    wm = make_wm()
    assert hasattr(wm, "encoder")
    assert hasattr(wm, "rssm")
    assert hasattr(wm, "decoder")
    assert hasattr(wm, "reward_head")
    # parameters が登録されている
    n_params = sum(p.numel() for p in wm.parameters())
    assert n_params > 1_000_000, f"too few params: {n_params}"


# ─────────────────────────────────────────────────────────────────────
# observe
# ─────────────────────────────────────────────────────────────────────


def test_observe_shape():
    """observe の出力形状."""
    wm = make_wm()
    B, T = 2, 5
    obs = torch.randn(B, T, 3, 64, 64)
    action = torch.randn(B, T, A_DIM)
    priors, posts, embed = wm.observe(obs, action)

    assert embed.shape == (B, T, 1024)
    for d in (priors, posts):
        assert d["deter"].shape == (B, T, 200)
        assert d["stoch"].shape == (B, T, 30)


# ─────────────────────────────────────────────────────────────────────
# compute_loss
# ─────────────────────────────────────────────────────────────────────


def test_compute_loss_shape():
    """compute_loss が scalar + dict + posts を返す."""
    wm = make_wm()
    batch = make_batch(B=2, T=5)
    total, metrics, posts = wm.compute_loss(batch, free_nats=3.0, kl_scale=1.0)

    assert total.dim() == 0  # scalar
    assert "wm/total" in metrics
    assert "wm/recon" in metrics
    assert "wm/reward" in metrics
    assert "wm/kl" in metrics
    assert posts["deter"].shape == (2, 5, 200)


def test_compute_loss_finite():
    """損失が有限値."""
    wm = make_wm()
    batch = make_batch(B=2, T=5)
    total, metrics, _ = wm.compute_loss(batch)
    assert torch.isfinite(total)
    for k, v in metrics.items():
        assert np.isfinite(v), f"{k} = {v} is not finite"


def test_compute_loss_grad_flows():
    """backward で全パラメータに勾配が流れる."""
    wm = make_wm()
    batch = make_batch(B=2, T=5)
    total, _, _ = wm.compute_loss(batch, free_nats=0.0)
    total.backward()
    no_grad = []
    for name, p in wm.named_parameters():
        if p.grad is None or p.grad.abs().sum() == 0:
            no_grad.append(name)
    # 勾配が 0 のパラメータは存在してはいけない
    assert len(no_grad) == 0, f"params without grad: {no_grad[:5]} ... ({len(no_grad)} total)"


def test_compute_loss_decreases():
    """簡易学習: 固定 batch で 200 step 学習し loss が下がる."""
    torch.manual_seed(0)
    wm = make_wm()
    opt = torch.optim.Adam(wm.parameters(), lr=6e-4)

    batch = make_batch(B=4, T=8)

    initial = None
    final = None
    for step in range(200):
        total, _, _ = wm.compute_loss(batch, free_nats=0.0, kl_scale=1.0)
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
        opt.step()

        if step == 0:
            initial = total.item()
        final = total.item()

    print(f"initial loss: {initial:.4f}, final loss: {final:.4f}")
    assert final < initial, f"loss did not decrease: {initial:.4f} → {final:.4f}"


# ─────────────────────────────────────────────────────────────────────
# update
# ─────────────────────────────────────────────────────────────────────


def test_update_step_runs():
    """update が optimizer step を実行し、metrics dict と posts を返す."""
    wm = make_wm()
    opt = torch.optim.Adam(wm.parameters(), lr=6e-4)
    batch = make_batch(B=2, T=5)

    metrics, posts = wm.update(batch, opt, free_nats=3.0, kl_scale=1.0, grad_clip=100.0)
    assert "wm/total" in metrics
    assert posts["deter"].shape == (2, 5, 200)


def test_update_changes_params():
    """update を 1 step 走らせるとパラメータが変化."""
    torch.manual_seed(0)
    wm = make_wm()
    opt = torch.optim.Adam(wm.parameters(), lr=6e-4)
    batch = make_batch(B=2, T=5)

    before = wm.encoder.layers[0].weight.detach().clone()
    wm.update(batch, opt, free_nats=0.0, kl_scale=1.0, grad_clip=100.0)
    after = wm.encoder.layers[0].weight.detach().clone()
    assert not torch.allclose(before, after)
