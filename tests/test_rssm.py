"""RSSM のテスト (Phase 3).

実行:
    uv run pytest tests/test_rssm.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F

from dreamer.rssm import RSSM


# ─────────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────────

# 軽量化のため小さめ設定
A_DIM = 6      # walker
E_DIM = 1024   # encoder 出力
H_DIM = 200    # deter
Z_DIM = 30     # stoch


def make_rssm() -> RSSM:
    return RSSM(
        action_dim=A_DIM,
        embed_dim=E_DIM,
        deter_dim=H_DIM,
        stoch_dim=Z_DIM,
        hidden_dim=200,
        min_std=0.1,
    )


# ─────────────────────────────────────────────────────────────────────
# __init__ と initial_state
# ─────────────────────────────────────────────────────────────────────


def test_init():
    """サブモジュールが揃っている."""
    rssm = make_rssm()
    # nn.GRUCell, nn.Linear が存在する
    assert hasattr(rssm, "gru")
    assert hasattr(rssm, "fc_input")
    assert hasattr(rssm, "fc_prior1")
    assert hasattr(rssm, "fc_prior2")
    assert hasattr(rssm, "fc_post1")
    assert hasattr(rssm, "fc_post2")
    # parameters が登録されている
    assert sum(p.numel() for p in rssm.parameters()) > 0


def test_initial_state():
    """zero-init された state dict を返す."""
    rssm = make_rssm()
    state = rssm.initial_state(batch_size=4, device=torch.device("cpu"))
    assert state["deter"].shape == (4, H_DIM)
    assert state["stoch"].shape == (4, Z_DIM)
    assert state["mean"].shape == (4, Z_DIM)
    assert state["std"].shape == (4, Z_DIM)
    assert torch.allclose(state["deter"], torch.zeros(4, H_DIM))
    assert torch.allclose(state["stoch"], torch.zeros(4, Z_DIM))


# ─────────────────────────────────────────────────────────────────────
# img_step
# ─────────────────────────────────────────────────────────────────────


def test_img_step_shape():
    """img_step の出力形状."""
    rssm = make_rssm()
    state = rssm.initial_state(batch_size=4, device=torch.device("cpu"))
    action = torch.randn(4, A_DIM)
    out = rssm.img_step(state, action)
    assert out["deter"].shape == (4, H_DIM)
    assert out["stoch"].shape == (4, Z_DIM)
    assert out["mean"].shape == (4, Z_DIM)
    assert out["std"].shape == (4, Z_DIM)


def test_img_step_std_min():
    """std は min_std (=0.1) 以上."""
    rssm = make_rssm()
    state = rssm.initial_state(batch_size=4, device=torch.device("cpu"))
    action = torch.randn(4, A_DIM)
    out = rssm.img_step(state, action)
    assert (out["std"] >= 0.1 - 1e-6).all()


def test_img_step_reparam_grad():
    """rsample で勾配が prior MLP に流れる."""
    rssm = make_rssm()
    state = rssm.initial_state(batch_size=4, device=torch.device("cpu"))
    action = torch.randn(4, A_DIM)
    out = rssm.img_step(state, action)
    # stoch の合計に対して backward
    out["stoch"].sum().backward()
    # fc_prior1 の重みに勾配が来ている
    assert rssm.fc_prior1.weight.grad is not None
    assert rssm.fc_prior1.weight.grad.abs().sum() > 0


# ─────────────────────────────────────────────────────────────────────
# obs_step
# ─────────────────────────────────────────────────────────────────────


def test_obs_step_shape():
    """obs_step は (prior, posterior) を返す."""
    rssm = make_rssm()
    state = rssm.initial_state(batch_size=4, device=torch.device("cpu"))
    action = torch.randn(4, A_DIM)
    embed = torch.randn(4, E_DIM)
    prior, post = rssm.obs_step(state, action, embed)

    for d in (prior, post):
        assert d["deter"].shape == (4, H_DIM)
        assert d["stoch"].shape == (4, Z_DIM)
        assert d["mean"].shape == (4, Z_DIM)
        assert d["std"].shape == (4, Z_DIM)

    # prior と posterior は deter を共有している
    assert torch.allclose(prior["deter"], post["deter"])


def test_obs_step_uses_embed():
    """embed を変えると posterior の mean/std が変わる (prior は不変)."""
    rssm = make_rssm()
    state = rssm.initial_state(batch_size=4, device=torch.device("cpu"))
    action = torch.randn(4, A_DIM)
    embed1 = torch.randn(4, E_DIM)
    embed2 = torch.randn(4, E_DIM)

    torch.manual_seed(0)
    prior1, post1 = rssm.obs_step(state, action, embed1)
    torch.manual_seed(0)
    prior2, post2 = rssm.obs_step(state, action, embed2)

    # prior は embed を使わないので一致
    assert torch.allclose(prior1["mean"], prior2["mean"])
    # posterior は embed を使うので異なる
    assert not torch.allclose(post1["mean"], post2["mean"])


# ─────────────────────────────────────────────────────────────────────
# observe (T ステップ unroll)
# ─────────────────────────────────────────────────────────────────────


def test_observe_shape():
    """T ステップ unroll. 出力は (B, T, ...)."""
    rssm = make_rssm()
    B, T = 4, 8
    state = rssm.initial_state(batch_size=B, device=torch.device("cpu"))
    embed = torch.randn(B, T, E_DIM)
    action = torch.randn(B, T, A_DIM)

    priors, posts = rssm.observe(embed, action, state)
    for d in (priors, posts):
        assert d["deter"].shape == (B, T, H_DIM)
        assert d["stoch"].shape == (B, T, Z_DIM)
        assert d["mean"].shape == (B, T, Z_DIM)
        assert d["std"].shape == (B, T, Z_DIM)


def test_observe_uses_posterior_for_next():
    """observe ループで state は post で更新される
    (prior で更新すると teacher forcing にならない).

    間接的なチェック: 同じ embed を T 回与えて、各 step の posterior が
    異なる embed を使ったときの結果と integrally 異なる.
    厳密性ではなく 'observe が走る + 形状が正しい' を主目的にする.
    """
    rssm = make_rssm()
    B, T = 2, 4
    state = rssm.initial_state(batch_size=B, device=torch.device("cpu"))
    embed = torch.randn(B, T, E_DIM)
    action = torch.zeros(B, T, A_DIM)

    priors, posts = rssm.observe(embed, action, state)
    # 連続するステップの deter が等しくない (= 状態が進行している)
    assert not torch.allclose(posts["deter"][:, 0], posts["deter"][:, 1])


# ─────────────────────────────────────────────────────────────────────
# imagine (T ステップ prior-only)
# ─────────────────────────────────────────────────────────────────────


def test_imagine_shape():
    """imagine の出力形状."""
    rssm = make_rssm()
    B, T = 4, 8
    state = rssm.initial_state(batch_size=B, device=torch.device("cpu"))
    action = torch.randn(B, T, A_DIM)

    priors = rssm.imagine(action, state)
    assert priors["deter"].shape == (B, T, H_DIM)
    assert priors["stoch"].shape == (B, T, Z_DIM)


def test_imagine_no_nan():
    """想像 rollout で NaN が出ない."""
    rssm = make_rssm()
    B, T = 4, 15  # Phase 5 と同じ horizon
    state = rssm.initial_state(batch_size=B, device=torch.device("cpu"))
    action = torch.randn(B, T, A_DIM) * 0.5
    priors = rssm.imagine(action, state)
    assert not torch.isnan(priors["stoch"]).any()
    assert not torch.isnan(priors["deter"]).any()


# ─────────────────────────────────────────────────────────────────────
# kl_loss
# ─────────────────────────────────────────────────────────────────────


def test_kl_scalar():
    """KL は scalar."""
    rssm = make_rssm()
    B, T = 4, 8
    state = rssm.initial_state(batch_size=B, device=torch.device("cpu"))
    embed = torch.randn(B, T, E_DIM)
    action = torch.randn(B, T, A_DIM)
    priors, posts = rssm.observe(embed, action, state)

    kl = rssm.kl_loss(posts, priors, free_nats=0.0)
    assert kl.dim() == 0  # scalar


def test_kl_free_nats_clamp():
    """free_nats=10 の時、KL がそれ未満なら 10 に clamp."""
    rssm = make_rssm()
    B, T = 4, 8
    state = rssm.initial_state(batch_size=B, device=torch.device("cpu"))
    embed = torch.randn(B, T, E_DIM)
    action = torch.randn(B, T, A_DIM)
    priors, posts = rssm.observe(embed, action, state)

    kl_no_clamp = rssm.kl_loss(posts, priors, free_nats=0.0)
    kl_with_clamp = rssm.kl_loss(posts, priors, free_nats=kl_no_clamp.item() + 5.0)
    # clamp が効いていれば +5.0 後の値になる
    assert kl_with_clamp.item() >= kl_no_clamp.item() + 5.0 - 1e-5


def test_kl_decreases():
    """簡易学習: ランダム embed で観測した posterior と prior の KL を
    minimize する Adam ステップで KL が減る.

    注: 最終的に KL=0 にはならない (embed がランダムだと post が広がる)
    """
    torch.manual_seed(0)
    rssm = make_rssm()
    opt = torch.optim.Adam(rssm.parameters(), lr=3e-4)

    B, T = 4, 5
    embed = torch.randn(B, T, E_DIM)
    action = torch.randn(B, T, A_DIM) * 0.3

    initial_kl = None
    final_kl = None
    for step in range(200):
        state = rssm.initial_state(batch_size=B, device=torch.device("cpu"))
        priors, posts = rssm.observe(embed, action, state)
        kl = rssm.kl_loss(posts, priors, free_nats=0.0)

        opt.zero_grad()
        kl.backward()
        opt.step()

        if step == 0:
            initial_kl = kl.item()
        final_kl = kl.item()

    print(f"initial KL: {initial_kl:.4f}, final KL: {final_kl:.4f}")
    assert final_kl < initial_kl, f"KL did not decrease: {initial_kl:.4f} → {final_kl:.4f}"


# ─────────────────────────────────────────────────────────────────────
# get_feat (Phase 4 でも使う)
# ─────────────────────────────────────────────────────────────────────


def test_get_feat():
    """feat = concat(deter, stoch) の形状."""
    rssm = make_rssm()
    state = rssm.initial_state(batch_size=4, device=torch.device("cpu"))
    feat = rssm.get_feat(state)
    assert feat.shape == (4, H_DIM + Z_DIM)
    assert rssm.feat_dim == H_DIM + Z_DIM
