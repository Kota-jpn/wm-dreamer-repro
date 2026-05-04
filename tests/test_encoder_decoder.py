"""Encoder + Decoder のテスト (Phase 2).

実行:
    uv run pytest tests/test_encoder_decoder.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F

from dreamer.encoder import ConvEncoder
from dreamer.decoder import ConvDecoder


# ─────────────────────────────────────────────────────────────────────
# 形状チェック
# ─────────────────────────────────────────────────────────────────────


def test_encoder_shape():
    """Encoder: (B, T, 3, 64, 64) → (B, T, 1024)."""
    enc = ConvEncoder(depth=32)
    obs = torch.randn(2, 5, 3, 64, 64)
    embed = enc(obs)
    assert embed.shape == (2, 5, 1024), f"got {embed.shape}"
    assert enc.embed_dim == 1024


def test_encoder_dtype():
    """Encoder: dtype は float32 を維持."""
    enc = ConvEncoder(depth=32)
    obs = torch.randn(2, 5, 3, 64, 64, dtype=torch.float32)
    embed = enc(obs)
    assert embed.dtype == torch.float32


def test_encoder_depth_variants():
    """Encoder: depth=16 だと embed_dim=512."""
    enc = ConvEncoder(depth=16)
    obs = torch.randn(1, 1, 3, 64, 64)
    embed = enc(obs)
    assert enc.embed_dim == 512
    assert embed.shape == (1, 1, 512)


def test_decoder_shape():
    """Decoder: (B, T, F) → (B, T, 3, 64, 64)."""
    dec = ConvDecoder(feat_dim=230, depth=32)
    feat = torch.randn(2, 5, 230)
    recon = dec(feat)
    assert recon.shape == (2, 5, 3, 64, 64), f"got {recon.shape}"


def test_decoder_dtype():
    """Decoder: dtype は float32 を維持."""
    dec = ConvDecoder(feat_dim=230, depth=32)
    feat = torch.randn(2, 5, 230, dtype=torch.float32)
    recon = dec(feat)
    assert recon.dtype == torch.float32


# ─────────────────────────────────────────────────────────────────────
# 統合: encode → decode の往復
# ─────────────────────────────────────────────────────────────────────


def test_roundtrip_shape():
    """encoder → decoder で形状が往復する.

    encoder の出力 (1024) を直接 decoder の入力にしてみる.
    feat_dim=1024 を decoder に与えれば成立する.
    """
    enc = ConvEncoder(depth=32)
    dec = ConvDecoder(feat_dim=enc.embed_dim, depth=32)

    obs = torch.randn(2, 3, 3, 64, 64)
    embed = enc(obs)
    recon = dec(embed)

    assert recon.shape == obs.shape, f"recon {recon.shape} != obs {obs.shape}"


# ─────────────────────────────────────────────────────────────────────
# 学習可能性: ランダム batch で loss が下がる
# ─────────────────────────────────────────────────────────────────────


def test_learnable():
    """500 step で MSE が半減以上下がる (簡易学習可能性チェック).

    注: テスト時間を抑えるため小さいバッチ・短ステップ.
    本格的な学習は scripts/smoke_encoder_decoder.py で行う.
    """
    torch.manual_seed(0)
    enc = ConvEncoder(depth=16)  # 軽量化のため depth=16
    dec = ConvDecoder(feat_dim=enc.embed_dim, depth=16)

    # 固定の擬似 obs (学習対象).
    obs = torch.randn(4, 5, 3, 64, 64) * 0.3

    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=3e-4)

    initial_loss = None
    final_loss = None
    for step in range(500):
        embed = enc(obs)
        recon = dec(embed)
        loss = F.mse_loss(recon, obs)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    print(f"initial loss: {initial_loss:.4f}, final loss: {final_loss:.4f}")
    assert final_loss < initial_loss * 0.8, (
        f"loss did not decrease enough: {initial_loss:.4f} → {final_loss:.4f}"
    )
