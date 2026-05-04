"""ConvDecoder: latent feature → 64x64 RGB image.

Phase 2 / File 2.

設計書: docs/design_notes.md §10.2 ConvDecoder
出典:   公式 https://github.com/danijar/dreamer/blob/master/models.py
        (class ConvDecoder)

アーキテクチャ:
    入力 feat: (B, T, F)        F = 230 (= deter 200 + stoch 30)
                                  ※Phase 4 で feat = concat(h, s) になる
                                  ※Phase 2 単体では F は任意の値で OK
      ↓ flatten time                                   → (B*T, F)
      ↓ Linear(F, 32*depth=1024)                       → (B*T, 1024)
      ↓ Reshape → (B*T, 1024, 1, 1)
      ↓ ConvT2d(1024,    4*depth=128, k=5, s=2) + ReLU → (B*T, 128, 5, 5)
      ↓ ConvT2d(128,     2*depth=64,  k=5, s=2) + ReLU → (B*T, 64,  13, 13)
      ↓ ConvT2d(64,      1*depth=32,  k=6, s=2) + ReLU → (B*T, 32,  30, 30)
      ↓ ConvT2d(32,      3,           k=6, s=2)        → (B*T, 3,   64, 64)
      ↓ unflatten time                                 → (B, T, 3, 64, 64)

★ kernel size が **5, 5, 6, 6 と非対称** に注意:
    - 公式コードと厳密に一致させる必要がある
    - 4, 4, 4, 4 にすると 64x64 出力にならない
    - ConvT の出力サイズ: H_out = (H_in - 1) * s - 2*p + k
        1 → (1-1)*2 - 0 + 5 = 5
        5 → (5-1)*2 - 0 + 5 = 13
        13 → (13-1)*2 - 0 + 6 = 30
        30 → (30-1)*2 - 0 + 6 = 64 ✓

ポイント:
    - 最終 ConvT2d の後に活性化はない (生のピクセル値を出力)
    - 出力は Normal(mean=output, std=1) の mean とみなす → 学習は MSE/2 等価
    - 公式は depth=32 で固定. 1*depth=32, 2*depth=64, 4*depth=128
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .utils import flatten_time, unflatten_time


class ConvDecoder(nn.Module):
    """latent feature を 64x64 RGB image に復元する ConvT.

    Args:
        feat_dim: 入力特徴量の次元. Phase 4 で 230 (deter 200 + stoch 30).
        depth: ConvT のチャネル multiplier. 公式 cnn_depth=32.

    Attributes:
        feat_dim: 入力 feature 次元.
        depth: チャネル multiplier.
    """

    def __init__(self, feat_dim: int, depth: int = 32):
        super().__init__()
        self.feat_dim = feat_dim
        self.depth = depth
        self.embed_dim = 32 * depth  # 1024 with depth=32

        # ═════════════════════════════════════════════════════════════════
        # 📝 WRITE ME [#2.2.1]: ConvDecoder.__init__
        # ─────────────────────────────────────────────────────────────────
        # 【目的】 Linear 1 層 + ConvT2d 4 層を構築する.
        #
        # 【やること】
        #   1. self.fc = nn.Linear(feat_dim, self.embed_dim)
        #        → feat (F,) を 1024 次元に持ち上げる
        #
        #   2. self.deconv = nn.Sequential(
        #          nn.ConvTranspose2d(self.embed_dim, 4*depth,  kernel_size=5, stride=2),
        #          nn.ReLU(),
        #          nn.ConvTranspose2d(4*depth,        2*depth,  kernel_size=5, stride=2),
        #          nn.ReLU(),
        #          nn.ConvTranspose2d(2*depth,        1*depth,  kernel_size=6, stride=2),
        #          nn.ReLU(),
        #          nn.ConvTranspose2d(1*depth,        3,        kernel_size=6, stride=2),
        #          # ↑ 最後は活性化なし
        #      )
        #
        # 【ヒント】
        #   - ConvTranspose2d のデフォルト padding=0
        #   - ChannelLast の人は注意: PyTorch は Channel-First (B,C,H,W)
        #   - ★ kernel size の組み合わせ 5,5,6,6 を絶対変えない
        #
        # 【公式参考】
        #   models.py::ConvDecoder._cnn:
        #       x = self.get('h1', tfkl.Dense, 32*self._depth)(features)
        #       x = tf.reshape(x, [-1, 1, 1, 32*self._depth])
        #       x = self._activation(self.get('h2', tfkl.Conv2DTranspose, 4*depth, 5, strides=2)(x))
        #       x = self._activation(self.get('h3', tfkl.Conv2DTranspose, 2*depth, 5, strides=2)(x))
        #       x = self._activation(self.get('h4', tfkl.Conv2DTranspose, 1*depth, 6, strides=2)(x))
        #       x = self.get('h5', tfkl.Conv2DTranspose, self._shape[-1], 6, strides=2)(x)
        #
        # 【完了条件】
        #   - tests/test_encoder_decoder.py::test_decoder_shape が通る
        # ═════════════════════════════════════════════════════════════════
        self.fc = nn.Linear(feat_dim, self.embed_dim)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, 4*depth,  kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4*depth,        2*depth,  kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(2*depth,        1*depth,  kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(1*depth,        3,        kernel_size=6, stride=2),
        )
    # ═════════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#2.2.2]: ConvDecoder.forward
    # ─────────────────────────────────────────────────────────────────────
    # 【目的】 (B, T, F) → (B, T, 3, 64, 64) に復号.
    #
    # 【やること】
    #   1. flatten time: x, T = flatten_time(feat)
    #        → x: (B*T, F)
    #   2. Linear で次元を持ち上げる: x = self.fc(x)
    #        → (B*T, 1024)
    #   3. (B*T, 1024, 1, 1) に reshape:
    #        x = x.view(-1, self.embed_dim, 1, 1)
    #        または x = rearrange(x, "n d -> n d 1 1")
    #   4. self.deconv(x) で復号
    #        → (B*T, 3, 64, 64)
    #   5. unflatten_time で時間軸を戻す
    #        → (B, T, 3, 64, 64)
    #
    # 【ヒント】
    #   - 最終 ConvT に活性化を**加えない** (生のピクセル値が出力)
    #   - Sigmoid や tanh も**かけない**. 値域は self-consistent なら OK
    #   - 出力は学習で [-0.5, 0.5] 付近に収束していく
    #
    # 【形状チェック】
    #   B=2, T=3, F=230 で:
    #     入力 feat: (2, 3, 230)
    #     fc 後:    (6, 1024)
    #     reshape:  (6, 1024, 1, 1)
    #     deconv:   (6, 3, 64, 64)
    #     unflatten:(2, 3, 3, 64, 64)
    #
    # 【完了条件】
    #   - tests/test_encoder_decoder.py::test_decoder_shape が通る
    # ═════════════════════════════════════════════════════════════════════
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """latent feature から画像を復元する.

        Args:
            feat: (B, T, feat_dim) float32.

        Returns:
            recon: (B, T, 3, 64, 64) float32.
                   Normal(mean=recon, std=1) の mean としても解釈する.
        """
        x, T = flatten_time(feat)
        x = self.fc(x)
        x = x.view(-1, self.embed_dim, 1, 1)
        x = self.deconv(x)
        recon = unflatten_time(x, T)
        return recon