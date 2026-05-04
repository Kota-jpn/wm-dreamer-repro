"""ConvEncoder: 64x64 RGB image → 1024-dim embedding.

Phase 2 / File 1.

設計書: docs/design_notes.md §10.1 ConvEncoder
出典:   公式 https://github.com/danijar/dreamer/blob/master/models.py
        (class ConvEncoder)

アーキテクチャ:
    入力 obs: (B, T, 3, 64, 64) float32 in [-0.5, 0.5]
      ↓ flatten time              → (B*T, 3, 64, 64)
      ↓ Conv2d(3,   32,  k=4, s=2, padding=0) + ReLU   → (B*T, 32, 31, 31)
      ↓ Conv2d(32,  64,  k=4, s=2, padding=0) + ReLU   → (B*T, 64, 14, 14)
      ↓ Conv2d(64,  128, k=4, s=2, padding=0) + ReLU   → (B*T, 128, 6, 6)
      ↓ Conv2d(128, 256, k=4, s=2, padding=0) + ReLU   → (B*T, 256, 2, 2)
      ↓ Flatten                                         → (B*T, 1024)
      ↓ unflatten time                                  → (B, T, 1024)

ポイント:
    - padding は **valid (=0)**. 'same' を使わない
    - channels = [1, 2, 4, 8] * depth (depth=32)
    - 最終出力 = 2*2*256 = 1024 = 32 * depth
    - 活性化は **ReLU** (RSSM の Dense は ELU だが、ここは ReLU)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .utils import flatten_time, unflatten_time


class ConvEncoder(nn.Module):
    """画像 obs を embedding に圧縮する CNN.

    Args:
        depth: チャネル multiplier. 公式 cnn_depth=32.
               最終 embedding 次元 = 32 * depth = 1024.

    Attributes:
        embed_dim: 出力 embedding 次元 = 32 * depth.
    """

    def __init__(self, depth: int = 32):
        super().__init__()
        self.depth = depth
        self.embed_dim = 32 * depth  # 1024 with depth=32

        self.layers = nn.Sequential(
            nn.Conv2d(3,         1*depth, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(1*depth,   2*depth, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(2*depth,   4*depth, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(4*depth,   8*depth, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
        )

        # ═════════════════════════════════════════════════════════════════
        # 📝 WRITE ME [#2.1.1]: ConvEncoder.__init__ の self.layers
        # ─────────────────────────────────────────────────────────────────
        # 【目的】 4 つの Conv2d + ReLU を nn.Sequential として組み立てる.
        #
        # 【やること】
        #   self.layers = nn.Sequential(
        #       nn.Conv2d(3,         1*depth, kernel_size=4, stride=2, padding=0),
        #       nn.ReLU(),
        #       nn.Conv2d(1*depth,   2*depth, kernel_size=4, stride=2, padding=0),
        #       nn.ReLU(),
        #       nn.Conv2d(2*depth,   4*depth, kernel_size=4, stride=2, padding=0),
        #       nn.ReLU(),
        #       nn.Conv2d(4*depth,   8*depth, kernel_size=4, stride=2, padding=0),
        #       nn.ReLU(),
        #   )
        #   ↑ ヒントそのまま書き写してOK. 暗記より構造の理解が目的.
        #
        # 【ヒント】
        #   - padding は 0 (デフォルト). 'same' にしない
        #   - 最後の Flatten は forward() 側で einops で行う
        #   - ReLU は inplace=True にしても良い (微小な省メモリ)
        #
        # 【公式参考】
        #   models.py::ConvEncoder._cnn:
        #       x = self._activation(self.get('h1', tfkl.Conv2D, 1*self._depth, 4, 2)(x))
        #       x = self._activation(self.get('h2', tfkl.Conv2D, 2*self._depth, 4, 2)(x))
        #       ...
        #
        # 【完了条件】
        #   - tests/test_encoder_decoder.py::test_encoder_shape が通る
        # ═════════════════════════════════════════════════════════════════

    # ═════════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#2.1.2]: ConvEncoder.forward
    # ─────────────────────────────────────────────────────────────────────
    # 【目的】 (B, T, 3, 64, 64) → (B, T, 1024) に変換.
    #
    # 【やること】
    #   1. obs を時間軸込みで flatten: x, T = flatten_time(obs)
    #        → x: (B*T, 3, 64, 64)
    #   2. self.layers(x) で CNN を通す
    #        → (B*T, 256, 2, 2)
    #   3. (B*T, 1024) に reshape: x.flatten(start_dim=1)
    #        → (B*T, 1024)
    #   4. unflatten_time(x, T) で時間軸を復元
    #        → (B, T, 1024)
    #
    # 【ヒント】
    #   - flatten/unflatten は src/dreamer/utils.py で提供 (einops ラッパー)
    #   - x.flatten(start_dim=1) で channel*H*W を 1 軸に畳める
    #   - assert を入れて中間形状を確認すると初回の自信になる
    #     例: assert x.shape[-3:] == (256, 2, 2)
    #
    # 【形状チェック】
    #   B=2, T=3 で:
    #     入力: (2, 3, 3, 64, 64)
    #     conv 後: (6, 256, 2, 2)
    #     flatten: (6, 1024)
    #     unflatten: (2, 3, 1024)
    #
    # 【完了条件】
    #   - tests/test_encoder_decoder.py::test_encoder_shape が通る
    #   - tests/test_encoder_decoder.py::test_encoder_dtype が通る
    # ═════════════════════════════════════════════════════════════════════
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """画像 obs から embedding を計算する.

        Args:
            obs: (B, T, 3, 64, 64) float32 in [-0.5, 0.5].

        Returns:
            embed: (B, T, embed_dim=1024) float32.
        """

        x, T = flatten_time(obs)
        x = self.layers(x)
        x = x.flatten(start_dim=1)
        x = unflatten_time(x, T)
        return x
        
