"""共通ユーティリティ.

Phase 2 以降の全モジュールで使う小さな関数群.
"""

from __future__ import annotations

import torch
from einops import rearrange


# ─────────────────────────────────────────────────────────────────────
# 時間軸の畳み込み (CNN は 4D 入力しか受け付けないので B*T に flatten)
# ─────────────────────────────────────────────────────────────────────


def flatten_time(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """(B, T, ...) → ((B*T), ...) と時間長 T を返す.

    使い方:
        x4d, T = flatten_time(obs)  # obs (B,T,3,64,64) → x4d (B*T,3,64,64)
        y4d = encoder(x4d)
        y = unflatten_time(y4d, T)  # (B*T, F) → (B, T, F)
    """
    B, T = x.shape[:2]
    return rearrange(x, "b t ... -> (b t) ..."), T


def unflatten_time(x: torch.Tensor, T: int) -> torch.Tensor:
    """((B*T), ...) → (B, T, ...).

    与えられた T で逆変換する. B は自動で割り出す.
    """
    return rearrange(x, "(b t) ... -> b t ...", t=T)


# ─────────────────────────────────────────────────────────────────────
# RSSM の state (dict) を時間軸で stack するヘルパ
# ─────────────────────────────────────────────────────────────────────


def stack_states(states: list[dict]) -> dict:
    """T 個の state dict を時間方向に stack する.

    入力: [state_0, state_1, ..., state_{T-1}]
        各 state は {'deter': (B, H), 'stoch': (B, Z), 'mean': (B, Z), 'std': (B, Z)}
    出力: {'deter': (B, T, H), 'stoch': (B, T, Z), 'mean': (B, T, Z), 'std': (B, T, Z)}

    使い方 (RSSM.observe 等で):
        priors = []
        for t in range(T):
            prior = self.img_step(state, action[:, t])
            priors.append(prior)
        priors = stack_states(priors)
    """
    keys = states[0].keys()
    return {k: torch.stack([s[k] for s in states], dim=1) for k in keys}


def detach_state(state: dict) -> dict:
    """state dict の全 tensor を detach する (Phase 6 想像 rollout 開始時に使用)."""
    return {k: v.detach() for k, v in state.items()}


# ─────────────────────────────────────────────────────────────────────
# λ-return (Phase 6 Actor-Critic 学習で使用)
# ─────────────────────────────────────────────────────────────────────


# ═════════════════════════════════════════════════════════════════════════
# 📝 WRITE ME [#6.3.1]: lambda_return
# ─────────────────────────────────────────────────────────────────────────
# 【目的】 λ-return を計算する. Dreamer Actor-Critic 学習の核心.
#
# 【数学】
#   V^λ_t = r_t + γ_t · ((1 - λ) · V_{t+1} + λ · V^λ_{t+1})
#   V^λ_T = bootstrap                                            ← 末尾境界
#
#   λ=0 → TD(0) (1-step return)
#   λ=1 → Monte Carlo (∞-step return)
#   λ=0.95 → Dreamer v1 デフォルト (中間)
#
# 【やること】
#   1. T = reward.shape[0] (時間が第 0 軸)
#
#   2. next_value[t] = V_{t+1} を準備:
#        next_value[t] = value[t+1]   for t = 0..T-2
#        next_value[T-1] = bootstrap
#      実装:
#        next_value = torch.cat([value[1:], bootstrap.unsqueeze(0)], dim=0)
#
#   3. inputs_t = r_t + γ_t (1-λ) V_{t+1} を一括計算:
#        inputs = reward + discount * (1 - lambda_) * next_value
#
#   4. 後ろから前への再帰で V^λ_t = inputs_t + γ_t λ V^λ_{t+1}:
#        outputs = []
#        last = bootstrap                          # V^λ_T = bootstrap
#        for t in reversed(range(T)):
#            last = inputs[t] + discount[t] * lambda_ * last
#            outputs.insert(0, last)               # 先頭に挿入
#
#   5. torch.stack(outputs, dim=0) で (T, ...) を返す
#
# 【ヒント】
#   - 「後ろから」が重要. V^λ_t は V^λ_{t+1} に依存するので逆順.
#   - inputs を事前計算するのは効率化. 数式を直接書いても OK.
#   - bootstrap は (B,) や (...,) (時間軸を持たない). value[t] と同じ非時間形状.
#   - PyTorch の `torch.stack(list, dim=0)` で list の tensor を新しい次元 0 で連結.
#   - `outputs.insert(0, x)` は O(N) なので、T が小さいときは問題なし. 巨大なら
#     append → reverse の方が効率的だが Phase 6 では H=15 程度.
#
# 【数値検証 (簡単な例)】
#   T=1, reward=[1], value=[2], discount=[0.99], bootstrap=3, lambda=0.5:
#     next_value = [3]
#     inputs = [1 + 0.99 * 0.5 * 3] = [2.485]
#     last = bootstrap = 3
#     t=0: last = 2.485 + 0.99 * 0.5 * 3 = 2.485 + 1.485 = 3.97
#     return [3.97]
#
#   直接計算:
#     V^λ_0 = 1 + 0.99 * (0.5 * 3 + 0.5 * 3) = 1 + 0.99 * 3 = 3.97 ✓
#
# 【公式参考】 tools.py::lambda_return
#   inputs = reward + pcont * value * (1 - lambda_)
#   ↑ 公式は value (= V_t) を使う形. 代数的に等価だが視点が違うだけ.
#
# 【完了条件】
#   - tests/test_actor_critic.py::test_lambda_return_simple が通る
#   - tests/test_actor_critic.py::test_lambda_return_lambda_zero が通る
#   - tests/test_actor_critic.py::test_lambda_return_lambda_one が通る
# ═════════════════════════════════════════════════════════════════════════
def lambda_return(
    reward: torch.Tensor,
    value: torch.Tensor,
    discount: torch.Tensor,
    bootstrap: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    T = reward.shape[0]

    # next_value[t] = V_{t+1}
    next_value = torch.cat([value[1:], bootstrap.unsqueeze(0)], dim=0)

    # inputs_t = r_t + γ_t * (1 - λ) * V_{t+1}
    inputs = reward + discount * (1 - lambda_) * next_value

    # 後ろから前へ再帰: V^λ_t = inputs_t + γ_t * λ * V^λ_{t+1}
    returns = []
    last = bootstrap  # V^λ_T
    for t in reversed(range(T)):
        last = inputs[t] + discount[t] * lambda_ * last
        returns.insert(0, last)  # 先頭に挿入

    return torch.stack(returns, dim=0)
    """λ-return を計算する (時間が第 0 軸).

    Args:
        reward:    (T, ...) — 各ステップの報酬 r_0..r_{T-1}
        value:     (T, ...) — Critic 推定値 V_0..V_{T-1}
        discount:  (T, ...) — γ × pcont. DMC では γ=0.99 固定で OK
        bootstrap: (...,)   — V_T (= 末尾の Critic 推定で代用するのが普通)
        lambda_:   float    — TD(λ) の λ. Dreamer v1 は 0.95

    Returns:
        returns: (T, ...) — V^λ_0..V^λ_{T-1}
    """
