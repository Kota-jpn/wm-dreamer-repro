"""Actor (policy) + Critic (value) + λ-return を使った想像内学習.

Phase 6 / File 1.

設計書: docs/design_notes.md §10.5 Critic, §10.6 Actor, §10.8 AC 損失
出典:   公式 https://github.com/danijar/dreamer/blob/master/models.py
        + dreamer.py::Dreamer._train (AC 損失計算)

─── 全体像 ──────────────────────────────────────────────────────────────
WM 学習の posterior (B, T, ...) を flatten して B*T 個の開始点にする.
そこから actor を policy_fn として WorldModel.imagine_rollout で H ステップ想像.
得られた imag_feat に critic を適用して value, reward_head で reward を取得.
λ-return を計算して:
    L_actor  = -E[V^λ]                   (rsample で actor に勾配)
    L_critic = 0.5 * MSE(V, sg(V^λ))     (returns は detach)
の 2 つの損失を別オプティマイザで step.

─── Actor (TanhNormal) ──────────────────────────────────────────────────
公式 (verified):
    raw = MLP(feat)             (B, 2*A)
    mean_raw, std_raw = split(raw)
    mean = 5 * tanh(mean_raw / 5)             ← (-5, 5) に soft-clip
    std  = softplus(std_raw + log(e^5 - 1)) + 1e-4    ← 初期 std ≈ 5

    base = Normal(mean, std)                  ← (-5, 5) 近辺の Gaussian
    dist = TransformedDistribution(base, TanhTransform())
                                              ← tanh で [-1, 1] に押し込む
    action = dist.rsample()                   ← reparam サンプル

なぜこの設計か:
    - mean に 5*tanh(./5) をかけることで gradient の暴走を防ぐ
    - softplus(std_raw + bias) + 1e-4 で初期 std=5 の **強い探索** を仕込む
    - 最後の tanh で連続行動を [-1, 1] に押し込む

─── Critic ──────────────────────────────────────────────────────────────
シンプル: MLP → scalar.
公式は Normal(value, 1) を返すが、log_prob 学習は MSE と等価.
実装は MSE を使う (design_notes §10.11 参照).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.distributions.transforms import TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution

from .utils import detach_state, lambda_return
from .world_model import WorldModel


# 初期 std ≈ 5 にするための bias 値: log(e^5 - 1) ≈ 4.99326
# softplus(0 + INIT_STD_BIAS) = log(1 + e^4.99326) = log(e^5) = 5
INIT_STD_BIAS = math.log(math.exp(5.0) - 1.0)


# ═════════════════════════════════════════════════════════════════════════
# Actor
# ═════════════════════════════════════════════════════════════════════════


class Actor(nn.Module):
    """TanhNormal policy. feat (B, F) → action distribution.

    Args:
        feat_dim: 入力次元 (= rssm.feat_dim = 230).
        action_dim: 出力 action の次元 (walker=6).
        num_units: 各層の幅 (公式 400).
        num_layers: 隠れ層の数 (公式 3).
    """

    def __init__(
        self,
        feat_dim: int,
        action_dim: int,
        num_units: int = 400,
        num_layers: int = 3,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.action_dim = action_dim

        layers = []
        in_dim = feat_dim
        
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, num_units))
            layers.append(nn.ELU())
            in_dim = num_units
        
        layers.append(nn.Linear(num_units, 2 * action_dim))
        self.mlp = nn.Sequential(*layers)

        # ═════════════════════════════════════════════════════════════════
        # 📝 WRITE ME [#6.1.1]: Actor.__init__
        # ─────────────────────────────────────────────────────────────────
        # 【目的】 num_layers 層の Dense(num_units, ELU) + 出力 Dense(2*A) を構築.
        #
        # 【やること】 RewardHead と同じパターンだが、出力は **2*action_dim**:
        #
        #   layers = []
        #   in_dim = feat_dim
        #   for _ in range(num_layers):
        #       layers += [nn.Linear(in_dim, num_units), nn.ELU()]
        #       in_dim = num_units
        #   layers.append(nn.Linear(num_units, 2 * action_dim))
        #   self.mlp = nn.Sequential(*layers)
        #
        # 【ヒント】
        #   - 出力 2*A は (mean, std_raw) を後で chunk(2, dim=-1) で分割するため.
        #   - 最終層は活性化なし (transform は forward で別途適用).
        #
        # 【完了条件】
        #   - tests/test_actor_critic.py::test_actor_init が通る
        # ═════════════════════════════════════════════════════════════════

    # ═════════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#6.1.2]: Actor.forward
    # ─────────────────────────────────────────────────────────────────────
    # 【目的】 feat を受けて TanhNormal 分布を返す (.rsample() で action を取れる).
    #
    # 【やること】
    #
    #   1. raw = self.mlp(feat)           # (..., 2*A)
    #   2. mean_raw, std_raw = raw.chunk(2, dim=-1)   # 各 (..., A)
    #
    #   3. mean に soft-clip:
    #        mean = 5.0 * torch.tanh(mean_raw / 5.0)
    #
    #   4. std に bias を加えて softplus:
    #        std = F.softplus(std_raw + INIT_STD_BIAS) + 1e-4
    #
    #   5. base = Independent(Normal(mean, std), 1)
    #        ← 最後 1 軸 (action_dim) を event 化. KL や log_prob が独立 Gaussian になる.
    #
    #   6. dist = TransformedDistribution(base, TanhTransform())
    #        ← tanh で squash. action は [-1, 1] に.
    #
    #   7. return dist
    #
    # 【ヒント】
    #   - INIT_STD_BIAS = log(e^5 - 1) はファイル冒頭で定義済み.
    #   - dist.rsample() で reparam サンプル. dist.sample() だと勾配が止まる.
    #   - tanh の squash は値域を [-1, 1] に. action_spec も [-1, 1] なのでぴったり.
    #
    # 【公式参考】 models.py::ActionDecoder
    #   x = self.get('hout', tfkl.Dense, 2 * size)(x)
    #   mean, std = tf.split(x, 2, -1)
    #   mean = 5 * tf.tanh(mean / 5)
    #   std = tf.nn.softplus(std + tf.math.log(tf.exp(5.) - 1)) + 1e-4
    #   dist = tfd.Normal(mean, std)
    #   dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
    #   dist = tfd.Independent(dist, 1)  ← 軸の event 化
    #
    # 【完了条件】
    #   - tests/test_actor_critic.py::test_actor_output_range が通る
    #   - tests/test_actor_critic.py::test_actor_initial_std が通る
    # ═════════════════════════════════════════════════════════════════════
    def forward(self, feat: torch.Tensor) -> TransformedDistribution:
        """feat → TanhNormal 分布. rsample() で [-1, 1] の行動が得られる."""
        raw = self.mlp(feat)
        mean_raw, std_raw = raw.chunk(2, dim=-1)

        mean = 5.0 * torch.tanh(mean_raw / 5.0)
        std = F.softplus(std_raw + INIT_STD_BIAS) + 1e-4

        base = Independent(Normal(mean, std), 1)
        dist = TransformedDistribution(base, TanhTransform())

        return dist
    # ─────────────────────────────────────────────────────────────────────
    # 補助: rsample で action を取る (Phase 5 imagine_rollout で policy_fn として使う)
    # ─────────────────────────────────────────────────────────────────────

    def act(self, feat: torch.Tensor, training: bool = True) -> torch.Tensor:
        """feat (B, F) → action (B, A).

        Args:
            training: True なら rsample, False なら deterministic (tanh(mean)).

        【勾配】 rsample は reparameterization なので feat に勾配が流れる.
                 Actor の勾配学習はこの経路を使う.
        """
        dist = self.forward(feat)
        if training:
            return dist.rsample()
        # 評価時は deterministic action (mean を tanh で squash)
        # base_dist の mean は (-5, 5) range なので tanh で [-1, 1] に
        return torch.tanh(dist.base_dist.mean)


# ═════════════════════════════════════════════════════════════════════════
# Critic
# ═════════════════════════════════════════════════════════════════════════


class Critic(nn.Module):
    """Value network. feat (B, F) → value (B,).

    アーキテクチャ: RewardHead と完全に同じ (出力 1 次元の MLP).
    """

    def __init__(
        self,
        feat_dim: int,
        num_units: int = 400,
        num_layers: int = 3,
    ):
        super().__init__()
        self.feat_dim = feat_dim

        # ═════════════════════════════════════════════════════════════════
        # 📝 WRITE ME [#6.2.1]: Critic.__init__
        # ─────────────────────────────────────────────────────────────────
        # 【目的】 RewardHead と同じパターン. Linear(F→400)→ELU を 3 段 + 出力 Linear(400→1).
        #
        # 【やること】
        #   layers = []
        #   in_dim = feat_dim
        #   for _ in range(num_layers):
        #       layers += [nn.Linear(in_dim, num_units), nn.ELU()]
        #       in_dim = num_units
        #   layers.append(nn.Linear(num_units, 1))
        #   self.mlp = nn.Sequential(*layers)
        #
        # 【ヒント】
        #   - RewardHead と全く同じ構造. コピペで OK.
        #   - 別クラスにしているのは、後で Distributional Critic に差し替えやすくするため.
        #
        # 【完了条件】
        #   - tests/test_actor_critic.py::test_critic_init が通る
        # ═════════════════════════════════════════════════════════════════
        layers = []
        in_dim = feat_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, num_units), nn.ELU()]
            in_dim = num_units
        layers.append(nn.Linear(num_units, 1))
        self.mlp = nn.Sequential(*layers)


    # ═════════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#6.2.2]: Critic.forward
    # ─────────────────────────────────────────────────────────────────────
    # 【目的】 feat (..., F) → value (...,)  を返す.
    #
    # 【やること】
    #   out = self.mlp(feat)        # (..., 1)
    #   return out.squeeze(-1)      # (...,)
    #
    # 【ヒント】
    #   - RewardHead.forward と完全に同じ.
    #   - 出力は Normal(mean=value, std=1) の mean とみなす (損失で MSE を使う = 等価).
    #
    # 【完了条件】
    #   - tests/test_actor_critic.py::test_critic_output_shape が通る
    # ═════════════════════════════════════════════════════════════════════
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat (..., F) → value (...,)."""
        out = self.mlp(feat)
        return out.squeeze(-1)

# ═════════════════════════════════════════════════════════════════════════
# ActorCritic
# ═════════════════════════════════════════════════════════════════════════


class ActorCritic(nn.Module):
    """Actor + Critic + 想像内学習ロジック.

    使い方 (Phase 7):
        ac = ActorCritic(feat_dim=wm.rssm.feat_dim, action_dim=A)
        actor_opt = Adam(ac.actor.parameters(), lr=8e-5)
        critic_opt = Adam(ac.critic.parameters(), lr=8e-5)

        # WM update で得た posts を起点に
        metrics = ac.update(
            wm=wm, posts=posts,
            actor_opt=actor_opt, critic_opt=critic_opt,
            ...
        )
    """

    def __init__(
        self,
        feat_dim: int,
        action_dim: int,
        num_units: int = 400,
        num_layers: int = 3,
    ):
        super().__init__()
        self.actor = Actor(feat_dim, action_dim, num_units, num_layers)
        self.critic = Critic(feat_dim, num_units, num_layers)



    # ═════════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#6.4.1]: ActorCritic.compute_loss
    # ─────────────────────────────────────────────────────────────────────
    # 【目的】 imagine rollout を回し、actor/critic loss を計算.
    #
    # 【やること】
    #
    #   1. 開始点を準備 (B, T, ...) → (B*T, ...) flatten + detach:
    #        init_state = wm.flatten_state_for_imagine(detach_state(posts))
    #
    #   2. imagine_rollout を回す (policy = actor.act):
    #        def policy_fn(feat):
    #            return self.actor.act(feat, training=True)
    #
    #        out = wm.imagine_rollout(init_state, policy_fn, horizon=horizon)
    #        imag_feat   = out['feat']        # (B*T, H, F)
    #        imag_reward = out['reward']      # (B*T, H)
    #
    #   3. critic value (時間軸 H に対して):
    #        imag_value = self.critic(imag_feat)   # (B*T, H)
    #
    #   4. λ-return: 末尾を bootstrap として H-1 ステップ分の return を計算.
    #
    #      時間を第 0 軸に: transpose(0, 1) で (H, B*T) に.
    #      reward_for_lambda = imag_reward[:, :-1].transpose(0, 1)   # (H-1, B*T)
    #      value_for_lambda  = imag_value[:,  :-1].transpose(0, 1)   # (H-1, B*T)
    #      bootstrap         = imag_value[:, -1]                      # (B*T,)
    #
    #      discount は γ で全埋め:
    #      discount_seq = gamma * torch.ones_like(reward_for_lambda)
    #
    #      returns = lambda_return(
    #          reward=reward_for_lambda,
    #          value=value_for_lambda,
    #          discount=discount_seq,
    #          bootstrap=bootstrap,
    #          lambda_=lambda_,
    #      )                                                # (H-1, B*T)
    #
    #   5. Actor 損失 (returns を最大化):
    #        actor_loss = -returns.mean()
    #      ★ ここで returns には actor.act の reparameterization 経由で勾配が流れる.
    #         WM の paramaters には流したくない (detach_state 済み) → wm に勾配なし.
    #
    #   6. Critic 損失 (returns を予測):
    #        value_pred = imag_value[:, :-1].transpose(0, 1)   # (H-1, B*T)
    #        critic_loss = 0.5 * F.mse_loss(value_pred, returns.detach())
    #      ★ returns を detach することが重要 (= stop_gradient).
    #         さもないと critic 学習で actor の勾配経路に逆流する.
    #
    #   7. metrics dict:
    #        metrics = {
    #            'ac/actor':   actor_loss.item(),
    #            'ac/critic':  critic_loss.item(),
    #            'ac/return':  returns.mean().item(),
    #            'ac/value':   imag_value.mean().item(),
    #            'ac/reward':  imag_reward.mean().item(),
    #        }
    #
    #   8. return actor_loss, critic_loss, metrics
    #
    # 【ヒント (重要)】
    #   - posts の detach は **必須**: WM を AC 学習で動かさない.
    #   - returns の detach は **必須**: critic 学習で actor 経路に逆流させない.
    #   - actor_loss と critic_loss は同じ imag_feat / imag_value を使うが、
    #     **別々の backward** を呼ぶ (= 2 つの optimizer で別々に step).
    #   - retain_graph=True が必要 (1 つ目 backward で graph が消えないように).
    #
    # 【公式参考】 dreamer.py::Dreamer._train
    #   imag_feat = self._imagine_ahead(post)
    #   reward = self._reward(imag_feat).mode()
    #   value = self._value(imag_feat).mode()
    #   returns = tools.lambda_return(reward[:-1], value[:-1], pcont[:-1],
    #       bootstrap=value[-1], lambda_=self._c.disclam, axis=0)
    #   actor_loss = -tf.reduce_mean(discount * returns)
    #   value_loss = -tf.reduce_mean(discount * value_pred.log_prob(returns))
    #
    # 【完了条件】
    #   - tests/test_actor_critic.py::test_compute_loss_shape が通る
    #   - tests/test_actor_critic.py::test_compute_loss_grad_flow が通る
    # ═════════════════════════════════════════════════════════════════════
    def compute_loss(
        self,
        wm: WorldModel,
        posts: dict,
        horizon: int = 15,
        gamma: float = 0.99,
        lambda_: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:

        init_state = wm.flatten_state_for_imagine(detach_state(posts))

        def policy_fn(feat: torch.Tensor) -> torch.Tensor:
            return self.actor.act(feat, training=True)

        out = wm.imagine_rollout(init_state, policy_fn, horizon=horizon)
        imag_feat   = out['feat']        # (B*T, H, F)
        imag_reward = out['reward']      # (B*T, H)

        imag_value = self.critic(imag_feat)   # (B*T, H)

        reward_for_lambda = imag_reward[:, :-1].transpose(0, 1)
        value_for_lambda = imag_value[:, :-1].transpose(0, 1)
        bootstrap = imag_value[:, -1]

        discount_seq = gamma * torch.ones_like(reward_for_lambda)

        returns = lambda_return(
            reward=reward_for_lambda,
            value=value_for_lambda,
            discount=discount_seq,
            bootstrap=bootstrap,
            lambda_=lambda_,
        )

        actor_loss = -returns.mean()

        # Critic は Actor の更新に関与しないため、特徴量を detach して推論する
        detached_feat = imag_feat[:, :-1].detach()
        value_pred = self.critic(detached_feat).transpose(0, 1)   # (H-1, B*T)
        critic_loss = 0.5 * F.mse_loss(value_pred, returns.detach())

        metrics = {
            'ac/actor':   actor_loss.item(),
            'ac/critic':  critic_loss.item(),
            'ac/return':  returns.mean().item(),
            'ac/value':   imag_value.mean().item(),
            'ac/reward':  imag_reward.mean().item(),
        }

        return actor_loss, critic_loss, metrics

        """想像 rollout から actor/critic loss を計算.

        Args:
            wm: 学習済み WorldModel (この呼び出しでは更新しない).
            posts: WM.update が返した posterior dict (B, T, ...).
            horizon: 想像 rollout の長さ (Phase 7 では 15).
            gamma: 割引率 (0.99).
            lambda_: TD(λ) の λ (0.95).

        Returns:
            actor_loss, critic_loss, metrics
        """
        
    # ─────────────────────────────────────────────────────────────────────
    # 補助: 1 step の AC 学習 (2 つの optimizer.step)
    # ─────────────────────────────────────────────────────────────────────

    def update(
        self,
        wm: WorldModel,
        posts: dict,
        actor_opt: torch.optim.Optimizer,
        critic_opt: torch.optim.Optimizer,
        horizon: int = 15,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        grad_clip: float = 100.0,
    ) -> dict:
        """compute_loss → backward → 2 optimizer step.

        actor_loss と critic_loss は同じ計算グラフを共有するので
        retain_graph=True で 2 回 backward する.
        """
        actor_loss, critic_loss, metrics = self.compute_loss(
            wm, posts, horizon=horizon, gamma=gamma, lambda_=lambda_
        )

        # Actor step (retain_graph で graph を残す)
        actor_opt.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_grad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), grad_clip)
        actor_opt.step()

        # Critic step
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip)
        critic_opt.step()

        metrics['ac/actor_grad'] = actor_grad.item()
        metrics['ac/critic_grad'] = critic_grad.item()
        return metrics
