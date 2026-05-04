"""RSSM: Recurrent State-Space Model (Dreamer v1 の心臓部).

Phase 3 / File 1.

設計書: docs/design_notes.md §10.3 RSSM
出典:   公式 https://github.com/danijar/dreamer/blob/master/models.py
        (class RSSM)
派生:   PlaNet (Hafner 2018) の RSSM がオリジナル.

─── 概念 ───────────────────────────────────────────────────────────────
状態:
    h_t (deter, decision-deterministic, GRU hidden, dim=200)
    s_t (stoch, stochastic, Gaussian latent, dim=30)

確率モデル:
    p(s_t | h_t)            ← prior     (観測なしの予測)
    q(s_t | h_t, e_t)       ← posterior (観測 e_t を取り込んだ事後)
    h_t = f(h_{t-1}, s_{t-1}, a_{t-1})  ← deterministic GRU

学習: ELBO
    L_wm = -E_q[log p(x_t|h_t,s_t)]
           -E_q[log p(r_t|h_t,s_t)]
           +β·KL(q(s_t|...) || p(s_t|...))

─── 1 ステップ ──────────────────────────────────────────────────────────
img_step (prior path, 観測なし):
    入力: prev_state {h_{t-1}, s_{t-1}}, prev_action a_{t-1}
    1. x = concat(s_{t-1}, a_{t-1})
    2. x = Dense(200, ELU)(x)
    3. h_t = GRUCell(x, h_{t-1})
    4. p = Dense(200, ELU)(h_t)
    5. p = Dense(2*Z=60)(p)
    6. prior_mean, prior_std_raw = split(p)
    7. prior_std = softplus(prior_std_raw) + 0.1
    8. s_t ~ Normal(prior_mean, prior_std)        ← prior サンプル
    出力: prior {deter=h_t, stoch=s_t, mean, std}

obs_step (posterior path, 観測 e_t を取り込む):
    1. prior = img_step(prev_state, prev_action)  ← まず prior 計算で h_t を得る
    2. q = concat(prior.deter, e_t)
    3. q = Dense(200, ELU)(q)
    4. q = Dense(2*Z=60)(q)
    5. post_mean, post_std_raw = split(q)
    6. post_std = softplus(post_std_raw) + 0.1
    7. s_t ~ Normal(post_mean, post_std)          ← posterior サンプル
    出力: prior, posterior {deter=h_t, stoch=s_t, mean, std}

─── State 表現 ──────────────────────────────────────────────────────────
状態は dict で表現 (公式と同じ):
    state = {
        'deter': (B, H=200),     # h_t
        'stoch': (B, Z=30),      # s_t (sample)
        'mean':  (B, Z=30),      # prior or posterior の mean
        'std':   (B, Z=30),      # prior or posterior の std
    }

T-step 時:
    {'deter': (B, T, H), 'stoch': (B, T, Z), 'mean': (B, T, Z), 'std': (B, T, Z)}
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl_divergence

from .utils import stack_states


class RSSM(nn.Module):
    """Recurrent State-Space Model.

    Args:
        action_dim:   A. 連続行動の次元.
        embed_dim:    E. ConvEncoder の出力次元 (1024).
        deter_dim:    H. GRU の hidden 次元 (200).
        stoch_dim:    Z. 確率的潜在の次元 (30).
        hidden_dim:   RSSM 内部 Dense の幅 (200, 公式の hidden_size).
        min_std:      σ の下限 (0.1).
        act:          活性化関数. デフォルト nn.ELU.
    """

    def __init__(
        self,
        action_dim: int,
        embed_dim: int = 1024,
        deter_dim: int = 200,
        stoch_dim: int = 30,
        hidden_dim: int = 200,
        min_std: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.hidden_dim = hidden_dim
        self.min_std = min_std

        # ═════════════════════════════════════════════════════════════════
        # 📝 WRITE ME [#3.1.1]: RSSM.__init__ のサブモジュール構築
        # ─────────────────────────────────────────────────────────────────
        # 【目的】 GRU + 4 つの Linear 層を構築する.
        #
        # 【やること】
        #
        # 1. GRU 入力を作る Linear:
        #    self.fc_input = nn.Linear(stoch_dim + action_dim, hidden_dim)
        #
        # 2. GRUCell:
        #    self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=deter_dim)
        #
        # 3. Prior MLP (1 層 + 出力層):
        #    self.fc_prior1 = nn.Linear(deter_dim, hidden_dim)
        #    self.fc_prior2 = nn.Linear(hidden_dim, 2 * stoch_dim)
        #    ↑ 出力は 2*Z (mean と std_raw を split で分ける)
        #
        # 4. Posterior MLP (1 層 + 出力層):
        #    self.fc_post1 = nn.Linear(deter_dim + embed_dim, hidden_dim)
        #    self.fc_post2 = nn.Linear(hidden_dim, 2 * stoch_dim)
        #
        # 【ヒント】
        #   - 公式は KerasModel + tfk.GRUCell 構成だが、PyTorch では nn.GRUCell が
        #     全く同じ役割.
        #   - 活性化 (ELU) は forward 内で F.elu(...) で適用. Linear 自体には付けない.
        #   - 「fc_input → GRU」と「fc_prior1 → fc_prior2」の 2 ルートがあることに注意.
        #
        # 【公式参考】 models.py::RSSM
        #   self.get('fc1', tfkl.Dense, hidden_size, self._activation) ← fc_input 相当
        #   self.get('cell', tfkl.GRUCell, deter_size)                  ← gru
        #   self.get('img1', tfkl.Dense, hidden_size, self._activation) ← fc_prior1
        #   self.get('img2', tfkl.Dense, 2 * stoch_size)                ← fc_prior2
        #   self.get('obs1', tfkl.Dense, hidden_size, self._activation) ← fc_post1
        #   self.get('obs2', tfkl.Dense, 2 * stoch_size)                ← fc_post2
        #
        # 【完了条件】
        #   - tests/test_rssm.py::test_init が通る
        # ═════════════════════════════════════════════════════════════════
        self.fc_input = nn.Linear(stoch_dim + action_dim, hidden_dim)
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=deter_dim)

        self.fc_prior1 = nn.Linear(deter_dim, hidden_dim)
        self.fc_prior2 = nn.Linear(hidden_dim, 2 * stoch_dim)

        self.fc_post1 = nn.Linear(deter_dim + embed_dim, hidden_dim)
        self.fc_post2 = nn.Linear(hidden_dim, 2 * stoch_dim)


        


    # ═════════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#3.1.2]: initial_state
    # ─────────────────────────────────────────────────────────────────────
    # 【目的】 zero-init された state dict を返す.
    #
    # 【やること】
    #   batch_size と device を引数に取って:
    #       return {
    #           'deter': torch.zeros(batch_size, self.deter_dim, device=device),
    #           'stoch': torch.zeros(batch_size, self.stoch_dim, device=device),
    #           'mean':  torch.zeros(batch_size, self.stoch_dim, device=device),
    #           'std':   torch.ones(batch_size,  self.stoch_dim, device=device),
    #       }
    #   ↑ std は 1.0 で初期化する (mean=0, std=1 = 標準正規)
    #
    # 【ヒント】
    #   - Phase 4 で 1 batch の最初に呼ぶ. 系列学習なので毎エピソード初期化ではない.
    #   - device は引数で受け取る (model がどの device にあるか分からないため).
    #
    # 【完了条件】
    #   - tests/test_rssm.py::test_initial_state が通る
    # ═════════════════════════════════════════════════════════════════════
    def initial_state(self, batch_size: int, device: torch.device) -> dict:
        return {
            'deter': torch.zeros(batch_size, self.deter_dim, device=device),
            'stoch': torch.zeros(batch_size, self.stoch_dim, device=device),
            'mean': torch.zeros(batch_size, self.stoch_dim, device=device),
            'std': torch.ones(batch_size, self.stoch_dim, device=device)
        }
    # ═════════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#3.1.3]: img_step (prior 1 ステップ)
    # ─────────────────────────────────────────────────────────────────────
    # 【目的】 観測なしで 1 ステップ進める. prior 分布を計算してサンプル.
    #
    # 【やること】 (Module docstring §「1 ステップ」img_step を参照)
    #
    #   1. x = torch.cat([prev_state['stoch'], prev_action], dim=-1)
    #        → (B, Z + A)
    #   2. x = F.elu(self.fc_input(x))
    #        → (B, hidden_dim)
    #   3. deter = self.gru(x, prev_state['deter'])
    #        → (B, H)   nn.GRUCell の戻り値は new hidden
    #
    #   4. p = F.elu(self.fc_prior1(deter))
    #        → (B, hidden_dim)
    #   5. p = self.fc_prior2(p)
    #        → (B, 2*Z)
    #   6. mean, std_raw = p.chunk(2, dim=-1)
    #        → 各 (B, Z)
    #   7. std = F.softplus(std_raw) + self.min_std
    #
    #   8. dist = Independent(Normal(mean, std), 1)   ← 最後 1 軸を「event」化
    #   9. stoch = dist.rsample()                     ← reparam サンプル ★
    #
    #   10. return {
    #           'deter': deter,
    #           'stoch': stoch,
    #           'mean':  mean,
    #           'std':   std,
    #       }
    #
    # 【ヒント】
    #   - .rsample() は reparameterization. .sample() だと勾配が流れない.
    #   - Independent で diag Gaussian にする → KL 計算が綺麗になる.
    #   - chunk(2, dim=-1) は 2 つに等分割.
    #
    # 【完了条件】
    #   - tests/test_rssm.py::test_img_step_shape が通る
    #   - tests/test_rssm.py::test_img_step_reparam_grad が通る
    # ═════════════════════════════════════════════════════════════════════
    def img_step(self, prev_state: dict, prev_action: torch.Tensor) -> dict:
        x = torch.cat([prev_state['stoch'], prev_action], dim=-1)
        x = F.elu(self.fc_input(x))
        deter = self.gru(x, prev_state['deter'])
        p = F.elu(self.fc_prior1(deter))
        p = self.fc_prior2(p)
        mean, std_raw = p.chunk(2, dim=-1)
        std = F.softplus(std_raw) + self.min_std

        dist = Independent(Normal(mean, std), 1)
        stoch = dist.rsample()

        return {
            'deter': deter,
            'stoch': stoch,
            'mean': mean,
            'std': std,
        }

    # ═════════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#3.1.4]: obs_step (posterior 1 ステップ)
    # ─────────────────────────────────────────────────────────────────────
    # 【目的】 観測 embed を取り込んだ posterior を計算してサンプル.
    #
    # 【やること】
    #
    #   1. prior = self.img_step(prev_state, prev_action)
    #        ← まず prior を計算. prior['deter'] が h_t になる.
    #
    #   2. q = torch.cat([prior['deter'], embed], dim=-1)
    #        → (B, H + E)
    #   3. q = F.elu(self.fc_post1(q))
    #        → (B, hidden_dim)
    #   4. q = self.fc_post2(q)
    #        → (B, 2*Z)
    #   5. mean, std_raw = q.chunk(2, dim=-1)
    #   6. std = F.softplus(std_raw) + self.min_std
    #
    #   7. dist = Independent(Normal(mean, std), 1)
    #   8. stoch = dist.rsample()
    #
    #   9. posterior = {
    #          'deter': prior['deter'],   ← prior と同じ h_t を共有
    #          'stoch': stoch,             ← posterior 由来のサンプル
    #          'mean':  mean,
    #          'std':   std,
    #      }
    #
    #   10. return prior, posterior
    #
    # 【ヒント】
    #   - posterior の deter は prior の deter と同じ. 別々に GRU を回すと
    #     RSSM の意味が崩れる.
    #   - 学習時は posterior の stoch を使う. prior の stoch は不要 (KL 計算には
    #     mean/std しか使わない).
    #
    # 【公式参考】 models.py::RSSM.obs_step:
    #   prior = self.img_step(prev_state, prev_action)
    #   x = tf.concat([prior['deter'], embed], -1)
    #   x = self.get('obs1', tfkl.Dense, hidden, self._activation)(x)
    #   x = self.get('obs2', tfkl.Dense, 2*stoch)(x)
    #   mean, std = tf.split(x, 2, -1)
    #   std = tf.nn.softplus(std) + 0.1
    #   stoch = self._sample(mean, std)
    #   post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
    #   return post, prior
    #
    # 【完了条件】
    #   - tests/test_rssm.py::test_obs_step_shape が通る
    #   - tests/test_rssm.py::test_obs_step_uses_embed が通る
    # ═════════════════════════════════════════════════════════════════════
    def obs_step(
        self,
        prev_state: dict,
        prev_action: torch.Tensor,
        embed: torch.Tensor,
    ) -> tuple[dict, dict]:
        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior['deter'], embed], dim=-1)
        x = F.elu(self.fc_post1(x))
        x = self.fc_post2(x)
        mean, std_raw = x.chunk(2, dim=-1)
        std = F.softplus(std_raw) + self.min_std
        dist = Independent(Normal(mean, std), 1)
        stoch = dist.rsample()
        posterior = {
            'deter': prior['deter'],
            'stoch': stoch,
            'mean': mean,
            'std': std,
        }
        return prior, posterior
    # ═════════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#3.1.5]: observe (T ステップ posterior unroll)
    # ─────────────────────────────────────────────────────────────────────
    # 【目的】 (B, T, E) の embed と (B, T, A) の action を受けて,
    #          T ステップ分 obs_step を回し prior/posterior の系列を返す.
    #
    # 【やること】
    #
    #   B, T = embed.shape[:2]
    #   state = init_state                    ← prev_state, posterior を持ち回る
    #
    #   priors, posteriors = [], []
    #   for t in range(T):
    #       prior, post = self.obs_step(state, action[:, t], embed[:, t])
    #       priors.append(prior)
    #       posteriors.append(post)
    #       state = post                      ← ★ 次ステップの prev_state は post
    #
    #   priors = stack_states(priors)         ← {'deter': (B,T,H), ...}
    #   posteriors = stack_states(posteriors)
    #   return priors, posteriors
    #
    # 【ヒント】
    #   - PyTorch の RNN layer ではなく **手動の for ループ**. 各 step で obs_step を
    #     呼ぶ必要があるため (vanilla GRU では posterior を組み込めない).
    #   - state を post で更新するのが重要 (prior で更新すると teacher forcing にならない).
    #   - stack_states は src/dreamer/utils.py で提供.
    #
    # 【完了条件】
    #   - tests/test_rssm.py::test_observe_shape が通る
    #   - tests/test_rssm.py::test_observe_uses_posterior_for_next が通る
    # ═════════════════════════════════════════════════════════════════════
    def observe(
        self,
        embed: torch.Tensor,
        action: torch.Tensor,
        init_state: dict,
    ) -> tuple[dict, dict]:
        priors, posteriors = [], []
        state = init_state
        for t in range(action.shape[1]):
            prior, post = self.obs_step(
                state, action[:, t], embed[:, t]
            )
            priors.append(prior)
            posteriors.append(post)
            state = post
        return stack_states(priors), stack_states(posteriors)
    # ═════════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#3.1.6]: imagine (T ステップ prior-only unroll)
    # ─────────────────────────────────────────────────────────────────────
    # 【目的】 観測なしで T ステップ想像 rollout. Phase 5 / 6 で actor と組み合わせて
    #          想像 rollout に使う.
    #
    # 【やること】
    #
    #   B, T = action.shape[:2]
    #   state = init_state
    #
    #   priors = []
    #   for t in range(T):
    #       prior = self.img_step(state, action[:, t])
    #       priors.append(prior)
    #       state = prior                     ← ★ 想像なので prior で更新
    #
    #   return stack_states(priors)
    #
    # 【ヒント】
    #   - observe との違いは「embed を使わない」「state を prior で更新する」だけ.
    #   - Phase 5 で「行動」も actor から取るが、Phase 3 単体では action は外部入力.
    #
    # 【完了条件】
    #   - tests/test_rssm.py::test_imagine_shape が通る
    # ═════════════════════════════════════════════════════════════════════
    def imagine(self, action: torch.Tensor, init_state: dict) -> dict:
        priors = []
        state = init_state
        for t in range(action.shape[1]):
            prior = self.img_step(state, action[:, t])
            priors.append(prior)
            state = prior
        return stack_states(priors)
    # ═════════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#3.1.7]: kl_loss (KL(q || p) with free nats)
    # ─────────────────────────────────────────────────────────────────────
    # 【目的】 posterior と prior の KL ダイバージェンスを計算し、free nats clamp.
    #
    # 【やること】
    #
    #   1. p_dist = Independent(Normal(prior['mean'], prior['std']), 1)
    #   2. q_dist = Independent(Normal(post['mean'],  post['std']),  1)
    #   3. kl = kl_divergence(q_dist, p_dist)        ← q から p への KL. (B, T)
    #   4. kl = kl.mean()                             ← scalar
    #   5. kl = torch.clamp(kl, min=free_nats)        ← scalar に対して max
    #   6. return kl
    #
    # 【ヒント】
    #   - 順番が大事: KL(q || p), つまり posterior が前. Normal(q, p) は逆になる.
    #   - free_nats は scalar 全体に対してクランプ (per-element ではない).
    #     公式コードもそう. これにより勾配が滑らかになる.
    #   - Independent(Normal(...), 1) で「最後 1 軸を独立 Gaussian の event」と
    #     扱う → kl_divergence が Z 軸を sum してくれる.
    #
    # 【数式メモ】
    #   KL(q(z) || p(z)) = log(σ_p/σ_q) + (σ_q² + (μ_q - μ_p)²) / (2σ_p²) - 1/2
    #   ↑ これを diag Gaussian で各次元で計算して sum.
    #
    # 【完了条件】
    #   - tests/test_rssm.py::test_kl_scalar が通る
    #   - tests/test_rssm.py::test_kl_free_nats_clamp が通る
    #   - tests/test_rssm.py::test_kl_decreases が通る (smoke 学習)
    # ═════════════════════════════════════════════════════════════════════
    def kl_loss(
        self,
        post: dict,
        prior: dict,
        free_nats: float = 3.0,
    ) -> torch.Tensor:
        p_dist = Independent(Normal(prior['mean'], prior['std']), 1)
        q_dist = Independent(Normal(post['mean'], post['std']), 1)
        kl = kl_divergence(q_dist, p_dist)
        kl = kl.mean()
        kl = torch.clamp(kl, min=free_nats)
        return kl
    # ─────────────────────────────────────────────────────────────────────
    # 補助: state から (h, s) を concat した feature を作る
    # (Phase 4 で decoder, reward_head に渡すため)
    # ─────────────────────────────────────────────────────────────────────

    def get_feat(self, state: dict) -> torch.Tensor:
        """state dict から feat = concat(deter, stoch) を返す.

        (B, T, deter_dim+stoch_dim) または (B, deter_dim+stoch_dim).
        """
        return torch.cat([state["deter"], state["stoch"]], dim=-1)

    @property
    def feat_dim(self) -> int:
        """Phase 4 の Decoder の feat_dim に渡す値."""
        return self.deter_dim + self.stoch_dim
