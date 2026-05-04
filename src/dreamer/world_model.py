"""World Model: Encoder + RSSM + Decoder + RewardHead を統合.

Phase 4 / File 1.

設計書: docs/design_notes.md §10.4 (Reward), §10.7 (WM 損失)
出典:   公式 https://github.com/danijar/dreamer/blob/master/models.py
        + dreamer.py::Dreamer._train (損失計算)

─── データ規約 ─────────────────────────────────────────────────────────
ReplayBuffer.sample() の戻り値:
    obs:      (B, T+1, 3, 64, 64) float32 ∈ [-0.5, 0.5]
    action:   (B, T,   A) float32
    reward:   (B, T,)    float32
    discount: (B, T,)    float32

学習で使うのは:
    obs_seq = obs[:, :T]   ← 最初の T フレーム (T+1 ではなく T)
    action  = action[:, :T]
    reward  = reward[:, :T]
    各 t ∈ [0..T-1] で「state を作って、対応する obs/reward を予測する」

注: 「prev_action[t]」の解釈は、action[t] を「state t を作るときに与える action」
    と捉える. 公式コードもこの解釈. obs[t] と同じ index で同期させる.

─── 損失 ──────────────────────────────────────────────────────────────
L_recon  = 0.5 · ||x̂ − x||²  (per pixel sum, batch mean)
L_reward = 0.5 · (r̂ − r)²
L_kl     = max(KL(post || prior), free_nats)
L_wm     = L_recon + L_reward + kl_scale · L_kl

注: 公式は Normal(μ, 1).log_prob() を使うが、これは Normal(μ, 1) → MSE/2 + const
    と等価. design_notes §10.11 参照.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ConvEncoder
from .decoder import ConvDecoder
from .rssm import RSSM
from .utils import stack_states


# ═════════════════════════════════════════════════════════════════════════
# RewardHead
# ═════════════════════════════════════════════════════════════════════════


class RewardHead(nn.Module):
    """feat (B, T, F) → reward 予測 (B, T).

    アーキテクチャ (design_notes §10.4):
        Dense(num_units, ELU) × num_layers → Dense(1) (no act)
        → Normal(mean=output, std=1) (実装は MSE で等価)

    Args:
        feat_dim: 入力次元 (Phase 4 では rssm.feat_dim = 230).
        num_units: 各層の幅 (公式 400).
        num_layers: 隠れ層の数 (公式 3).
    """

    def __init__(self, feat_dim: int, num_units: int = 400, num_layers: int = 3):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_units = num_units
        self.num_layers = num_layers

        # ═════════════════════════════════════════════════════════════════
        # 📝 WRITE ME [#4.1.1]: RewardHead.__init__
        # ─────────────────────────────────────────────────────────────────
        # 【目的】 num_layers 層の Dense(num_units, ELU) + 出力 Dense(1) を構築.
        #
        # 【やること】
        #   nn.Sequential を 1 つ作る. 構造:
        #     Linear(feat_dim,   num_units), ELU,
        #     Linear(num_units,  num_units), ELU,
        #     Linear(num_units,  num_units), ELU,
        #     Linear(num_units,  1)             ← 最終層は活性化なし
        #
        #   コード例 (num_layers=3 を想定):
        #     layers = []
        #     in_dim = feat_dim
        #     for _ in range(num_layers):
        #         layers += [nn.Linear(in_dim, num_units), nn.ELU()]
        #         in_dim = num_units
        #     layers.append(nn.Linear(num_units, 1))
        #     self.mlp = nn.Sequential(*layers)
        #
        # 【ヒント】
        #   - 最終層は活性化なし (生のスカラー出力 = Normal の mean)
        #   - 全 hidden は **ELU** (ReLU ではない. RSSM/MLP は ELU 統一)
        #   - num_layers は将来 4 や 5 に変えたい時に効くので可変にしておく
        #
        # 【公式参考】 models.py::DenseDecoder
        #   for index in range(self._layers):
        #       x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
        #   x = self.get('hout', tfkl.Dense, np.prod(self._shape))(x)
        #
        # 【完了条件】
        #   - tests/test_world_model.py::test_reward_head_shape が通る
        # ═════════════════════════════════════════════════════════════════
        layers = []
        in_dim = feat_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, num_units), nn.ELU()]
            in_dim = num_units
        layers.append(nn.Linear(num_units, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:

        """feat (B, T, F) → reward (B, T).

        最終 squeeze で出力 (B, T, 1) → (B, T) にする.
        """
        out = self.mlp(feat)
        return out.squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════
# WorldModel
# ═════════════════════════════════════════════════════════════════════════


class WorldModel(nn.Module):
    """Encoder + RSSM + Decoder + RewardHead を統合した世界モデル.

    Args:
        action_dim:  walker=6 など.
        cfg_model:   dict. configs/dmc_walker.yaml の "model" セクションに対応.
                     → cnn_depth, rssm_deter, rssm_stoch, rssm_hidden, min_std,
                        num_units, num_layers
    """

    def __init__(
        self,
        action_dim: int,
        cnn_depth: int = 32,
        rssm_deter: int = 200,
        rssm_stoch: int = 30,
        rssm_hidden: int = 200,
        min_std: float = 0.1,
        num_units: int = 400,
        num_layers: int = 3,
    ):
        super().__init__()
        self.action_dim = action_dim

        # ═════════════════════════════════════════════════════════════════
        # 📝 WRITE ME [#4.1.2]: WorldModel.__init__
        # ─────────────────────────────────────────────────────────────────
        # 【目的】 サブモジュール (encoder, rssm, decoder, reward_head) を構築.
        #
        # 【やること】
        #
        #   1. Encoder:
        #      self.encoder = ConvEncoder(depth=cnn_depth)
        #
        #   2. RSSM:
        #      self.rssm = RSSM(
        #          action_dim=action_dim,
        #          embed_dim=self.encoder.embed_dim,   # = 1024
        #          deter_dim=rssm_deter,
        #          stoch_dim=rssm_stoch,
        #          hidden_dim=rssm_hidden,
        #          min_std=min_std,
        #      )
        #
        #   3. Decoder:
        #      self.decoder = ConvDecoder(
        #          feat_dim=self.rssm.feat_dim,        # = 230
        #          depth=cnn_depth,
        #      )
        #
        #   4. RewardHead:
        #      self.reward_head = RewardHead(
        #          feat_dim=self.rssm.feat_dim,
        #          num_units=num_units,
        #          num_layers=num_layers,
        #      )
        #
        # 【ヒント】
        #   - 順番に依存あり: encoder.embed_dim を使う rssm を後で.
        #   - 全 submodule は self.* に登録すれば nn.Module が parameter を集める.
        #
        # 【完了条件】
        #   - tests/test_world_model.py::test_init が通る
        # ═════════════════════════════════════════════════════════════════
        self.encoder = ConvEncoder(depth=cnn_depth)
        self.rssm = RSSM(
            action_dim=action_dim,
            embed_dim=self.encoder.embed_dim,   # = 1024
            deter_dim=rssm_deter,
            stoch_dim=rssm_stoch,
            hidden_dim=rssm_hidden,
            min_std=min_std,
        )
        self.decoder = ConvDecoder(
            feat_dim=self.rssm.feat_dim,        # = 230
            depth=cnn_depth,
        )
        self.reward_head = RewardHead(
            feat_dim=self.rssm.feat_dim,
            num_units=num_units,
            num_layers=num_layers,
        )
    # ═════════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#4.1.3]: WorldModel.observe
    # ─────────────────────────────────────────────────────────────────────
    # 【目的】 obs と action から (priors, posts, embed) を計算する.
    #          Phase 5 以降で imagine の起点として使う.
    #
    # 【やること】
    #
    #   B, T = obs.shape[0], obs.shape[1]
    #   device = obs.device
    #
    #   embed = self.encoder(obs)
    #     → (B, T, E=1024)
    #
    #   init_state = self.rssm.initial_state(B, device)
    #
    #   priors, posts = self.rssm.observe(embed, action, init_state)
    #     → 各 dict が (B, T, ...)
    #
    #   return priors, posts, embed
    #
    # 【ヒント】
    #   - obs は既に (B, T, 3, 64, 64) で渡される (compute_loss の前処理で .)
    #   - device は obs から取れる (obs.device)
    #
    # 【完了条件】
    #   - tests/test_world_model.py::test_observe_shape が通る
    # ═════════════════════════════════════════════════════════════════════
    def observe(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[dict, dict, torch.Tensor]:
        B, T = obs.shape[0], obs.shape[1]
        device = obs.device

        embed = self.encoder(obs)
        # (B, T, E=1024)

        init_state = self.rssm.initial_state(B, device)

        priors, posts = self.rssm.observe(embed, action, init_state)
        return priors, posts, embed
        """obs と action を受けて (priors, posts, embed) を返す.

        Args:
            obs: (B, T, 3, 64, 64) float32.
            action: (B, T, A) float32.

        Returns:
            priors, posts: 各 state dict (B, T, ...).
            embed: (B, T, E=1024).
        """

    # ═════════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#4.1.4]: WorldModel.compute_loss
    # ─────────────────────────────────────────────────────────────────────
    # 【目的】 batch から L_wm = L_recon + L_reward + β·L_kl を計算.
    #
    # 【やること】
    #
    #   1. データ整形:
    #      T = batch['action'].shape[1]
    #      obs_seq = batch['obs'][:, :T]          # (B, T, 3, 64, 64)
    #      action  = batch['action']              # (B, T, A)
    #      reward  = batch['reward']              # (B, T)
    #
    #   2. observe で priors, posts, embed を計算:
    #      priors, posts, embed = self.observe(obs_seq, action)
    #
    #   3. feat を作る:
    #      feat = self.rssm.get_feat(posts)       # (B, T, deter+stoch)
    #
    #   4. 再構成損失:
    #      recon = self.decoder(feat)             # (B, T, 3, 64, 64)
    #      # Normal(μ, 1).log_prob = -0.5 MSE - const なので,
    #      # 0.5 * MSE per-sample-summed の平均 を使う.
    #      recon_loss = 0.5 * F.mse_loss(
    #          recon, obs_seq, reduction='none'
    #      ).sum(dim=(-3, -2, -1)).mean()
    #
    #   5. 報酬損失:
    #      reward_pred = self.reward_head(feat)   # (B, T)
    #      reward_loss = 0.5 * F.mse_loss(reward_pred, reward)
    #         ← per-element 平均で OK (報酬は scalar なので)
    #
    #   6. KL 損失:
    #      kl = self.rssm.kl_loss(posts, priors, free_nats=free_nats)
    #
    #   7. 合算:
    #      total = recon_loss + reward_loss + kl_scale * kl
    #
    #   8. metrics dict:
    #      metrics = {
    #          'wm/total':  total.item(),
    #          'wm/recon':  recon_loss.item(),
    #          'wm/reward': reward_loss.item(),
    #          'wm/kl':     kl.item(),
    #      }
    #
    #   9. return total, metrics, posts
    #      ← posts は Phase 6 で想像 rollout の起点として使う
    #
    # 【ヒント】
    #   - recon_loss の reduction='none' → sum で「画像全体の負対数尤度」を作る.
    #     reduction='mean' を直接使うと 1 ピクセル平均になり、loss が 200 倍小さくなる.
    #   - reward は scalar なのでそのまま reduction='mean' で OK.
    #   - free_nats は config から渡す (デフォルト 3.0).
    #
    # 【公式参考】 dreamer.py::Dreamer._train
    #   image_pred = self._decode(feat)
    #   reward_pred = self._reward(feat)
    #   likes.image  = tf.reduce_mean(image_pred.log_prob(data['image']))
    #   likes.reward = tf.reduce_mean(reward_pred.log_prob(data['reward']))
    #   div = tf.maximum(tfd.kl_divergence(post, prior).mean(), self._c.free_nats)
    #   model_loss = self._c.kl_scale * div - sum(likes.values())
    #
    # 【完了条件】
    #   - tests/test_world_model.py::test_compute_loss_shape が通る
    #   - tests/test_world_model.py::test_compute_loss_decreases が通る
    # ═════════════════════════════════════════════════════════════════════
    def compute_loss(
        self,
        batch: dict,
        free_nats: float = 3.0,
        kl_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict, dict]:
        """batch (ReplayBuffer.sample 形式) から WM 損失を計算.

        Args:
            batch: dict with keys 'obs', 'action', 'reward', 'discount'.
                   全て torch.Tensor で device に乗っている前提.
            free_nats: KL の下限.
            kl_scale: KL の重み β.

        Returns:
            total: scalar (backward 可能).
            metrics: 各損失成分の float dict.
            posts: 想像 rollout の起点として使う posterior dict (B, T, ...).
        """
        T = batch['action'].shape[1]
        obs_seq = batch['obs'][:, :T]          # (B, T, 3, 64, 64)
        action  = batch['action']              # (B, T, A)
        reward  = batch['reward']              # (B, T)

        priors, posts, embed = self.observe(obs_seq, action)
        feat = self.rssm.get_feat(posts)
        
        reconstruction = self.decoder(feat)
        recon_loss = 0.5 * F.mse_loss(reconstruction, obs_seq, reduction='none').sum(dim=(-3, -2, -1)).mean()

        reward_pred = self.reward_head(feat)
        reward_loss = 0.5 * F.mse_loss(reward_pred, reward)

        kl = self.rssm.kl_loss(posts, priors, free_nats=free_nats)
        total = recon_loss + reward_loss + kl_scale * kl

        metrics = {
            'wm/total':  total.item(),
            'wm/recon':  recon_loss.item(),
            'wm/reward': reward_loss.item(),
            'wm/kl':     kl.item(),
        }

        return total, metrics, posts


    # ═════════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#4.1.5]: WorldModel.update
    # ─────────────────────────────────────────────────────────────────────
    # 【目的】 1 step の WM 学習: zero_grad → backward → grad_clip → step.
    #
    # 【やること】
    #
    #   total, metrics, posts = self.compute_loss(batch, free_nats, kl_scale)
    #
    #   optimizer.zero_grad()
    #   total.backward()
    #   torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
    #   optimizer.step()
    #
    #   metrics['wm/grad_norm'] = ... ← 計測したければ clip_grad_norm_ の戻り値
    #   return metrics, posts
    #
    # 【ヒント】
    #   - clip_grad_norm_ は in-place で勾配をスケールし、元の grad norm を返す.
    #     `grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)`
    #   - posts はそのまま返す. Phase 6 で「想像 rollout の起点」として使う.
    #
    # 【完了条件】
    #   - tests/test_world_model.py::test_update_step_runs が通る
    # ═════════════════════════════════════════════════════════════════════
    def update(
        self,
        batch: dict,
        optimizer: torch.optim.Optimizer,
        free_nats: float = 3.0,
        kl_scale: float = 1.0,
        grad_clip: float = 100.0,
    ) -> tuple[dict, dict]:
        total, metrics, posts = self.compute_loss(batch, free_nats, kl_scale)
        optimizer.zero_grad()
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        optimizer.step()
        metrics['wm/grad_norm'] = grad_norm.item()
        return metrics, posts

    # ═════════════════════════════════════════════════════════════════════
    # Phase 5: Imagination rollout
    # ─────────────────────────────────────────────────────────────────────
    # 想像 rollout は「学習済み WM の latent space を policy で進む」操作.
    # Phase 6 で actor と組み合わせて Actor-Critic 学習に使う.
    # Phase 5 ではテストとして random policy で動かし、暴走しないことを確認.
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def flatten_state_for_imagine(state: dict) -> dict:
        """(B, T, ...) → (B*T, ...) に flatten.

        observe の戻り値 posts (B, T, ...) を imagine の起点として使うため,
        各 posterior step を独立した「開始点」として扱う.
        Phase 6 でも同じ形で使う.
        """
        return {k: v.reshape(-1, *v.shape[2:]) for k, v in state.items()}

    # ═════════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#5.1.1]: WorldModel.imagine_rollout
    # ─────────────────────────────────────────────────────────────────────
    # 【目的】 学習済み WM の中で policy_fn に従って H ステップ想像する.
    #          観測なし (img_step のみ). reward_head で各ステップの報酬を予測.
    #
    # 【やること】
    #
    #   B 個のスタート状態から H ステップ前進する:
    #
      
    #
    #   return {
    #       'states':  states_seq,    # 状態系列 dict (B, H, ...)
    #       'feat':    feat,          # 特徴量 (B, H, F)
    #       'action':  actions,       # 取った行動 (B, H, A)
    #       'reward':  reward_pred,   # 予測報酬 (B, H)
    #   }
    #
    # 【ヒント】
    #
    #   1. policy_fn のシグネチャ:
    #        feat (B, F) → action (B, A)
    #      Phase 5: random policy (torch.randn).
    #      Phase 6: actor (TanhNormal を rsample).
    #
    #   2. 「ステップ h の action」の意味:
    #        state[h-1] (前) → policy_fn → action[h] → state[h] (次)
    #      つまり action[h] は state[h-1] で決まり、state[h] を生む.
    #      最初のステップ (h=0) では action[0] = policy_fn(init_state) を使い,
    #      state[0] = img_step(init_state, action[0]).
    #      → init_state は出力に含めない (公式と同じ).
    #
    #   3. stack_states は src/dreamer/utils.py で提供 (Phase 3 で追加).
    #
    #   4. ★ Phase 6 では policy_fn 内で「feat を detach してから actor に通す」
    #      パターンを使う (gradient を WM に戻さないため).
    #      Phase 5 ではそんな細かいことは気にしなくて OK.
    #
    # 【公式参考】 dreamer.py::Dreamer._imagine_ahead
    #   flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
    #   start = {k: flatten(v) for k, v in post.items()}
    #   policy = lambda state: self._actor(get_feat(state)).sample()
    #   states = static_scan(
    #       lambda prev, _: self._dynamics.img_step(prev, policy(prev)),
    #       tf.range(self._c.horizon),
    #       start,
    #   )
    #   imag_feat = self._dynamics.get_feat(states)
    #
    # 【完了条件】
    #   - tests/test_imagination.py::test_imagine_rollout_shape が通る
    #   - tests/test_imagination.py::test_imagine_no_nan が通る
    #   - tests/test_imagination.py::test_imagine_grad_to_policy が通る
    # ═════════════════════════════════════════════════════════════════════
    def imagine_rollout(
        self,
        init_state: dict,
        policy_fn,
        horizon: int = 15,
    ) -> dict:
        state = init_state                  # dict, 各 (B, ...)
        states_list = []
        actions_list = []
        for h in range(horizon):
            feat_h = self.rssm.get_feat(state)   # (B, F)  ← policy 入力
            action_h = policy_fn(feat_h)         # (B, A)
            state = self.rssm.img_step(state, action_h)
            states_list.append(state)
            actions_list.append(action_h)
        states_seq = stack_states(states_list)        # (B, H, ...)
        actions = torch.stack(actions_list, dim=1)    # (B, H, A)
        feat = self.rssm.get_feat(states_seq)         # (B, H, F)
        reward_pred = self.reward_head(feat)          # (B, H)

        return {
            'states':  states_seq,    # 状態系列 dict (B, H, ...)
            'feat':    feat,          # 特徴量 (B, H, F)
            'action':  actions,       # 取った行動 (B, H, A)
            'reward':  reward_pred,   # 予測報酬 (B, H)
        }
        
        """init_state (B, ...) から policy_fn で H ステップ想像 rollout.

        Args:
            init_state: state dict, 各 tensor (B, ...). すでに flatten 済み前提.
            policy_fn: feat (B, F) → action (B, A) の callable.
            horizon: 想像ステップ数 H.

        Returns:
            dict with:
                'states': state dict (B, H, ...) — 各想像ステップの状態.
                'feat':   (B, H, F) — get_feat(states).
                'action': (B, H, A) — 取った行動.
                'reward': (B, H)    — reward_head の予測.
        """