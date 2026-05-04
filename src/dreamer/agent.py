"""Agent: WorldModel + ActorCritic + 環境 interaction.

Phase 7 / File 1.

設計書: docs/design_notes.md §10.9 学習スケジュール, §10.10 探索
出典:   公式 https://github.com/danijar/dreamer/blob/master/dreamer.py

─── 役割 ────────────────────────────────────────────────────────────────
Agent は thin orchestrator. WM と AC を持ち、外部に対しては:

    agent.act(obs) → action       ← 環境との interaction
    agent.learn(batch) → metrics  ← 1 step の学習
    agent.reset()                  ← episode 開始時に内部 state クリア

を提供する.

─── 内部状態 (env interaction 専用) ─────────────────────────────────────
env step は 1 obs ずつ来るので、RSSM の state を **永続的に** 保持する必要がある:

    self._state:        最後の posterior state dict (B=1, ...)
    self._prev_action:  前 step に取った action (1, A)

reset() でこれらをクリアし、init_state を改めて作る.

─── 探索 (verified, design_notes §10.10) ────────────────────────────────
学習中の env interaction では:
    action = actor.act(feat, training=True)               ← TanhNormal rsample
    action = action + Normal(0, 0.3).sample()             ← 探索ノイズ
    action = action.clamp(-1, 1)
評価時 (training=False):
    action = actor.act(feat, training=False)              ← deterministic (tanh(mean))
"""

from __future__ import annotations

import numpy as np
import torch

from .world_model import WorldModel
from .actor_critic import ActorCritic


class Agent:
    """Dreamer agent. WM + AC を持つ.

    使い方:
        agent = Agent(action_dim=6, ...).to(device)

        # Env interaction
        agent.reset()
        for step in range(1000):
            obs = env.reset() if step == 0 else obs
            action = agent.act(obs, training=True)   # numpy (A,)
            obs, r, done, _ = env.step(action)
            ...

        # Training
        for _ in range(100):
            batch = buf.sample(...)
            metrics = agent.learn(batch)
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
        wm_lr: float = 6e-4,
        actor_lr: float = 8e-5,
        critic_lr: float = 8e-5,
        kl_free_nats: float = 3.0,
        kl_scale: float = 1.0,
        grad_clip: float = 100.0,
        imagine_horizon: int = 15,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        expl_noise: float = 0.3,
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.device = device

        # 学習ハイパラを保持
        self.kl_free_nats = kl_free_nats
        self.kl_scale = kl_scale
        self.grad_clip = grad_clip
        self.imagine_horizon = imagine_horizon
        self.gamma = gamma
        self.lambda_ = lambda_
        self.expl_noise = expl_noise

        # ═════════════════════════════════════════════════════════════════
        # 📝 WRITE ME [#7.1.1]: Agent.__init__
        # ─────────────────────────────────────────────────────────────────
        # 【目的】 WorldModel と ActorCritic を作り、3 つの optimizer を準備.
        #
        # 【やること】
        #
        #   1. WorldModel:
        #      self.wm = WorldModel(
        #          action_dim=action_dim,
        #          cnn_depth=cnn_depth,
        #          rssm_deter=rssm_deter,
        #          rssm_stoch=rssm_stoch,
        #          rssm_hidden=rssm_hidden,
        #          min_std=min_std,
        #          num_units=num_units,
        #          num_layers=num_layers,
        #      ).to(device)
        #
        #   2. ActorCritic:
        #      self.ac = ActorCritic(
        #          feat_dim=self.wm.rssm.feat_dim,    ← WM の feat_dim を使う
        #          action_dim=action_dim,
        #          num_units=num_units,
        #          num_layers=num_layers,
        #      ).to(device)
        #
        #   3. 3 つの optimizer (公式と同じ学習率):
        #      self.wm_opt     = torch.optim.Adam(self.wm.parameters(),         lr=wm_lr)
        #      self.actor_opt  = torch.optim.Adam(self.ac.actor.parameters(),   lr=actor_lr)
        #      self.critic_opt = torch.optim.Adam(self.ac.critic.parameters(),  lr=critic_lr)
        #
        #   4. 内部状態の初期化:
        #      self._state = None         ← episode 内で永続化する RSSM state
        #      self._prev_action = None   ← 前 step の action
        #
        # 【ヒント】
        #   - WM と AC は別ファイルだが、Agent では同じレベルで扱う.
        #   - WM optimizer は **WM 全体** を更新.
        #     AC optimizer は **actor / critic を別々に** (公式と同じ).
        #   - device は cuda or cpu. 各 module を to(device) する.
        #
        # 【完了条件】
        #   - tests/test_agent.py::test_init が通る
        # ═════════════════════════════════════════════════════════════════
        self.wm = WorldModel(
            action_dim=action_dim,
            cnn_depth=cnn_depth,
            rssm_deter=rssm_deter,
            rssm_stoch=rssm_stoch,
            rssm_hidden=rssm_hidden,
            min_std=min_std,
            num_units=num_units,
            num_layers=num_layers,
        ).to(device)

        self.ac = ActorCritic(
            feat_dim=self.wm.rssm.feat_dim,
            action_dim=action_dim,
            num_units=num_units,
            num_layers=num_layers,
        ).to(device)

        self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=wm_lr)
        self.actor_opt = torch.optim.Adam(self.ac.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.ac.critic.parameters(), lr=critic_lr)

        self._state = None
        self._prev_action = None
    # ═════════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#7.1.2]: Agent.reset
    # ─────────────────────────────────────────────────────────────────────
    # 【目的】 episode 開始時に内部 state をクリア.
    #
    # 【やること】
    #   self._state = None
    #   self._prev_action = None
    #
    # 【完了条件】
    #   - tests/test_agent.py::test_reset_clears_state が通る
    # ═════════════════════════════════════════════════════════════════════
    def reset(self) -> None:
        self._state = None
        self._prev_action = None
    # ─────────────────────────────────────────────────────────────────────
    # 補助: numpy obs → torch tensor (1, 1, 3, 64, 64) on device
    # ─────────────────────────────────────────────────────────────────────

    def _preprocess_obs(self, obs: np.ndarray) -> torch.Tensor:
        """env の (H, W, C) uint8 → (1, 1, 3, 64, 64) float32 [-0.5, 0.5]."""
        # uint8 [0, 255] → float [-0.5, 0.5]
        x = obs.astype(np.float32) / 255.0 - 0.5
        # HWC → CHW, batch + time dim 追加
        x = np.transpose(x, (2, 0, 1))[None, None]  # (1, 1, 3, 64, 64)
        return torch.from_numpy(x).to(self.device)

    # ═════════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#7.1.3]: Agent.act
    # ─────────────────────────────────────────────────────────────────────
    # 【目的】 1 obs → 1 action. 内部 RSSM state を更新しながら.
    #
    # 【ステップ】 (Phase 7 の心臓部. 各 step が何をしているか追うこと.)
    #
    #   with torch.no_grad():            ← inference のみ. 学習は learn() で.
    #
    #     1. obs 前処理:
    #        x = self._preprocess_obs(obs)        # (1, 1, 3, 64, 64) on device
    #
    #     2. encoder で embed:
    #        embed = self.wm.encoder(x)            # (1, 1, E)
    #        embed = embed.squeeze(1)              # (1, E)
    #
    #     3. 初回 (state が None) なら初期化:
    #        if self._state is None:
    #            self._state = self.wm.rssm.initial_state(1, self.device)
    #            self._prev_action = torch.zeros(1, self.action_dim, device=self.device)
    #
    #     4. RSSM の posterior step:
    #        _, post = self.wm.rssm.obs_step(self._state, self._prev_action, embed)
    #        self._state = post                    ← 永続化
    #
    #     5. feat 計算 → actor:
    #        feat = self.wm.rssm.get_feat(post)    # (1, F)
    #        action = self.ac.actor.act(feat, training=training)  # (1, A) ∈ [-1, 1]
    #
    #     6. 探索ノイズ (training=True のみ):
    #        if training:
    #            noise = torch.randn_like(action) * self.expl_noise
    #            action = (action + noise).clamp(-1, 1)
    #
    #     7. 内部 state 更新:
    #        self._prev_action = action
    #
    #   return action.squeeze(0).cpu().numpy()     # numpy (A,)
    #
    # 【ヒント】
    #   - torch.no_grad() を必ずかける (env step では学習しない).
    #   - 「self._state が dict」「self._prev_action が tensor」で永続化.
    #   - action は最終的に numpy で返す (env.step が numpy を期待).
    #   - 最初の step は obs_step で h_0 が GRU の zeros から進む.
    #
    # 【完了条件】
    #   - tests/test_agent.py::test_act_shape が通る
    #   - tests/test_agent.py::test_act_state_persistence が通る
    #   - tests/test_agent.py::test_act_eval_deterministic が通る
    # ═════════════════════════════════════════════════════════════════════
    def act(self, obs: np.ndarray, training: bool = True) -> np.ndarray:
        """1 obs → 1 action.

        Args:
            obs: numpy (H=64, W=64, C=3) uint8 from env.
            training: True で探索ノイズあり. False で deterministic.

        Returns:
            action: numpy (A,) float32 in [-1, 1].
        """
        with torch.no_grad():
            x = self._preprocess_obs(obs)
            embed = self.wm.encoder(x).squeeze(1)

            if self._state is None:
                self._state = self.wm.rssm.initial_state(1, self.device)
                self._prev_action = torch.zeros(1, self.action_dim, device=self.device)
            
            _, post = self.wm.rssm.obs_step(self._state, self._prev_action, embed)
            self._state = post

            feat = self.wm.rssm.get_feat(post)
            action = self.ac.actor.act(feat, training=training)

            if training:
                noise = torch.randn_like(action) * self.expl_noise
                action = (action + noise).clamp(-1, 1)
            
            self._prev_action = action
            return action.squeeze(0).cpu().numpy()
                        
    # ═════════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#7.1.4]: Agent.learn
    # ─────────────────────────────────────────────────────────────────────
    # 【目的】 1 batch 分の学習: WM update → AC update.
    #
    # 【やること】
    #
    #   1. WM 更新:
    #      wm_metrics, posts = self.wm.update(
    #          batch, self.wm_opt,
    #          free_nats=self.kl_free_nats, kl_scale=self.kl_scale, grad_clip=self.grad_clip,
    #      )
    #
    #   2. AC 更新 (WM は固定. posts を起点に想像 rollout):
    #      ac_metrics = self.ac.update(
    #          self.wm, posts, self.actor_opt, self.critic_opt,
    #          horizon=self.imagine_horizon, gamma=self.gamma, lambda_=self.lambda_,
    #          grad_clip=self.grad_clip,
    #      )
    #
    #   3. metrics をマージして返す:
    #      return {**wm_metrics, **ac_metrics}
    #
    # 【ヒント】
    #   - WM update が posts (B, T, ...) を返す → そのまま AC update に渡す.
    #   - WM の grad は WM optimizer のみが触るので、AC update で WM が壊れない.
    #   - dict のマージは {**a, **b} が読みやすい.
    #
    # 【完了条件】
    #   - tests/test_agent.py::test_learn_returns_metrics が通る
    # ═════════════════════════════════════════════════════════════════════
    def learn(self, batch: dict) -> dict:
        """1 train step の WM + AC 更新.

        Args:
            batch: ReplayBuffer.sample が返す dict (torch tensor, on device).

        Returns:
            metrics: 'wm/...' と 'ac/...' をマージした dict.
        """
        wm_metrics, posts = self.wm.update(
            batch, self.wm_opt,
            free_nats=self.kl_free_nats, kl_scale=self.kl_scale, grad_clip=self.grad_clip,
        )
        ac_metrics = self.ac.update(
            self.wm, posts, self.actor_opt, self.critic_opt,
            horizon=self.imagine_horizon, gamma=self.gamma, lambda_=self.lambda_,
            grad_clip=self.grad_clip,
        )
        return {**wm_metrics, **ac_metrics}
        
    # ─────────────────────────────────────────────────────────────────────
    # 補助メソッド (Claude 提供): evaluate, save, load
    # ─────────────────────────────────────────────────────────────────────

    def evaluate(self, env, n_episodes: int = 10, max_steps: int = 1000) -> float:
        """N エピソード走らせて mean return を返す (deterministic, no exploration).

        eval 中も _state は使う (各 episode 内で永続) が、reset で消す.
        """
        returns = []
        for _ in range(n_episodes):
            self.reset()
            obs = env.reset()
            ep_return = 0.0
            for _ in range(max_steps):
                action = self.act(obs, training=False)
                obs, r, done, _ = env.step(action)
                ep_return += r
                if done:
                    break
            returns.append(ep_return)
        return float(np.mean(returns))

    def save(self, path: str) -> None:
        """全 module + optimizer の state を保存."""
        torch.save({
            'wm': self.wm.state_dict(),
            'actor': self.ac.actor.state_dict(),
            'critic': self.ac.critic.state_dict(),
            'wm_opt': self.wm_opt.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.wm.load_state_dict(ckpt['wm'])
        self.ac.actor.load_state_dict(ckpt['actor'])
        self.ac.critic.load_state_dict(ckpt['critic'])
        self.wm_opt.load_state_dict(ckpt['wm_opt'])
        self.actor_opt.load_state_dict(ckpt['actor_opt'])
        self.critic_opt.load_state_dict(ckpt['critic_opt'])
