"""Replay Buffer (episode 単位保存 + チャンクサンプル).

Phase 1 / File 2.

設計:
  - エピソードを list[dict] で保持 (時系列が分断されない)
  - sample 時に「(episode index, start_t)」をランダム選択し、
    長さ T の連続チャンクを切り出す
  - 画像は uint8 で保存、sample 時に float32 [-0.5, 0.5] へ変換
    (RAM 削減のため。50 万 frame で 6GB → 24GB の差)
  - capacity 超過時は古いエピソードから FIFO で削除

形状規約 (docs/design_notes.md §4):
  - obs:      (T+1, H, W, C) uint8     ← episode 内
  - action:   (T, A) float32
  - reward:   (T,)  float32
  - discount: (T,)  float32
  sample 後: 全て (B, T, ...) で先頭 batch 次元

公式参考:
    https://github.com/danijar/dreamer/blob/master/tools.py (load_episodes 等)
    https://github.com/juliusfrost/dreamer-pytorch/blob/master/dreamer/replay.py
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Episode 単位で軌跡を保持し、長さ T のチャンクをサンプルする.

    Attributes:
        capacity: 全 step 数の上限. 超えたら古い episode から削除.
        obs_shape: (H, W, C) tuple.
        action_dim: 連続 action の次元.
        _episodes: list[dict]. 各 dict は obs/action/reward/discount を持つ.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, int, int],
        action_dim: int,
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self._episodes: list[dict] = []

    # ─────────────────────────────────────────────────────────────────
    # 状態クエリ
    # ─────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """全 episode の合計 step 数 (= 合計 action 数)."""
        return sum(len(ep["action"]) for ep in self._episodes)

    def num_episodes(self) -> int:
        return len(self._episodes)

    # ═════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#1.2.1]: add_episode
    # ─────────────────────────────────────────────────────────────────
    # 【目的】 エピソードを buffer に追加する. capacity 超過なら古い順に削除.
    #
    # 【やること】
    #   1. episode の形状を validate (assert で十分)
    #        - obs.shape == (T+1, *self.obs_shape)
    #        - action.shape == (T, self.action_dim)
    #        - reward.shape == (T,)
    #        - discount.shape == (T,)
    #        - dtype: obs=uint8, others=float32
    #   2. self._episodes.append(episode)
    #   3. while len(self) > self.capacity:
    #          self._episodes.pop(0)   # FIFO
        
    #
    # 【ヒント】
    #   - 「T+1 obs と T action」なのは「初期 obs + 各 step 後の obs」のため
    #   - assert で形状チェックを書いておくと、Phase 7 でデバッグが楽
    #§
    # 【完了条件】
    #   - tests/test_replay.py::test_add_and_len が通る
    # ═════════════════════════════════════════════════════════════════
    def add_episode(self, episode: dict) -> None:
        T = episode["action"].shape[0]
        assert episode["obs"].shape == (T+1, *self.obs_shape)
        assert episode["action"].shape == (T, self.action_dim)
        assert episode["reward"].shape == (T,)
        assert episode["discount"].shape == (T,)

        self._episodes.append(episode)
        while len(self) > self.capacity:
            self._episodes.pop(0)

    # ═════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#1.2.2]: sample
    # ─────────────────────────────────────────────────────────────────
    # 【目的】 batch_size 個のチャンク (長さ seq_len) を均一サンプリング.
    #
    # 【やること】
    #   1. batch_size 回ループ (B 回):
    #        a. episode を選ぶ
    #             - 単純: np.random.randint(0, len(self._episodes))
    #             - 推奨: episode 長さで重み付け (短い ep に偏らない)
    #               weights = [len(ep['action']) for ep in self._episodes]
    #               idx = np.random.choice(len(weights), p=weights/sum(weights))
    #        b. その episode 内で start ∈ [0, len(action)-seq_len] を選ぶ
    #             - len(action) < seq_len の場合は除外 or リトライ
    #        c. チャンクを切り出す:
    #             obs_chunk      = ep['obs'][start : start+seq_len+1]   # T+1
    #             action_chunk   = ep['action'][start : start+seq_len]
    #             reward_chunk   = ep['reward'][start : start+seq_len]
    #             discount_chunk = ep['discount'][start : start+seq_len]
    #   2. 全チャンクを np.stack で (B, T+1, ...), (B, T, ...) に
    #   3. obs を変換:
    #        - uint8 [0, 255] → float32 [-0.5, 0.5]
    #          式: obs = obs.astype(np.float32) / 255.0 - 0.5
    #        - HWC → CHW : np.transpose(obs, (0, 1, 4, 2, 3))
    #          (B, T+1, H, W, C) → (B, T+1, C, H, W)
    #   4. obs は (B, T+1, ...) のままでもよいし、(B, T, ...) に切ってもよい.
    #      → 本実装では「obs (B, T+1, ...) + action/reward/discount (B, T, ...)」を返す
    #        Phase 3 RSSM では obs[:, :T] を使う.
    #
    # 【ヒント】
    #   - 公式 (Hafner) は obs を (B, T+1) で持ち、最後の obs を「次状態用」に使う
    #   - 形状を間違えると Phase 3 で死ぬ. 必ず print して確認すること
    #
    # 【完了条件】
    #   - tests/test_replay.py::test_sample_shapes が通る
    #   - tests/test_replay.py::test_sample_range が通る
    # ═════════════════════════════════════════════════════════════════
    def sample(self, batch_size: int, seq_len: int) -> dict:
        obs_chunks = []
        action_chunks = []
        reward_chunks = []
        discount_chunks = []
        for i in range(batch_size):
            weights = np.array([len(ep['action']) for ep in self._episodes])
            idx = np.random.choice(len(weights), p=weights/sum(weights))

            ep = self._episodes[idx]
            T = ep['action'].shape[0]

            start = np.random.randint(0, T - seq_len)
            obs_chunk = ep['obs'][start : start + seq_len + 1]
            action_chunk = ep['action'][start : start + seq_len]
            reward_chunk = ep['reward'][start : start + seq_len]
            discount_chunk = ep['discount'][start : start + seq_len]

            obs_chunks.append(obs_chunk)
            action_chunks.append(action_chunk)
            reward_chunks.append(reward_chunk)
            discount_chunks.append(discount_chunk)
        
        obs_chunks = np.stack(obs_chunks, axis=0)
        action_chunks = np.stack(action_chunks, axis=0)
        reward_chunks = np.stack(reward_chunks, axis=0)
        discount_chunks = np.stack(discount_chunks, axis=0)

        obs_chunks = obs_chunks.astype(np.float32) / 255.0 - 0.5
        obs_chunks = np.transpose(obs_chunks, (0, 1, 4, 2, 3))

        return {
            "obs": obs_chunks,
            "action": action_chunks,
            "reward": reward_chunks,
            "discount": discount_chunks,
        }
