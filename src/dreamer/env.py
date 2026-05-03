"""DMControl 環境ラッパー.

Phase 1 / File 1.

64x64 画像観測 + action_repeat に統一する wrapper。
Dreamer 系の論文は基本的に **画像観測のみ**を入力とするので、
DMControl の dict observation は捨てて、`physics.render()` で画像を取る。

公式参考:
    https://github.com/danijar/dreamer/blob/master/wrappers.py
    https://github.com/juliusfrost/dreamer-pytorch/blob/master/dreamer/envs/dmc.py
"""

from __future__ import annotations

import numpy as np
from dm_control import suite


class DMCEnv:
    """DeepMind Control Suite を Dreamer 用にラップする.

    Attributes:
        domain: e.g., "walker"
        task: e.g., "walk"
        image_size: 64 が Dreamer 標準
        action_repeat: 同じ action を何回連続で env に渡すか (DMC 標準は 2)
    """

    def __init__(
        self,
        domain: str,
        task: str,
        image_size: int = 64,
        action_repeat: int = 2,
        seed: int = 0,
    ):
        self.domain = domain
        self.task = task
        self.image_size = image_size
        self.action_repeat = action_repeat

        self._env = suite.load(domain, task, task_kwargs={"random": seed})
        self._action_spec = self._env.action_spec()

    # ─────────────────────────────────────────────────────────────────
    # 公開プロパティ
    # ─────────────────────────────────────────────────────────────────

    @property
    def action_dim(self) -> int:
        """連続 action の次元数."""
        return self._action_spec.shape[0]

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        """画像 obs の形状 (H, W, C)."""
        return (self.image_size, self.image_size, 3)

    # ─────────────────────────────────────────────────────────────────
    # 内部ユーティリティ
    # ─────────────────────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        """64x64x3 uint8 を返す.

        camera_id=0 はデフォルトカメラ。Dreamer 公式と同じ.
        """
        return self._env.physics.render(
            height=self.image_size,
            width=self.image_size,
            camera_id=0,
        )

    # ═════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#1.1.1]: reset
    # ─────────────────────────────────────────────────────────────────
    # 【目的】 環境を初期化し、最初の画像 obs を返す.
    #
    # 【やること】
    #   1. self._env.reset() で TimeStep を取得
    #   2. self._render() で画像を取得して return

    
    #
    # 【ヒント】
    #   - 戻り値は np.ndarray uint8 (H, W, 3)
    #   - DMControl の reset 戻り値は TimeStep (NamedTuple) だが、
    #     画像のみを返すラッパーなので捨てて OK
    #
    # 【完了条件】
    #   - scripts/check_env_wrapper.py の reset 部分が通る
    # ═════════════════════════════════════════════════════════════════
    def reset(self) -> np.ndarray:
        self._env.reset()
        return self._render()

    # ═════════════════════════════════════════════════════════════════
    # 📝 WRITE ME [#1.1.2]: step
    # ─────────────────────────────────────────────────────────────────
    # 【目的】 action を action_repeat 回繰り返し実行し、報酬合算で 1 step.
    #
    # 【やること】
    #   1. action を np.float32 にキャスト・clip [-1, 1]
    #   2. action_repeat 回ループ:
    #        time_step = self._env.step(action)
    #        reward の合計を取る
    #        time_step.last() (= done) になったら break
    #   3. obs = self._render()
    #   4. (obs, total_reward, done, info) を返す

    
        
    #
    # 【ヒント】
    #   - DMControl の reward は最初の TimeStep (FIRST) では None。
    #     reset 直後の reward は無いが、step では数値が返る。安全のため
    #     `time_step.reward or 0.0` のように None ガードしてもよい
    #   - time_step.last() : エピソード終了 True/False
    #   - time_step.discount : 通常 1.0、絶対終端で 0.0
    #   - info: 空の dict {} で OK
    #
    # 【数式メモ】
    #   total_reward = Σ_{k=1}^{action_repeat} reward_k
    #
    # 【公式参考】
    #   https://github.com/danijar/dreamer/blob/master/wrappers.py
    #   ActionRepeat クラスの step を参照
    #
    # 【完了条件】
    #   - scripts/check_env_wrapper.py の step ループが通る
    # ═════════════════════════════════════════════════════════════════
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action.astype(np.float32), -1, 1)
        total_reward = 0.0
        for _ in range(self.action_repeat):
            time_step = self._env.step(action)
            total_reward += time_step.reward or 0.0
            if time_step.last():
                break
        obs = self._render()
        return obs, total_reward, time_step.last(), {}
