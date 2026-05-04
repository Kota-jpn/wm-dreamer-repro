"""Phase 7 統合学習ループの短縮版 smoke test.

実行:
    uv run python scripts/smoke_train_short.py

何をするか:
    本番の dreamer/train.py と同じロジックを **大幅に短縮** して
    全部つながっているかだけ確認する.

期待出力:
    prefill ...
    pretrain 20 steps...
    main loop start (target: env_step 5000)
    [env_step=  500] return=15.5 | wm/total=120.0 wm/recon=85.0 ...
    ...
    [env_step= 5000] return=22.3 | ...
    eval mean return: 25.4
    OK

時間: CPU で 10-15 分, GPU で 3-5 分.
学習が進むかは保証しない (5000 env_step では足りない). あくまで「動く」確認.

本番学習は configs/dmc_walker.yaml + dreamer/train.py で.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from dreamer.env import DMCEnv
from dreamer.replay import ReplayBuffer
from dreamer.agent import Agent
from dreamer.train import (
    collect_random_episode,
    collect_with_agent,
    to_tensor_batch,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ─── 短縮版設定 ───
    PREFILL_EPISODES = 2     # 公式 5
    PRETRAIN = 20            # 公式 100
    TOTAL_ENV_STEPS = 5_000  # 公式 1_000_000
    TRAIN_STEPS_PER_COLLECT = 20  # 公式 100
    BATCH_SIZE = 16          # 公式 50
    SEQ_LEN = 20             # 公式 50
    LOG_EVERY = 5            # train_step ベース
    EVAL_EVERY = 2_000       # env_step ベース

    # ─── 環境とバッファ ───
    env = DMCEnv("walker", "walk", action_repeat=2, seed=0)
    eval_env = DMCEnv("walker", "walk", action_repeat=2, seed=42)
    buf = ReplayBuffer(
        capacity=100_000,
        obs_shape=env.obs_shape,
        action_dim=env.action_dim,
    )

    # ─── Agent ───
    agent = Agent(
        action_dim=env.action_dim,
        cnn_depth=32,
        rssm_deter=200,
        rssm_stoch=30,
        rssm_hidden=200,
        min_std=0.1,
        num_units=400,
        num_layers=3,
        wm_lr=6e-4,
        actor_lr=8e-5,
        critic_lr=8e-5,
        kl_free_nats=3.0,
        kl_scale=1.0,
        grad_clip=100.0,
        imagine_horizon=15,
        gamma=0.99,
        lambda_=0.95,
        expl_noise=0.3,
        device=device,
    )

    # ─── A. Prefill ───
    print("\n--- prefill ---")
    for i in range(PREFILL_EPISODES):
        ep = collect_random_episode(env)
        buf.add_episode(ep)
        print(f"  episode {i}: length={len(ep['action'])}, "
              f"return={ep['reward'].sum():.2f}")
    print(f"  buffer: {len(buf)} steps")

    # ─── B. Pretrain ───
    print(f"\n--- pretrain {PRETRAIN} steps ---")
    for step in range(PRETRAIN):
        np_batch = buf.sample(batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
        batch = to_tensor_batch(np_batch, device)
        metrics = agent.learn(batch)
        if step % 5 == 0:
            print(f"  pretrain {step:>3d}: "
                  f"wm/total={metrics['wm/total']:>6.2f} "
                  f"recon={metrics['wm/recon']:>6.2f} "
                  f"kl={metrics['wm/kl']:>4.2f}")

    # ─── C. Main loop ───
    print(f"\n--- main loop (target env_step={TOTAL_ENV_STEPS}) ---")
    env_step = 0
    train_step = 0
    last_t = time.time()
    last_eval_step = -EVAL_EVERY  # 最初の eval を即実行

    while env_step < TOTAL_ENV_STEPS:
        # 1. collect
        ep = collect_with_agent(env, agent, training=True, max_steps=500)
        buf.add_episode(ep)
        env_step += len(ep["action"]) * 2  # action_repeat=2
        ep_return = float(ep["reward"].sum())

        # 2. train
        for _ in range(TRAIN_STEPS_PER_COLLECT):
            np_batch = buf.sample(batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
            batch = to_tensor_batch(np_batch, device)
            metrics = agent.learn(batch)
            train_step += 1

        # 3. eval
        if env_step - last_eval_step >= EVAL_EVERY:
            eval_return = agent.evaluate(eval_env, n_episodes=2, max_steps=500)
            print(f"  [env_step={env_step:>5d}] eval/return = {eval_return:.2f}")
            last_eval_step = env_step

        # 4. log
        elapsed = time.time() - last_t
        last_t = time.time()
        print(
            f"  [env_step={env_step:>5d} train_step={train_step:>4d}] "
            f"return={ep_return:>5.2f} | "
            f"wm/total={metrics['wm/total']:>5.2f} "
            f"recon={metrics['wm/recon']:>5.2f} "
            f"kl={metrics['wm/kl']:>4.2f} | "
            f"actor={metrics['ac/actor']:>5.2f} "
            f"critic={metrics['ac/critic']:>4.2f} | "
            f"{elapsed:>4.1f}s"
        )

    # 最終 eval
    print("\n--- final evaluation ---")
    final_return = agent.evaluate(eval_env, n_episodes=3, max_steps=500)
    print(f"final eval mean return: {final_return:.2f}")
    print("OK")


if __name__ == "__main__":
    main()
