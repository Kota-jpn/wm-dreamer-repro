"""統合学習ループ. DMC で実際に学習する.

Phase 7 / File 2.

設計書: docs/design_notes.md §10.9 学習スケジュール
出典:   公式 https://github.com/danijar/dreamer/blob/master/dreamer.py

─── 全体スケジュール ────────────────────────────────────────────────────
1. Prefill:    random policy で N episode 集める (buffer 初期化)
2. Pretrain:   buffer のみで M step 学習 (env interaction なし)
3. Main loop:
     while step < total_steps:
         collect 1 episode with agent (training=True)
         for _ in range(train_steps):
             batch = buffer.sample
             agent.learn(batch)
         if step % eval_every == 0: evaluate
         if step % log_every  == 0: log
         if step % save_every == 0: save

─── 使い方 ──────────────────────────────────────────────────────────────
ローカル / Colab どちらでも:

    uv run python -m dreamer.train --config configs/dmc_walker.yaml

または notebook で:

    from dreamer.train import main
    main("configs/dmc_walker.yaml")
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .env import DMCEnv
from .replay import ReplayBuffer
from .agent import Agent


# ─────────────────────────────────────────────────────────────────────────
# Episode 収集ヘルパ (Claude 提供)
# ─────────────────────────────────────────────────────────────────────────


def collect_random_episode(env: DMCEnv, max_steps: int = 1000) -> dict:
    """Random policy で 1 episode 走らせて episode dict を返す.
    Phase 1 と同じロジック.
    """
    obs_list, action_list, reward_list, discount_list = [], [], [], []
    obs = env.reset()
    obs_list.append(obs)
    for _ in range(max_steps):
        a = np.random.uniform(-1, 1, env.action_dim).astype(np.float32)
        obs, r, done, _ = env.step(a)
        obs_list.append(obs)
        action_list.append(a)
        reward_list.append(r)
        discount_list.append(0.0 if done else 1.0)
        if done:
            break
    return {
        "obs": np.stack(obs_list).astype(np.uint8),
        "action": np.stack(action_list).astype(np.float32),
        "reward": np.array(reward_list, dtype=np.float32),
        "discount": np.array(discount_list, dtype=np.float32),
    }


def collect_with_agent(
    env: DMCEnv,
    agent: Agent,
    training: bool = True,
    max_steps: int = 1000,
) -> dict:
    """Agent で 1 episode 走らせて episode dict を返す."""
    obs_list, action_list, reward_list, discount_list = [], [], [], []
    agent.reset()
    obs = env.reset()
    obs_list.append(obs)
    for _ in range(max_steps):
        action = agent.act(obs, training=training)  # numpy (A,)
        obs, r, done, _ = env.step(action)
        obs_list.append(obs)
        action_list.append(action.astype(np.float32))
        reward_list.append(r)
        discount_list.append(0.0 if done else 1.0)
        if done:
            break
    return {
        "obs": np.stack(obs_list).astype(np.uint8),
        "action": np.stack(action_list).astype(np.float32),
        "reward": np.array(reward_list, dtype=np.float32),
        "discount": np.array(discount_list, dtype=np.float32),
    }


def to_tensor_batch(np_batch: dict, device: str) -> dict:
    """numpy batch を torch tensor batch に."""
    return {k: torch.from_numpy(v).to(device) for k, v in np_batch.items()}


# ─────────────────────────────────────────────────────────────────────────
# main loop
# ─────────────────────────────────────────────────────────────────────────


def main(config_path: str, use_wandb: bool = False, run_name: str = None) -> None:
    """全学習ループのエントリポイント.

    Args:
        config_path: yaml ハイパラファイルのパス.
        use_wandb: True なら wandb.log でログ. False なら stdout のみ.
        run_name: wandb run 名.
    """
    cfg = yaml.safe_load(open(config_path))
    print(f"loaded config: {config_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # wandb (任意)
    if use_wandb:
        import wandb
        wandb.init(project="dreamer-repro", config=cfg, name=run_name)

    # ─── 環境とバッファ ───
    env = DMCEnv(
        domain=cfg["env"]["domain"],
        task=cfg["env"]["task"],
        action_repeat=cfg["env"]["action_repeat"],
        image_size=cfg["env"]["image_size"],
        seed=0,
    )
    eval_env = DMCEnv(
        domain=cfg["env"]["domain"],
        task=cfg["env"]["task"],
        action_repeat=cfg["env"]["action_repeat"],
        image_size=cfg["env"]["image_size"],
        seed=42,  # eval は別 seed
    )
    buf = ReplayBuffer(
        capacity=cfg["replay"]["capacity"],
        obs_shape=env.obs_shape,
        action_dim=env.action_dim,
    )

    # ─── Agent ───
    agent = Agent(
        action_dim=env.action_dim,
        cnn_depth=cfg["model"]["cnn_depth"],
        rssm_deter=cfg["model"]["rssm_deter"],
        rssm_stoch=cfg["model"]["rssm_stoch"],
        rssm_hidden=cfg["model"]["rssm_hidden"],
        min_std=cfg["model"]["min_std"],
        num_units=cfg["model"]["num_units"],
        num_layers=cfg["model"]["num_layers"],
        wm_lr=cfg["optim"]["wm_lr"],
        actor_lr=cfg["optim"]["actor_lr"],
        critic_lr=cfg["optim"]["critic_lr"],
        kl_free_nats=cfg["optim"]["kl_free_nats"],
        kl_scale=cfg["optim"]["kl_beta"],
        grad_clip=cfg["optim"]["grad_clip"],
        imagine_horizon=cfg["imagine"]["horizon"],
        gamma=cfg["imagine"]["gamma"],
        lambda_=cfg["imagine"]["lambda_"],
        expl_noise=cfg["explore"]["noise_amount"],
        device=device,
    )
    print(f"agent params: {sum(p.numel() for p in agent.wm.parameters()) // 1000}k WM, "
          f"{sum(p.numel() for p in agent.ac.actor.parameters()) // 1000}k actor, "
          f"{sum(p.numel() for p in agent.ac.critic.parameters()) // 1000}k critic")

    # ─── 学習ループ ───
    train(cfg, env, eval_env, buf, agent, device, use_wandb)


# ═════════════════════════════════════════════════════════════════════════
# 📝 WRITE ME [#7.2.1]: train (メインループ)
# ─────────────────────────────────────────────────────────────────────────
# 【目的】 prefill → pretrain → main loop の学習を回す.
#
# 【全体構造】 (擬似コード)
#
 
#
# 【ヒント】
#   - env_step: 環境との相互作用ステップ数 (= action_repeat * actions). cfg.total_steps と比較.
#   - train_step: 学習ステップ数 (= optimizer.step() を呼んだ回数).
#   - 公式は「1 episode collect → 100 train」で env:train ≈ 1:100 の比率. 上記もそう.
#   - eval_every / save_every は env_step ベース (公式と同じ).
#   - log_every は train_step ベース (頻度を細かく).
#
# 【完了条件】
#   - 短い設定 (total_steps=10000, prefill=1, pretrain=5) で 1 周回る
#   - 数千 step 後に eval/return が random (~10) より明らかに上がっている
#
# 【Colab での実行例】
#   !cd /content/wm-dreamer-repro && uv run python -m dreamer.train \\
#       --config configs/dmc_walker.yaml --wandb
# ═════════════════════════════════════════════════════════════════════════
def train(cfg, env, eval_env, buf, agent, device, use_wandb):
      # ----- A. Prefill: random policy で N episode -----
      print("prefill...")
      for _ in range(cfg['train']['prefill_episodes']):
          ep = collect_random_episode(env)
          buf.add_episode(ep)
      print(f"  buffer: {len(buf)} steps")

      # ----- B. Pretrain: buffer のみで M step 学習 -----
      print(f"pretrain {cfg['train']['pretrain']} steps...")
      for step in range(cfg['train']['pretrain']):
          batch = to_tensor_batch(
              buf.sample(cfg['replay']['batch_size'], cfg['replay']['seq_len']),
              device
          )
          metrics = agent.learn(batch)
          if step % 50 == 0:
              print(f"  pretrain step {step}: total={metrics['wm/total']:.2f} "
                    f"recon={metrics['wm/recon']:.2f} kl={metrics['wm/kl']:.2f}")

      # ----- C. Main loop: collect + train + eval を繰り返す -----
      env_step = 0
      train_step = 0
      last_log_time = time.time()
      while env_step < cfg['train']['total_steps']:

          # 1. collect 1 episode (training=True)
          ep = collect_with_agent(env, agent, training=True,
                                   max_steps=1000 // cfg['env']['action_repeat'])
          buf.add_episode(ep)
          env_step += len(ep['action']) * cfg['env']['action_repeat']

          ep_return = float(ep['reward'].sum())

          # 2. train_steps 回学習
          recent_metrics = {}
          for _ in range(cfg['train']['train_steps']):
              batch = to_tensor_batch(
                  buf.sample(cfg['replay']['batch_size'], cfg['replay']['seq_len']),
                  device
              )
              recent_metrics = agent.learn(batch)
              train_step += 1

          # 3. eval (eval_every step ごと)
          if env_step % cfg['train']['eval_every'] < 1000:
              eval_return = agent.evaluate(eval_env, n_episodes=5)
              print(f"  [env_step={env_step}] eval/return = {eval_return:.2f}")
              if use_wandb:
                  import wandb
                  wandb.log({'eval/return': eval_return}, step=env_step)

          # 4. log (log_every train step ごと)
          if train_step % cfg['train']['log_every'] == 0:
              elapsed = time.time() - last_log_time
              last_log_time = time.time()
              print(f"[env_step={env_step:>7d} train_step={train_step:>5d}] "
                    f"return={ep_return:>6.2f} | "
                    f"wm/total={recent_metrics['wm/total']:>5.2f} "
                    f"recon={recent_metrics['wm/recon']:>5.2f} "
                    f"kl={recent_metrics['wm/kl']:>4.2f} | "
                    f"actor={recent_metrics['ac/actor']:>5.2f} "
                    f"critic={recent_metrics['ac/critic']:>4.2f} | "
                    f"{elapsed:.1f}s")
              if use_wandb:
                  import wandb
                  wandb.log({**recent_metrics, 'collect/return': ep_return}, step=env_step)

          # 5. save (save_every env_step ごと)
          if env_step % cfg['train']['save_every'] < 1000:
              ckpt_path = f"checkpoints/agent_{env_step}.pt"
              Path("checkpoints").mkdir(exist_ok=True)
              agent.save(ckpt_path)
              print(f"  saved: {ckpt_path}")

      print("training done.")


# ─────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dmc_walker.yaml")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    main(args.config, use_wandb=args.wandb, run_name=args.name)
