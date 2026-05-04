"""Actor-Critic + 想像内学習 smoke test (Phase 6).

実行:
    uv run python scripts/smoke_actor_critic.py

何をするか:
    1. 5 ランダムエピソードで WM を 500 step 事前学習 (Phase 4 と同じ)
    2. ActorCritic を作って AC のみを 500 step 想像内学習
       (WM は更新しない)
    3. 学習中に actor の log_prob, critic value, return を観察

期待挙動:
    - critic loss が下がる (V が return を予測する)
    - return / value が学習で上がる傾向 (random policy より良い)
    - actor の動きで learn しているか視覚確認

期待出力:
    pretraining WM 500 steps...
    Step    0 | recon=350.0 | kl=12.0
    ...
    training Actor-Critic in imagination 500 steps...
    Step    0 | actor=-2.5 | critic=8.0 | return=2.5 | value=0.1
    Step  100 | actor=-3.2 | critic=2.0 | return=3.2 | value=2.8
    ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from dreamer.env import DMCEnv
from dreamer.replay import ReplayBuffer
from dreamer.world_model import WorldModel
from dreamer.actor_critic import ActorCritic


def collect_random_episode(env: DMCEnv, max_steps: int = 1000) -> dict:
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


def to_tensor_batch(np_batch: dict, device: str) -> dict:
    return {k: torch.from_numpy(v).to(device) for k, v in np_batch.items()}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # 1. 経験収集
    print("collecting 5 random episodes...")
    env = DMCEnv("walker", "walk", seed=0)
    buf = ReplayBuffer(
        capacity=100_000,
        obs_shape=env.obs_shape,
        action_dim=env.action_dim,
    )
    for i in range(5):
        ep = collect_random_episode(env)
        buf.add_episode(ep)

    # 2. WM 事前学習 (500 step)
    wm = WorldModel(action_dim=env.action_dim).to(device)
    wm_opt = torch.optim.Adam(wm.parameters(), lr=6e-4)

    print("\npretraining WM 500 steps...")
    for step in range(500):
        np_batch = buf.sample(batch_size=16, seq_len=20)
        batch = to_tensor_batch(np_batch, device)
        metrics, _ = wm.update(
            batch, wm_opt, free_nats=3.0, kl_scale=1.0, grad_clip=100.0
        )
        if step % 100 == 0:
            print(
                f"  Step {step:>4d} | "
                f"recon={metrics['wm/recon']:>7.2f} | "
                f"kl={metrics['wm/kl']:>5.2f}"
            )

    # 3. ActorCritic 学習 (WM 固定)
    ac = ActorCritic(
        feat_dim=wm.rssm.feat_dim,
        action_dim=env.action_dim,
        num_units=400,
        num_layers=3,
    ).to(device)
    actor_opt = torch.optim.Adam(ac.actor.parameters(), lr=8e-5)
    critic_opt = torch.optim.Adam(ac.critic.parameters(), lr=8e-5)

    print("\ntraining Actor-Critic in imagination 500 steps...")
    logs = {"actor": [], "critic": [], "return": [], "value": [], "reward": []}

    # WM は fixed
    wm.eval()

    for step in range(500):
        np_batch = buf.sample(batch_size=16, seq_len=20)
        batch = to_tensor_batch(np_batch, device)

        # WM observe (no grad — gradient はここから先に流さない)
        with torch.no_grad():
            T = batch["action"].shape[1]
            obs_seq = batch["obs"][:, :T]
            _, posts, _ = wm.observe(obs_seq, batch["action"])

        metrics = ac.update(
            wm, posts, actor_opt, critic_opt,
            horizon=15, gamma=0.99, lambda_=0.95, grad_clip=100.0
        )

        for k in ("actor", "critic", "return", "value", "reward"):
            logs[k].append(metrics[f"ac/{k}"])

        if step % 50 == 0:
            print(
                f"  Step {step:>4d} | "
                f"actor={metrics['ac/actor']:>7.3f} | "
                f"critic={metrics['ac/critic']:>6.3f} | "
                f"return={metrics['ac/return']:>6.3f} | "
                f"value={metrics['ac/value']:>6.3f} | "
                f"reward={metrics['ac/reward']:>5.3f}"
            )

    # 4. 学習曲線
    Path("results").mkdir(exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, key in zip(axes.flat, ["actor", "critic", "return", "value", "reward"]):
            ax.plot(logs[key])
            ax.set_title(f"AC {key}")
            ax.set_xlabel("step")
        axes.flat[-1].set_visible(False)
        plt.tight_layout()
        plt.savefig("results/ac_smoke.png", dpi=100)
        print("\nSaved curve to: results/ac_smoke.png")
    except ImportError:
        print("matplotlib not installed; skipping plot.")

    # 5. 数値サマリ
    print(f"\n=== summary ===")
    print(f"actor:  {logs['actor'][0]:>7.3f} → {logs['actor'][-1]:>7.3f}")
    print(f"critic: {logs['critic'][0]:>7.3f} → {logs['critic'][-1]:>7.3f}")
    print(f"return: {logs['return'][0]:>7.3f} → {logs['return'][-1]:>7.3f}")
    print(f"value:  {logs['value'][0]:>7.3f} → {logs['value'][-1]:>7.3f}")
    print(f"reward: {logs['reward'][0]:>7.3f} → {logs['reward'][-1]:>7.3f}")

    # critic loss が下がっていれば成功
    if logs["critic"][-1] < logs["critic"][0]:
        print("OK")
    else:
        print("WARNING: critic loss did not decrease.")


if __name__ == "__main__":
    main()
