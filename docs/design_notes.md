# Dreamer v1 再現実装: 設計ノート

**目的**: Hafner et al. (2020) "Dream to Control" を**理解優先**で再現する。

参考実装:
- 公式 (TensorFlow): https://github.com/danijar/dreamer
- PyTorch: https://github.com/juliusfrost/dreamer-pytorch
- PyTorch v3: https://github.com/NM512/dreamerv3-torch (v3 だが構造参考)

論文: Hafner et al. (2020) "Dream to Control: Learning Behaviors by Latent Imagination"
arXiv:1912.01603

---

## 0. 学習方針 (WRITE ME 方式)

このリポジトリは「**公式実装を参考にしつつ、自分で書きながら理解する**」ことが目的。

各ファイル内に以下の形式の WRITE ME ブロックを配置する:

```python
# ═══════════════════════════════════════════════════════════════════
# 📝 WRITE ME [#X.Y.Z]: 関数名
# ───────────────────────────────────────────────────────────────────
# 【目的】1 行で何をする箇所か
# 【やること】手順
# 【ヒント】数式、形状、公式実装の対応箇所
# 【完了条件】test_xxx.py の test_yyy を通す
# ═══════════════════════════════════════════════════════════════════
```

ID 体系: `#Phase.File.Block` (例: `#1.2.3` = Phase 1 / 2nd file / 3rd WRITE ME)。

私 (Claude) が書く範囲:
- import, クラス外形, dataclass, enum
- 完成形の test
- 簡単な動作確認スクリプト
- 公式から自明にコピーすべきボイラープレート

あなたが書く範囲 (WRITE ME):
- 概念の理解が要る箇所 (RSSM, KL, λ-return, etc.)
- 形状操作で間違えやすい箇所
- 数式を直接実装する箇所

---

## 1. 設計思想

3 原則:
1. **責務分離**: World Model と Actor-Critic は独立。`(h, s)` の latent state でのみ会話。
2. **形状の一貫性**: 全テンソルは `(B, T, ...)` 形式。
3. **論文 ↔ コード対応**: Algorithm 1 の各行 ↔ コードの行。

---

## 2. ファイル構成 (最終形)

```
wm-dreamer-repro/
├── docs/
│   └── design_notes.md          # 本ファイル
├── configs/
│   └── dmc_walker.yaml          # ハイパラ
├── src/dreamer/
│   ├── env.py                   # [Phase 1] DMControl ラッパー
│   ├── replay.py                # [Phase 1] Replay Buffer
│   ├── encoder.py               # [Phase 2] CNN encoder
│   ├── decoder.py               # [Phase 2] ConvT decoder
│   ├── rssm.py                  # [Phase 3] RSSM (★中心)
│   ├── world_model.py           # [Phase 4] 統合 + L_wm
│   ├── actor_critic.py          # [Phase 6] Actor + Critic + λ-return
│   ├── agent.py                 # [Phase 7] 高レベル API
│   ├── train.py                 # [Phase 7] エントリポイント
│   └── utils.py                 # 共通ユーティリティ
├── scripts/
│   ├── check_env.py             # [Phase 0]
│   ├── check_pixels.py          # [Phase 0]
│   ├── check_env_wrapper.py     # [Phase 1]
│   └── collect_random.py        # [Phase 1]
├── tests/
│   └── test_*.py                # phase ごと
└── results/
    ├── curves/                  # 学習曲線
    └── videos/                  # 想像 rollout
```

---

## 3. 主要クラスの依存関係

```
                  Agent (orchestration)
                  /        \
         WorldModel        ActorCritic
        /    |    \              |
   Encoder  RSSM  Decoder    uses (h, s)
                  RewardHead
```

ポイント:
- `ActorCritic` は `WorldModel` 全体を知らない。`rssm` と初期 `(h, s)` だけ受け取る
- `Agent` は薄い orchestrator

---

## 4. テンソル形状の規約 (★最重要)

| 名前 | 意味 | 形状 | dtype |
|---|---|---|---|
| B | バッチ | scalar | - |
| T | 時刻 | scalar | - |
| H | 決定的隠れ次元 (RSSM GRU) | 200 | - |
| Z | 確率的潜在次元 | 30 | - |
| A | アクション次元 (walker=6) | - | - |
| E | 画像 embedding 次元 | 1024 | - |
| `obs` | 画像観測 | (B,T,3,64,64) | float32 |
| `action` | 行動 | (B,T,A) | float32 |
| `reward` | 報酬 | (B,T) | float32 |
| `discount` | 割引 (0.0 or 1.0) | (B,T) | float32 |
| `embed` | 画像 embedding | (B,T,E) | float32 |
| `h` | 決定的状態 | (B,T,H) | float32 |
| `s` | 確率的潜在 | (B,T,Z) | float32 |
| `prior_mean/std` | prior の Gaussian params | (B,T,Z) | float32 |
| `post_mean/std` | posterior の Gaussian params | (B,T,Z) | float32 |

規約:
- バッチ次元は常に最初
- 時間次元は常に第 2 次元
- 画像は CHW (PyTorch 流)。HWC は env から出る瞬間のみ
- `reward[t]` は `obs[t] → obs[t+1]` の遷移で得た報酬

---

## 5. データフロー: 1 学習ステップ

```
[A] 経験収集:
    env.step を回し、buffer.add_episode(...)

[B] バッチサンプル:
    buffer.sample(B=50, T=50) → dict of arrays

[C] World Model 学習:
    1. embed = encoder(obs)
    2. RSSM 観測フェーズ (posterior 使用):
         h_t = GRU(s_{t-1}, action_{t-1}, h_{t-1})
         prior_t = NN_prior(h_t)
         post_t  = NN_post(h_t, embed_t)
         s_t ~ post_t (reparam)
    3. 予測:
         x̂_t = decoder(h_t, s_t)
         r̂_t = reward_head(h_t, s_t)
    4. 損失:
         L_recon  = -log p(x_t | h_t, s_t)
         L_reward = -log p(r_t | h_t, s_t)
         L_kl     = KL(post_t || prior_t)  (free bits 適用)
         L_wm = L_recon + L_reward + β·L_kl

[D] 想像 rollout (Actor-Critic):
    1. (h, s) を flatten して (B*T, ...) 個の開始点
    2. H_imag=15 step 想像:
         a_τ ~ actor(h_τ, s_τ)  (reparam)
         h_{τ+1} = GRU(s_τ, a_τ, h_τ)
         s_{τ+1} ~ prior(h_{τ+1})  (no obs)
         r̂_τ = reward_head(...)
         V_τ = critic(...)
    3. λ-return: V^λ_τ = r̂_τ + γ ((1-λ) V_{τ+1} + λ V^λ_{τ+1})
    4. 損失:
         L_actor  = -mean(V^λ)  (reparam で a を通す)
         L_critic = mean((V - sg(V^λ))²)
```

---

## 6. 3 つの損失関数の所在

| 損失 | 数式 | 場所 | 更新対象 |
|---|---|---|---|
| L_wm | -log p(x|h,s) - log p(r|h,s) + β·KL(q||p) | `WorldModel.loss_wm()` | encoder, rssm, decoder, reward_head |
| L_actor | -E[V^λ] | `ActorCritic.loss_actor()` | actor のみ |
| L_critic | E[(V - sg(V^λ))²] | `ActorCritic.loss_critic()` | critic のみ |

重要:
- WM と AC は別オプティマイザ
- AC 学習時は WM 出力を `.detach()` する

---

## 7. ハイパラ (DMC walker_walk)

`configs/dmc_walker.yaml`:

```yaml
env:
  domain: walker
  task: walk
  action_repeat: 2
  image_size: 64

replay:
  capacity: 1_000_000
  batch_size: 50
  seq_len: 50

model:
  encoder_hidden: 1024
  rssm_hidden: 200
  rssm_stochastic: 30
  reward_layers: [400, 400, 400]
  actor_layers: [400, 400, 400]
  critic_layers: [400, 400, 400]
  min_std: 0.1

train:
  total_steps: 1_000_000
  prefill_episodes: 5
  collect_every: 1000
  log_every: 100
  eval_every: 10000
  save_every: 50000

optim:
  wm_lr: 6e-4
  actor_lr: 8e-5
  critic_lr: 8e-5
  grad_clip: 100.0
  kl_beta: 1.0
  kl_free_nats: 3.0

imagine:
  horizon: 15
  gamma: 0.99
  lambda_: 0.95
```

---

## 8. Phase ごとの進め方

| Phase | 触るファイル | 完了条件 |
|---|---|---|
| 0 | (環境構築) | DMC が動く |
| 1 | `env.py`, `replay.py` | random 5 episode を buffer に格納できる |
| 2 | `encoder.py`, `decoder.py` | random batch で再構成 loss が下がる |
| 3 | `rssm.py` | 1-step 予測の KL が下がる |
| 4 | `world_model.py` | reward が真値と相関 |
| 5 | `world_model.py` (imagine 追加) | 想像軌跡が暴走しない |
| 6 | `actor_critic.py` | 想像内で V が増える |
| 7 | `agent.py`, `train.py` | DMC walker で return が上がる |

各 phase 冒頭で「設計書のどこを実装するか」を再確認。

---

## 9. 公式 vs 本実装の差分

意図的に省略する点 (学習目的):
- Mixed precision training
- Distributed training
- TensorBoard (wandb のみ)
- Atari (DMC のみ)
- 細かい正則化 (weight decay 等)
- pcont (continuation/discount predictor): DMC では γ 固定でほぼ等価。Phase 7 で必要なら追加

学ぶ目的では上記は不要。鈴木先生コンタクト後の HCAI-WM 本実装で必要に応じて追加。

---

## 10. Verified Architecture (公式コード照合済み)

照合日: 2026-05
出典: https://github.com/danijar/dreamer (`models.py`, `dreamer.py`)

### 10.1 ConvEncoder

```
入力 obs: (B, T, 3, 64, 64) float32 in [-0.5, 0.5]
↓ flatten time: (B*T, 3, 64, 64)
↓ Conv2d(3, 32, kernel=4, stride=2, valid) + ReLU      → (B*T, 32, 31, 31)
↓ Conv2d(32, 64, kernel=4, stride=2, valid) + ReLU     → (B*T, 64, 14, 14)
↓ Conv2d(64, 128, kernel=4, stride=2, valid) + ReLU    → (B*T, 128, 6, 6)
↓ Conv2d(128, 256, kernel=4, stride=2, valid) + ReLU   → (B*T, 256, 2, 2)
↓ Flatten                                              → (B*T, 1024)
↓ unflatten time                                       → (B, T, 1024)
```

- channels = `[1, 2, 4, 8] * cnn_depth`, `cnn_depth = 32`
- padding = `valid` (無し). PyTorch では `padding=0`
- 活性化: **ReLU**
- 最終次元 E = 1024 (= 2*2*256)

### 10.2 ConvDecoder

```
入力 feat: (B, T, H+Z) = (B, T, 230)
↓ Linear(230, 1024)                                    → (B*T, 1024)
↓ Reshape                                              → (B*T, 1024, 1, 1)
↓ ConvT2d(1024, 128, kernel=5, stride=2) + ReLU        → (B*T, 128, 5, 5)
↓ ConvT2d(128, 64, kernel=5, stride=2) + ReLU          → (B*T, 64, 13, 13)
↓ ConvT2d(64, 32, kernel=6, stride=2) + ReLU           → (B*T, 32, 30, 30)
↓ ConvT2d(32, 3, kernel=6, stride=2) (活性化なし)      → (B*T, 3, 64, 64)
```

- カーネルサイズが **5,5,6,6 と非対称**. 公式と必ず合わせる
- 出力分布: Normal(mean=output, std=1) → log_prob は `-0.5 * MSE + const`
  - **実装は `mse_loss(reduction='sum') / 2 + const` でも等価**

### 10.3 RSSM

#### 状態
- `deter` (h): GRU hidden, dim=**200**
- `stoch` (s): Gaussian latent, dim=**30**

#### 1 ステップ (img_step, prior 計算)

```
入力: prev_stoch s_{t-1} (Z), prev_action a_{t-1} (A), prev_deter h_{t-1} (H)

x = concat(s_{t-1}, a_{t-1})                  # (Z+A,)
x = Dense(200, ELU)(x)                         # (200,)
h_t = GRUCell(input=x, hidden=h_{t-1})         # (H=200,)

# Prior:
p = Dense(200, ELU)(h_t)
p = Dense(2*Z=60, no act)(p)
prior_mean, prior_std_raw = split(p)
prior_std = softplus(prior_std_raw) + 0.1

prior_dist = Normal(prior_mean, prior_std)
s_t_prior = prior_dist.rsample()
```

#### Posterior 計算 (obs_step, embed 入力時)

```
入力: h_t (上で計算済み), embed_t (E=1024,)

q = concat(h_t, embed_t)                       # (200+1024 = 1224,)
q = Dense(200, ELU)(q)
q = Dense(2*Z=60, no act)(q)
post_mean, post_std_raw = split(q)
post_std = softplus(post_std_raw) + 0.1

post_dist = Normal(post_mean, post_std)
s_t = post_dist.rsample()  # ← WM 学習はこちらを使う
```

#### Forward (時刻 0..T-1 の系列)

- 学習時: 観測あり → `obs_step` で posterior を計算し、`s_t` をサンプル
- 想像時: 観測なし → `img_step` で prior を計算し、`s_t` をサンプル

### 10.4 Reward Head (DenseDecoder)

```
入力 feat: concat(h, s) = (B, T, 230)
↓ Dense(400, ELU) × 3
↓ Dense(1) (no act)
↓ Normal(mean=output, std=1)
```

学習: `-log_prob(reward)` ≒ MSE/2 + const

### 10.5 Critic / Value (DenseDecoder)

Reward と同じ MLP 構造 (出力は scalar の Normal)
- 入力 feat (B, T, 230)
- Dense(400, ELU) × 3 → Dense(1) → Normal(mean, std=1)
- 学習: `-log_prob(returns)`

### 10.6 Actor

```
入力 feat: concat(h, s) = (B, T, 230)
↓ Dense(400, ELU) × 3
↓ Dense(2*A) (no act)
↓ split → mean_raw, std_raw

# 公式の TanhNormal パラメトリゼーション:
mean = 5 * tanh(mean_raw / 5)                                 # (-5, 5) に soft-clip
std = softplus(std_raw + log(exp(5) - 1)) + 1e-4              # 初期 std ≈ 5

# サンプリング:
ε ~ Normal(mean, std)
action = tanh(ε)                                              # [-1, 1]
```

- ポイント:
  - `mean = 5*tanh(m/5)`: 大きな絶対値を抑制 (gradient stability)
  - `std + log(exp(5)-1)`: softplus の引数を底上げ → 初期 std=5 で**強い探索**
  - 最後に `tanh` で連続行動を [-1, 1] に
- 推論時 (`.act`) は `tanh(mean)` をそのまま使う or サンプル + Normal(0, 0.3) ノイズ

### 10.7 World Model 損失 (verified)

```python
# 公式 dreamer.py 抜粋に対応:
likes_image  = image_pred.log_prob(obs).mean()       # Normal(μ, σ=1)
likes_reward = reward_pred.log_prob(reward).mean()
kl = KL(post || prior).mean()
kl = max(kl, free_nats=3.0)                          # free nats clamp

L_wm = kl_scale * kl - (likes_image + likes_reward)  # = β*KL - log p
       # ↑ minimize
```

数学的には ELBO の負号:
```
L_wm = β · max(KL(q||p), free_nats) - E[log p(x|h,s)] - E[log p(r|h,s)]
```

### 10.8 Actor / Critic 損失 (verified)

想像 rollout (length H=15) で:
```
returns = lambda_return(reward[:-1], value[:-1], pcont[:-1],
                        bootstrap=value[-1], lambda=0.95)

L_actor  = - mean(discount * returns)                      # reparam で actor に勾配
L_critic = - mean(discount * value_pred.log_prob(stop_grad(returns)))
                                                           # ≒ mean(discount * 0.5*(value-returns)^2)
```

`discount` は `cumprod(γ * pcont)`. DMC では pcont=1 とみなして可。

### 10.9 学習スケジュール

- **prefill**: env から 5 random episode 集める
- **pretrain**: 100 step だけ純粋に学習 (env と無関係)
- **メインループ**:
    - env で 1 episode collect
    - その後 100 train_step (= WM update + AC update を 100 回)
    - これを繰り返す

→ env step と learn step の比は **約 1:100** (DMC walker の場合)

### 10.10 探索

`actor.act(obs)` の出力に **Normal(0, 0.3)** ノイズを加える (`expl_amount=0.3`).
評価時 (`eval`) はノイズなしで `tanh(mean)` を使う.

### 10.11 損失計算で実質 MSE になる箇所まとめ

| 損失 | 公式 | 実装 (等価) |
|---|---|---|
| 画像再構成 | `-Normal(μ, 1).log_prob(x)` | `0.5 * MSE(x̂, x)` (定数省略) |
| 報酬予測 | `-Normal(μ, 1).log_prob(r)` | `0.5 * MSE(r̂, r)` |
| 価値関数学習 | `-Normal(μ, 1).log_prob(returns)` | `0.5 * MSE(V, returns)` |

→ Phase 4-6 では **MSE で実装して全く問題なし**. 公式のように `Normal(mean, 1).log_prob(...)` を使ってもよい. 数値的には等価.
