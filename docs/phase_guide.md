# Phase 進行ガイド

**目的**: Dreamer v1 を Phase 0〜7 で段階的に組み立てる際の**手順書**。

このファイルは「次に何をやるか」「どこを書くか」「どこで合格と判断するか」を Phase ごとに詳述する。**設計の WHY は `design_notes.md`**、HOW は本ファイル。

---

## 使い方

1. 各 Phase の冒頭で**ゴールと WRITE ME 一覧**を確認
2. WRITE ME ブロックを 1 つずつ埋める
3. **テスト or check スクリプト**を回し、緑になったら次の WRITE ME へ
4. Phase 末で `git commit` + Phase ノートの**振り返り**を書く

---

## 全 Phase のオーバービュー

| Phase | テーマ | 触るファイル | 推定時間 (集中時) | 状態 |
|---|---|---|---|---|
| 0 | 環境構築 | (セットアップ) | 半日 | ✓ 完了 |
| 1 | データ供給 | `env.py`, `replay.py` | 半日 | 進行中 |
| 2 | 視覚 (Encoder/Decoder) | `encoder.py`, `decoder.py`, `utils.py` 一部 | 1日 | |
| 3 | **RSSM (★中心)** | `rssm.py` | 2-3日 | |
| 4 | World Model 統合 | `world_model.py` | 1日 | |
| 5 | 想像 rollout | `world_model.py` (imagine 追加) | 半日 | |
| 6 | Actor-Critic | `actor_critic.py`, `utils.py` (lambda_return) | 2日 | |
| 7 | 統合学習ループ | `agent.py`, `train.py` | 2-3日 | |

合計: 約 10-12 日 (集中時)。ただし詰まりや確認を入れると **2-3 週間が現実的**。

---

# Phase 0: 環境構築 (✓ 完了)

完了済み。詳細省略。

---

# Phase 1: データ収集 + Replay Buffer

## ゴール
Random policy で 5 episode 集めて Replay Buffer に格納し、`(B=4, T=20)` のチャンクをサンプルできる状態。

## 設計書の対応
`design_notes.md` §4 (テンソル形状), §5 [A]/[B] (データフロー)

## 触るファイル

| ファイル | 種類 |
|---|---|
| `src/dreamer/env.py` | WRITE ME 2 箇所 |
| `src/dreamer/replay.py` | WRITE ME 2 箇所 |
| `scripts/check_env_wrapper.py` | 動作確認 (完成形) |
| `scripts/collect_random.py` | 結合テスト (完成形) |
| `tests/test_replay.py` | pytest (完成形) |

## WRITE ME 一覧

| ID | 場所 | 何を |
|---|---|---|
| `#1.1.1` | `env.py::reset` | DMC reset → 画像 obs (uint8 HWC) を返す |
| `#1.1.2` | `env.py::step` | action_repeat 回ループ + 報酬合算 + done 判定 |
| `#1.2.1` | `replay.py::add_episode` | episode を append, capacity 超過なら FIFO 削除 |
| `#1.2.2` | `replay.py::sample` | チャンク切り出し + uint8→float32[-0.5,0.5] + HWC→CHW |

## テストと合格基準

```bash
# 1. env.py 単体
uv run python scripts/check_env_wrapper.py
# 期待: "OK" + obs (64,64,3), action_dim 6, total reward 数十

# 2. replay.py 単体
uv run pytest tests/test_replay.py -v
# 期待: 5 tests passed

# 3. 結合
uv run python scripts/collect_random.py
# 期待: 5 episodes 収集 + サンプル形状 dict 表示 + "OK"
```

## 落とし穴

1. **DMC の reset 戻り値**: `TimeStep` で reward は None。
   - `time_step.reward or 0.0` で None ガードする習慣を
2. **action のクリップ**: DMC の action_spec は `[-1, 1]` だが、
   モデル出力が範囲外になることがある (Phase 6 で actor から来る)。
   保険として `np.clip(action, -1, 1)` を入れておくと安心
3. **画像変換の順序**: `uint8 [0,255] → float32 / 255.0 → -0.5` の**割り算が先**
4. **HWC → CHW の `transpose`**: 軸の指定を間違えやすい。
   `np.transpose(obs, (0, 1, 4, 2, 3))` で `(B, T, H, W, C) → (B, T, C, H, W)`
5. **capacity 管理**: `len(self) > capacity` の判定は `sum(len(ep['action']))` で. obs の長さ T+1 を数えないこと

## 成功条件

- 全テストが緑
- `collect_random.py` 出力で **batch shapes が想定通り**:
  - `obs: (4, 21, 3, 64, 64)` (`T+1`!)
  - `action: (4, 20, 6)`, `reward: (4, 20)`, `discount: (4, 20)`

## 終了処理

```bash
git add src/dreamer/env.py src/dreamer/replay.py
git add scripts/check_env_wrapper.py scripts/collect_random.py tests/test_replay.py
git commit -m "phase 1: env wrapper + replay buffer"
git push
```

---

# Phase 2: Encoder + Decoder (CNN VAE)

## ゴール
画像 (3,64,64) を 1024 次元に圧縮 → 元に復元できる Encoder/Decoder を作る。
**ランダムバッチで再構成 loss が学習で下がる**ことを確認。

## 設計書の対応
`design_notes.md` §10.1 ConvEncoder, §10.2 ConvDecoder

## 触るファイル

| ファイル | 種類 |
|---|---|
| `src/dreamer/encoder.py` | WRITE ME |
| `src/dreamer/decoder.py` | WRITE ME |
| `src/dreamer/utils.py` | (補助) |
| `tests/test_encoder_decoder.py` | (Claude 提供) |
| `notebooks/02_encoder_decoder_smoke.ipynb` | 学習で loss が下がるか可視化 |

## WRITE ME 一覧 (予定)

| ID | 場所 | 何を |
|---|---|---|
| `#2.1.1` | `encoder.py::ConvEncoder.__init__` | 4 conv layer の Sequential 構築 |
| `#2.1.2` | `encoder.py::ConvEncoder.forward` | (B,T,3,64,64) → (B,T,1024) を `einops.rearrange` で |
| `#2.2.1` | `decoder.py::ConvDecoder.__init__` | Linear + 4 ConvT layer 構築 (kernel 5,5,6,6 注意) |
| `#2.2.2` | `decoder.py::ConvDecoder.forward` | (B,T,F) → (B,T,3,64,64) を返す |

## テストと合格基準

```bash
# 形状テスト
uv run pytest tests/test_encoder_decoder.py -v
# - encode: (B=4, T=10, 3, 64, 64) → (4, 10, 1024)
# - decode: (4, 10, 230) → (4, 10, 3, 64, 64)
# - 完全往復: encoder → decoder で形状が元に戻る

# 学習デモ (Notebook)
# random batch で Adam 1000 step 回し、MSE loss が初期 ~0.1 → ~0.01 まで下がる
```

## 落とし穴

1. **`padding=valid`**: PyTorch では `padding=0` (デフォルト)。`'same'` を指定しないこと
2. **kernel 5,5,6,6 の非対称**: 公式コード通り. ここを 4,4,4,4 にすると 64x64 出力にならない
3. **B*T flatten**: `einops.rearrange(x, 'b t c h w -> (b t) c h w')` で時間軸を畳むのが楽
4. **再構成 loss の単位**: 公式は `Normal(mean, 1).log_prob`. 等価な MSE で OK
   - `0.5 * F.mse_loss(x_hat, x, reduction='none').sum(dim=(-1,-2,-3)).mean()`
   - reduction='mean' で 1 ピクセル平均にすると loss が小さすぎて勾配が薄い

## 成功条件

- 形状テスト全緑
- Notebook で **再構成画像が元画像に類似**することを目視確認 (可視化必須)
- MSE loss が 1000 step で半減以上

---

# Phase 3: RSSM (★中心)

## ゴール
Recurrent State-Space Model: GRU + prior + posterior。
**1-step 予測の KL が学習で下がる**ことを確認。

これが Dreamer の心臓部. **ここを完全に理解することが本研究の最重要マイルストーン**。

## 設計書の対応
`design_notes.md` §10.3 RSSM (3 つの数式ブロック)

## 触るファイル

| ファイル | 種類 |
|---|---|
| `src/dreamer/rssm.py` | WRITE ME 多数 |
| `tests/test_rssm.py` | (Claude 提供) |
| `notebooks/03_rssm_smoke.ipynb` | 1-step 予測 KL の学習曲線 |

## WRITE ME 一覧 (予定)

| ID | 場所 | 何を |
|---|---|---|
| `#3.1.1` | `RSSM.__init__` | GRU cell + prior MLP + posterior MLP の Module 構築 |
| `#3.1.2` | `RSSM.initial_state` | h_0=zeros, s_0=zeros を返す |
| `#3.1.3` | `RSSM.img_step` | 1 ステップ prior path: (h, s, a) → (h_next, prior_mean, prior_std, s_prior) |
| `#3.1.4` | `RSSM.obs_step` | 1 ステップ posterior path: (h, s, a, embed) → (h_next, prior, posterior, s_post) |
| `#3.1.5` | `RSSM.observe` | T ステップ posterior unroll. 系列 (B,T,...) を返す |
| `#3.1.6` | `RSSM.imagine` | T ステップ prior-only unroll (Phase 5 で使う) |
| `#3.1.7` | `RSSM.kl_loss` | KL(post || prior) を free_nats clamp 込みで計算 |

## テストと合格基準

```bash
uv run pytest tests/test_rssm.py -v

# テスト項目:
# - img_step: 形状チェック ((B, H), (B, Z), (B, A)) → 全部正しい形
# - obs_step: posterior の入力に embed が反映される
# - observe: T ステップで (B, T, H), (B, T, Z) 等が出る
# - imagine: 観測なしでも T ステップ走る
# - kl_loss: scalar が出る、勾配が prior と posterior 両方に流れる
# - reparam: rsample の勾配が mean/std に流れる
```

学習デモ (Notebook):
- ランダム軌跡 (5 episode) で 1-step 予測 (posterior 経由) の loss を 1000 step 学習
- KL が初期 ~10 → ~1 まで下がる
- `prior_mean[t+1]` と `posterior_mean[t+1]` が近くなる (相関プロット)

## 落とし穴

1. **`rsample` vs `sample`**: 必ず `rsample()` (reparameterization). `sample()` だと勾配が流れない
2. **GRU の入力形成**: `concat(s_{t-1}, a_{t-1})` を Dense(200, ELU) に通してから GRUCell へ。
   直接 GRU に放り込まない
3. **時間ループは Python for**: PyTorch の GRU layer ではなく **GRUCell** を T 回呼ぶ.
   なぜなら img_step / obs_step を毎ステップ切り替える必要があるため
4. **`softplus(x) + 0.1`**: min_std=0.1 を忘れない. これがないと std=0 で NaN
5. **KL の axis**: `Independent(Normal, 1)` で diag Gaussian を作り `kl_divergence` を呼ぶと最後の次元 (Z=30) を sum してくれる. 自分で sum しない
6. **free_nats のかけ方**: `kl = max(kl, free_nats)` を **scalar 化した後** に行う.
   per-element に max すると勾配が滑らかでなくなる

## 成功条件

- 全テスト緑
- 1-step 予測の KL がランダム軌跡で**学習可能** (Notebook で曲線確認)
- 想像 rollout (15 step) が **NaN にならない**

## 振り返り (このフェーズで自分用に書く)

- prior と posterior の KL の意味を**自分の言葉で書く** (`docs/notes_phase3.md` に)
- なぜ posterior は embed を入力に取り、prior は取らないか?
- なぜ s_t は s_{t-1} から直接遷移するのではなく、**h_t** 経由なのか?
  → これが「Recurrent **State-Space** Model」たる所以

---

# Phase 4: Reward head + World Model 統合

## ゴール
Encoder + RSSM + Decoder + Reward head を `WorldModel` クラスに統合し、
**実際の収集軌跡で WM 全体を学習**できる状態。

## 設計書の対応
`design_notes.md` §5 [C], §10.4, §10.7

## 触るファイル

| ファイル | 種類 |
|---|---|
| `src/dreamer/world_model.py` | WRITE ME 多数 |
| `tests/test_world_model.py` | (Claude 提供) |
| `notebooks/04_wm_training.ipynb` | 収集軌跡で WM 損失が下がる |

## WRITE ME 一覧 (予定)

| ID | 場所 | 何を |
|---|---|---|
| `#4.1.1` | `WorldModel.__init__` | Encoder, RSSM, Decoder, RewardHead をメンバーに |
| `#4.1.2` | `RewardHead.__init__/forward` | Dense(400, ELU) × 3 + Dense(1) |
| `#4.1.3` | `WorldModel.observe` | batch を受けて (h, s, prior, post) を返す |
| `#4.1.4` | `WorldModel.loss` | recon + reward + KL の合算 |
| `#4.1.5` | `WorldModel.update` | optimizer.step() + grad clip |

## テストと合格基準

- WM の forward が走る
- 損失が scalar
- Notebook で **5-10 episode** 集めて WM だけ 1000 step 学習し:
  - recon loss が 0.5 → 0.1 オーダーに低下
  - KL が 5 → 1-3 オーダー
  - reward MSE が大幅に低下
  - **再構成画像が元画像に似る** (visual check)

## 落とし穴

1. **batch の最後の obs**: `obs (B, T+1, ...)` のうち、`embed` を計算するのは `obs[:, :T]` か `obs[:, 1:]` か?
   → 答え: **両方使うやり方と片方だけ使うやり方がある**. 公式は「t=1..T で post を計算」して obs[:, 1:] を使う形が多い. 揃えること
2. **action の時刻ずれ**: `obs[t]` から `obs[t+1]` への遷移に対応する action は `action[t]`
3. **KL の重み**: design_notes §10.7 の通り `kl_scale * max(kl, free_nats) - log_p`
4. **encoder の B*T flatten**: `einops` で時間軸を畳む。さもないと CNN が 5D 入力を拒否

## 成功条件

- 全テスト緑
- 5 episode で 1000 step 学習し、recon の visual が元画像と似る
- KL, recon, reward 全部が右肩下がり (上下動はあって OK)

---

# Phase 5: Imagination rollout

## ゴール
学習済み WM の latent space で 15-step rollout を実行できる。
**想像軌跡が暴走しない (NaN にならない、絵が破綻しない)** ことを確認。

## 設計書の対応
`design_notes.md` §5 [D], §10.3 RSSM.imagine

## 触るファイル

| ファイル | 種類 |
|---|---|
| `src/dreamer/world_model.py` | `imagine()` メソッド追加 |
| `tests/test_imagination.py` | (Claude 提供) |
| `notebooks/05_imagine.ipynb` | 想像 rollout を画像で可視化 |

## WRITE ME 一覧 (予定)

| ID | 場所 | 何を |
|---|---|---|
| `#5.1.1` | `WorldModel.imagine` | (h_0, s_0, policy_fn) → 15 step rollout |
| `#5.1.2` | (Notebook) | 想像 rollout を decoder で復号して **動画 GIF** 化 |

## テストと合格基準

- 15-step rollout が **NaN なし** で完走
- 動画にしたとき walker が **少なくとも数歩は破綻せず動く** (完璧でなくて OK)
- 想像 reward の予測値が**合理的な範囲** (例: 0-30)

## 落とし穴

1. **policy_fn のインターフェース**: `(h, s) → action` か `(feat) → action` か事前に決めて固定
   - Phase 6 で actor がここに入る
2. **GRUCell hidden の保持**: 時間ループで `h` を変数として持ち回る. tensor を新規作成しないように `.clone()` の必要性を確認
3. **想像中は policy_fn が actor**: Phase 5 単体では random policy で OK

## 成功条件

- Notebook の動画で walker が動く (数フレームは形を保つ)
- imagine の出力 dict に reward, value (Phase 6 で), feat が揃う

---

# Phase 6: Actor-Critic + λ-return

## ゴール
想像 rollout の上で Actor (policy) と Critic (value) を学習。
**想像内で V (value) が学習で増える**ことを確認 (= policy が報酬を取れるよう改善している)。

## 設計書の対応
`design_notes.md` §5 [D], §10.5, §10.6, §10.8

## 触るファイル

| ファイル | 種類 |
|---|---|
| `src/dreamer/actor_critic.py` | WRITE ME 多数 |
| `src/dreamer/utils.py` | `lambda_return()` 関数 |
| `tests/test_actor_critic.py` | (Claude 提供) |
| `notebooks/06_actor_critic.ipynb` | V のスカラー値が学習で増える曲線 |

## WRITE ME 一覧 (予定)

| ID | 場所 | 何を |
|---|---|---|
| `#6.1.1` | `Actor.__init__/forward` | Dense(400, ELU) × 3 + Dense(2A), TanhNormal 構築 |
| `#6.1.2` | `Actor.act(feat, training)` | rsample + tanh で action |
| `#6.2.1` | `Critic.__init__/forward` | 同上 + Dense(1) で scalar value |
| `#6.3.1` | `utils.py::lambda_return` | recursive な λ-return 計算 |
| `#6.4.1` | `ActorCritic.loss_actor` | -mean(discount × returns) |
| `#6.4.2` | `ActorCritic.loss_critic` | -mean(discount × value.log_prob(returns)) |
| `#6.4.3` | `ActorCritic.update` | 2 つの optimizer をそれぞれ step |

## テストと合格基準

- λ-return の単体テスト: 既知の値で recursion が正しい
- Actor の出力が `[-1, 1]` 範囲
- Critic loss が学習で減少
- Notebook で**想像内 V が学習で増える** (簡単な reward 環境で確認)

## 落とし穴

1. **`stop_gradient`**: critic 学習時に returns に sg、actor 学習時には sg しない. 逆だと壊れる
2. **`rsample` vs `sample`**: actor は rsample 必須 (reparam)
3. **`tanh(mean)` の数値範囲**: `5*tanh(m/5)` は **(-5, 5)** 範囲. その後 `tanh(Normal(.))` で `[-1, 1]`
4. **discount の cumprod**: `cumprod(γ * pcont)` の積算順序. 最初の step は 1.0 から始める
5. **想像 rollout の長さは 15 だが、λ-return 計算では 14 step (最後を bootstrap 用)**:
   `returns[:-1]`, `value[:-1]`, `bootstrap=value[-1]`

## 成功条件

- 全テスト緑
- 簡易環境 (例えば「常に reward=1」) で V が 0 → 約 1/(1-γ) ≈ 100 まで増える

---

# Phase 7: 統合学習ループ

## ゴール
Phase 1-6 を全部繋いで、**DMC walker_walk で実際の return が上がる**こと。

これが達成できれば**「Dreamer v1 を再現できた」**と言える。

## 設計書の対応
`design_notes.md` §5 全体, §10.9 学習スケジュール

## 触るファイル

| ファイル | 種類 |
|---|---|
| `src/dreamer/agent.py` | WRITE ME (高レベル API) |
| `src/dreamer/train.py` | WRITE ME (メインループ) |
| `notebooks/07_colab_train.ipynb` | Colab で 1M step 学習 |
| `results/` | 学習曲線 + 動画 |

## WRITE ME 一覧 (予定)

| ID | 場所 | 何を |
|---|---|---|
| `#7.1.1` | `Agent.__init__` | WM + AC を持つ |
| `#7.1.2` | `Agent.act(obs)` | 推論 (Normal(0,0.3) ノイズ込み) |
| `#7.1.3` | `Agent.learn(batch)` | WM.loss → step → AC.loss → step |
| `#7.1.4` | `Agent.evaluate(env, n_episodes)` | ノイズなしで return 平均 |
| `#7.2.1` | `train.py::main` | 引数パース + config 読み込み + ループ |
| `#7.2.2` | `train.py::collect_episode` | エージェント or random で 1 episode |

## 学習スケジュール

```
prefill: 5 random episodes
pretrain: 100 train_steps
loop:
  collect 1 episode with agent (action_repeat=2)
  for _ in range(100): learn(batch)
  if eval_every: evaluate(env, 10 episodes)
  if save_every: save checkpoint
```

env step ベースで 1M step まで.

## テストと合格基準

- Colab T4 で **walker_walk return が ~500 まで上がる**
- 公式: 約 750. ~500 でも「動いている」と言える
- 学習曲線、想像 rollout 動画を `results/` に保存

## 落とし穴

1. **collect 中の eval mode**: actor は noise を加える、critic は使わない
2. **Replay buffer のサイズ**: 1M で OK. 超えたら FIFO
3. **GPU/CPU の dtype**: 全て float32 で統一
4. **Colab の MUJOCO_GL='egl'** 設定を毎セル冒頭で
5. **wandb logging**: loss だけでなく eval/return, recon image, imagine video も記録
6. **Checkpoint**: torch.save 時に `optimizer.state_dict()` も保存. resume できるように

## 成功条件

- DMC walker_walk で return が上昇 (目標 500+)
- README に学習曲線と差分メモを掲載
- Colab で誰が走らせても再現できる

## 終了処理

```bash
# README 更新
# results/curves/walker_walk.png 配置
# results/videos/imagine.gif 配置
git add .
git commit -m "phase 7: integrated training loop, walker_walk return ~500"
git push
```

---

# 共通パターン

## Phase 終了時の振り返り

各 Phase 完了後に `docs/notes_phase{N}.md` に**自分の言葉で**:
1. 何を実装したか (1 段落)
2. 詰まった点 / 学んだ点 (3-5 個)
3. 公式実装と比べて分かりにくかった点 (1-2 個)
4. 次の Phase につながる疑問 (1 個)

→ これが鈴木先生コンタクト時の**「自分が動かして得た知見」**になる。

## エラーが出たときの調査手順

1. **形状を print する**: 一番多い原因
   ```python
   print(f"x.shape={x.shape}, dtype={x.dtype}, device={x.device}")
   ```
2. **NaN チェック**: `torch.isnan(x).any()`
3. **公式コードと比較**: design_notes §10 の対応箇所を再読
4. それでも詰まったら **WRITE ME ID とエラー全文** で質問

## Commit メッセージ規約

```
phase {N}: {何をしたか}

例:
phase 1: env wrapper + replay buffer
phase 3: RSSM with prior/posterior, KL loss
phase 7: integrated training loop, walker_walk return ~500
```

## ブランチ戦略

- `main`: phase 終了で merge する安定版
- `phase-{N}`: 各 phase の作業ブランチ (オプショナル)
- 単独開発なので main 直 push でも OK

---

## 困ったら

このファイルと `design_notes.md` を**Cmd+Shift+F で検索**するのが最速。
それでも分からなければ**WRITE ME ID** を添えて質問。

各 Phase の冒頭でこのファイルを開き、ゴールと WRITE ME 一覧を眺めることを習慣に。
