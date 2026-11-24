# C2 经验记录：稀疏 advantage 下，先调有效 batch（T4a）

- 日期：2026-11-21
- 场景：native PTCG，old C1（616 池、Muon 2B、arena .650），C2 固定 WALLY，
  head-only fine-tune（trunk/embedding 冻结，只训 policy 通路 + value 头）
- 一句话结论：**在稀疏终局奖励、弱 advantage 的 fine-tune 里，先把有效 batch
  提上去（`--accum-minibatches 4→8`）比换优化器/加 KL 更便宜、更有效；
  它把梯度的信噪比提上去，归一化优化器的随机游走就小了。**

> **对手约束（2026-11-21，用户定）**：C2 的对手不能使用 learner 的权重，
> 只能是 **frozen C1 或 C1 前期快照**；C2 期间对手不更新。本文 T4a 的 .705
> 是在 **95% mirror（learner 自己当前权重）+ 5% C1** 下拿到的，**不符合该约束**，
> 只能作为“batch 杠杆”的机制证据。同一配方在 **0% mirror（100% frozen C1 阶梯）**
> 下的复测是 F1/F2（见 §7），不能直接把 .705 当成约束下的 C2 结果。

## 1. 症状

V20/V21 的 C2（learner 固定 WALLY，对手 frozen C1 + self-play）在 native trainer 上：

- **全模型 fine-tune 必崩**：Muon 25M 到 .14，全参数 AdamW 到 .007；
- 各种尝试（lr 1e-4/3e-5/1e-5、ent 退火/ent=0、ref-KL、value 补丁、外部 bank、
  paired deck、blaziken 过采样、快照 ladder、headblock）都只能延缓，不能阻止；
- **head-only** 是唯一不崩的，但 3-seed 均值 ≈ C1（.646 vs .650），500M 长跑
  到 100M/200M 就 early stop。
- 训练内指标（ent 0.36–0.48、kl 1e-4、refkl）看起来完全健康，但外部 arena 在掉。

## 2. 机制

1. **reward 是终局 win/loss**：一局 100+ 决策只有终局一个 ±1；GAE 后
   per-decision advantage 很小（~0.01–0.1），噪声占比高。
2. **优化器是归一化的**：Muon 对每个 2D 参数做 Newton–Schulz 正交化，更新
   矩阵谱范数 ≈ 1；AdamW 的 `m/√v` 每元素 ≈ ±1。**lr 只缩放步长，不改变
   噪声/信号的相对权重**；cosine 降 lr 只是让随机游走变慢，不改变方向。
3. **弱信号 × 固定方向步长 × 128 minibatch/epoch × 几千 epoch = 累积随机游走**；
   trunk 冻住只防特征漂移，head 自己的策略仍会漂。
4. **有效 batch 是信噪比杠杆**：梯度方差 ~1/B，信号（均值）不变，SNR ~√B。
   `--accum-minibatches 8` 把 8 个 minibatch（8×1024=8192 样本）的梯度累加后
   再做一次 optimizer step；方向更接近真实均值，噪声方向不再拿到满幅步。

## 3. 关键实验（old C1、616 池、100M aggregate、4×4090）

Base = D1 配方：head-only + 5% C1 + ref-KL 0.5（decay 0.5）+ 95% self
+ ent 1e-4 anneal + lr 3e-5 cosine；D1 本身 accum 4。
（这里的 “95% self” 是 mirror：对手直接用 learner 当前权重，违反开头的对手约束，
见 §7 的 frozen-only 复测。）

| 臂 | 有效 batch | 25M | 50M | 100M | 形态 |
|---|---|---|---|---|---|
| B2（无 accum） | 1024 | .6200 | .5233 | .2250 | 崩 |
| D1 | 4096 | .6967 | .6400 | .6583 (200g) | 早期见顶后下滑（200M .5883） |
| **T4a** | **8192** | .6733 | .6933 | **.7050** (200g) | **单调上升** |
| T4b | 16384 | .6767 | .6867 | 待出 | — |
| H2（无 C1 锚） | 1024 | .6767 | .4300 | .0817 | 崩最快 |
| C1 基线 | — | .650 (202g) | — | — | — |

T4a 的 100M（200 局/agent，temp 0）：

```
ak .635  dra .795  bla .685  mean .7050  CI95 ±.0361
C1 .579      .698      .673       .650
```

**三项全部 ≥ C1，均值 +.055**；25M→50M→100M 单调上升（.673→.693→.705），
和 D1 的“25M 见顶后下滑”完全不同。T4a 只把 `--accum-minibatches` 从 4 改成 8，
**没有 KL、没有换优化器、没有改 reward/对手/结构**。

## 4. 可复用判据

- **先调 batch，再调 optimizer/KL**：在稀疏 reward 的 fine-tune 里，
  `--accum-minibatches`（4/8/16）是最便宜的 SNR 杠杆；有效 batch 上去后，
  归一化优化器的随机游走会明显变小。
- **评估口径**：100M 附近用 200 局/agent；看 per-agent 是否三项都 ≥ 基线，
  以及 25M→50M→100M 的趋势（单调上升 vs 见顶下滑）。
- **“固定步长”不是 lr**：lr 会 cosine 衰减；固定的是归一化更新方向的幅度。
  降 lr 只降速，不改变噪声/信号比。
- **训练内指标不可信**：ent/kl/refkl 健康不代表外部 arena 不漂；必须看 arena。

## 5. 边界（不要过度外推）

- T4a 只跑到 100M；200M/500M 是否继续上升未验证（趋势是正的，但需延长确认）。
- 单 seed；3 个外部 agent 的 arena 是代理指标，CI ±.036（200 局）。
- 只验证了 old C1 + 616 池 + head-only；新池/新 C1 未验证。
- **T4a 的对手是 95% mirror（learner 权重）+ 5% C1，违反 2026-11-21 对手约束**；
  .705 证明的是 batch 杠杆本身，**不是** frozen-only 下的 C2 结果。frozen-only 下
  learner 对 C1 起始胜率 ~0.82（F1 epoch1 lgw 0.818），advantage 更弱，是否还成立待 F1/F2。
- T4b（accum 16）只到 50M（.6767/.6867，mirror 版）；T1a（ref-KL β 2.0）未跑；
  trust region、EMA champion、stage ladder、head 优化器都还没出结果；
  不能把 T4a 当成“C2 已解决”。
- 未使用：稠密 reward、外部 agent 进训练（按用户要求排除）；C2 对手不做权重更新
  （用户要求，等 C2 首次成功后在 MoE 阶段再考虑）。

## 6. 复现

T4a（历史版，对手 5% C1 + 95% mirror，**不符合现行对手约束**）：

```bash
export PTCG_TABLES=native/ptcg_tables_616.bin PTCG_TRAIN_MODE=headonly \
       PYTHONPATH=data:.:native PTCG_BANK_ACT_CAP=64
python3 -u native/train_native.py --run-name v21_c2_t4a_100m \
  --timesteps 100000000 --total-agents 2048 --minibatch 1024 \
  --num-threads 40 --gpus 4 --cudagraphs -1 --seed 42 \
  --deck-pool decks/pool_616_c1 --lr 0.00003 \
  --anchor-decks dudunsparce_mega_lopunny_ex_wally_s_compassion__02 \
  --p-anchor-learner 1.0 --mix-self 1.0 \
  --league-ckpts experiments/native-v19_c1/final.bin --league-pct 0.05 \
  --ref-kl-coef 0.5 --ref-kl-bank 0 --ref-kl-decay-frac 0.5 \
  --ent-coef 0.0001 --anneal-ent-coef --min-ent-coef-ratio 0.1 \
  --checkpoint-interval 10 --state-history 2 \
  --accum-minibatches 8
```

F1/F2（**frozen-only 对手，0% mirror**；与 T4a 的唯一差别是对手行）：

```bash
# F1: 100% frozen C1 final
C1=experiments/native-v19_c1
python3 -u native/train_native.py --run-name v21_c2_f1_100m \
  ...same flags as above... \
  --league-ckpts $C1/final.bin --league-pct 1.0 \
  --ref-kl-coef 0.5 --ref-kl-bank 0 --ref-kl-decay-frac 0.5 \
  --accum-minibatches 8
# F2: C1 前期阶梯 ep2000/ep3000/ep4000/final，各 25%（合计 100% frozen）
python3 -u native/train_native.py --run-name v21_c2_f2_100m \
  ...same flags as above... \
  --league-ckpts $C1/0000000262144000.bin,$C1/0000000393216000.bin,$C1/0000000524288000.bin,$C1/final.bin \
  --league-pct 0.25 \
  --ref-kl-coef 0.5 --ref-kl-bank 3 --ref-kl-decay-frac 0.5 \
  --accum-minibatches 8
```

`lgfrac` 应为 1.000（100% 行是 frozen bank），日志里 `loaded N league banks @ pct rows each`。

评测：`ptcg.rl.arena`，候选 WALLY，对 `opp_alakazam/opp_dragapult/opp_blaziken`，
100/200 局，temp 0。

## 7. 下一步

- **F1/F2（frozen-only，0% mirror）100M 在跑**：25M/50M 100 局、100M 200 局 arena；
  早停规则 100M `<0.64` 或 `<50M-0.02`。这是约束下的第一个正式判定。
- 若 F1/F2 不崩但增益 < T4a：说明 batch 杠杆仍在，但 frozen-only 的弱 advantage
  需要更长的 200M/500M 或更强的信号手段（trust region/EMA/ladder）；
- 若 F1/F2 崩：说明 mirror 是 T4a 增益的必要成分，在“对手不更新”的约束下要换
  难度机制（例如 C1 前期 + 随机 deck 过采样、按 lgw 动态选 bank），而不是回到 mirror；
- T4b（accum 16）可在 frozen-only 下补 100M，确认 accum 8→16 是否还单调；
- 代码臂（T1b per-step KL trust region、T3 stage ladder、T5 EMA champion、T6 head 优化器）
  全部在 frozen-only base 上叠；
- 结果记录：`logs/c2_t_evals.md`；三臂 500M 汇总：`logs/c2_500m_summary.md`。

## 8. T4a 续训（95% mirror，反事实）最终结果

用户拍板“万一我错了”，把 T4a（95% mirror + accum 8）从 100M 续到 500M。
续训段用 100M 之后的状态接：ref-KL 关闭（原 run 50M 后已衰减到 0）、
ent 固定 1e-5、lr cosine 从 1e-5 起；其余不变。

| 总步数 | ak | dra | bla | mean | n |
|---|---|---|---|---|---|
| 100M | .635 | .795 | .685 | **.7050** | 200 |
| 150M | .640 | .750 | .660 | .6833 | 100 |
| 200M | .630 | .760 | .690 | .6933 | 200 |
| 300M | .610 | .770 | .650 | .6767 | 100 |
| 400M | .640 | .750 | .630 | .6733 | 100 |
| **500M** | .575 | .775 | .630 | **.6600** | 200 |

- **结论：95% mirror 也不能让 C2 继续上升**。100M 之后平台向下，500M 掉到
  .660（200 局，CI ±.037），比 100M 低 .045。
- 与合规臂（F1/F2/G1）合起来看：**所有 C2 变体都在 50–100M 见顶**
  （frozen-only 50M .68–.70，95% mirror 100M .705），**没有任何 C2 配置值得 500M**。
  单卡组 fine-tune 的天花板被对手/数据分布固定这件事本身锁死。
- 因此 C2 线正式封口，主线转到 MoE（见 `docs/design_moe5_v1_20261121.md`）：
  新 1369 池、5 个主流 family 专家头 + shared 头、双方 self-play、family 硬路由。
- 评测踩坑记录：把 `native_weights.py` 的 param_order 扩到 160 参数后，旧 120
  参数 checkpoint 的 import 直接报 reshape 错（500M 首评 NA）。已改为双布局
  兼容：`blob.size == old_total` 时只读 120 参数，`sd_to_blob` 缺 `moe.*` 时
  也只导出 120。任何“加参数”改动都要同时检查旧 checkpoint 的评测路径。
