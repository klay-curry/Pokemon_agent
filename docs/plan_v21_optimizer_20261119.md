# V21 规划：Muon/AdamW 混合优化器 + 新数据池 + C1 重训

- 日期：2026-11-19
- 状态：**V20 已归档（C2 未正式发车）；V21 执行中（2026-11-19 晚）**：① optimizer split 已实现并通过单测/逐位一致/审查（0 P0），A0 第一轮四臂未过闸 = aux lr 量级错 ~2 个数量级（§12.1），待校准；② B0 数据审计完成、可定稿（§12.2），开放 excadrill 拆分与 anchor 名单两个用户拍板项；③ BC 线按用户指令挂起（§12.0）；④ **C1 重训已发车（2026-11-19 20:05，用户拍板跳过 pilot）：v21_c1 = 新 1369 池 + 混合采样 v1 + hybrid(aux 3e-5) + 2B，SPS 15.3K，ETA ~36h，BUILD_LOG 23.6**
- 前置：V20 C2 约束实验（old 616 pool + ref-KL + entropy anneal）
- 关联：`docs/training.md` §9/§10、`gates_v20_c2.sh`、`BUILD_LOG.md` Step 22
- 归档补充：2026-11-19 17:xx 的完整验证矩阵见 §8（ref-KL β=10 / 90-10 mirror / learner-only adv norm 均未改变漂移轨迹）

## 0. TL;DR

1. **Muon 分组是当前 native trainer 的一个明确偏差**：`native/src/muon.cu` 的 `muon_step` 对所有 `ndim >= 2` 的参数一律做 Newton–Schulz 正交化，包括 `card_id_emb` 等词嵌入表、`score_mlp`（最终 policy 打分头）、`value_mlp`（critic 头）、`q_mlp`/`dec_mlp`；只有 1-D 参数走 EMA，不是 AdamW。
2. 按 Muon 官方实现与规模化实践（Moonlight/Kimi K2），正确分组是：**hidden 2D → Muon；embedding、输出头、1-D → AdamW**。把最终 policy head 交给 Muon 会放大 entropy/ref-KL 梯度的步长，是 C2 策略漂移的放大器。
3. **V21 = 新 1369 池（famfix2）+ C1 重训（2B，hybrid 优化器）+ real 60% / family 40% 混合采样 + plain C2 + family MoE**。C1 重训是唯一基础模型；C2/MoE experts 从新 C1 派生。A0 先在旧 C1/旧池做 optimizer split smoke（校验 split 本身），再发新 C1 2B（见 §10/§11）。
4. 当前 V20 证据：`v19_c2_kl_b005`（old 616 pool，`ent 0.001`，`ref-KL 0.05`，lr 1e-4）在 10 epoch 内仍崩：`lgw 0.766→0.131`、`ent 0.359→0.679`、`refkl 0.010→2.245`。ref-KL 实现本身在工作（`refkl` 单调上升），但 0.05 太弱；**β=10 的 4-epoch 对照与 β=0.05 几乎逐行相同**（refkl e4 0.455 vs 0.500，lgw e4 0.658 vs 0.659），说明 native loss 下 ref-KL 的有效 β 尺度远大于 Top4 的 5e-4；要靠 KL 钉住需要 β~100，反而会阻止专精。**V20 不再发车，优化器分组是下一根杠杆**（详见 §8）。

## 1. 外部证据

### 1.1 知乎文章《Muon: 新一代神经网络优化器》

- URL：https://zhuanlan.zhihu.com/p/30895340275
- 文章摘要明确：Muon 是“专门针对神经网络隐藏层的二维权重参数设计”的优化器；二维隐藏层权重（全连接/卷积展平）用 Muon，而 **bias、向量、输入词嵌入、输出分类头仍交给 AdamW**。
- 用户引用的实验结论（“词嵌入层和最后分类头仍用 AdamW、中间线性层用 Muon 能取得最佳性能”）与文章一致；这不是单一实验，而是 Muon 的默认配方。
- 取文限制：知乎开放平台 CLI 只返回标题/URL/约 1053 字摘要；原文直连 403，无公开全文接口。结论用 Muon 官方与规模化实践交叉验证，见 1.2。

### 1.2 Muon 官方与规模化实践

- Keller Jordan 官方实现 `MuonWithAuxAdam`：`body` 中 `ndim >= 2` 的 hidden 权重走 Muon；`head`、`embed`、所有 gain/bias/1-D 走 AdamW。作者原话：*"Empirically, we find that it is also important to optimize input and output parameters using AdamW, even though these are typically 2D."*
- 典型超参：Muon lr 0.02、aux AdamW lr 3e-4、betas (0.9, 0.95)、wd 0.01。
- Moonlight（3B/16B MoE）与 Kimi K2：Muon 用在 hidden 矩阵，embedding/head/norm 用 auxiliary AdamW，并加 weight decay / QK-clip 做稳定性。
- 边界：DORoT（30M 小模型）报告 embedding 也用 Muon 可降 loss；MuonIO 提出给 embedding/LM head 单独的 norm-aware 优化器。因此“最佳”是经验默认，不是定理；对 2.3M 的我们，hybrid 是安全默认。

### 1.3 Top 文档的防漂移方法（与本项目直接相关）

| 队伍 | 方法 | 关键数字 |
|---|---|---|
| Top1 | 固定单卡组 specialist；对手是 Stage1–3 + arena 的渐进冻结阶梯；按胜率 EMA 给弱项加权；entropy **1e-4** | Dragapult 76.2%→85.7–86.8%；Hydrapple 78.3%→83.5% |
| Top4 | EMA champion + **小 KL 到 champion（5e-4）** + entropy 4e-3 | 唯一明确写 KL 防漂移的队 |
| Top7 | entropy bonus **退火**（先大后小）；current+past checkpoints 当对手 | 无具体值 |
| Top8 | league 历史快照 archive “stops regression”；**单卡组 fine-tune 500k games 天花板，早停** | 超过 500k games 无收益 |
| Top3 | Phase2 固定卡组 4.2B steps；AdamW lr 1e-4、entropy 0.01；对手 = 固定 recent Phase1 policies + mirror snapshots | 66%→76–77% |
| Top2 | 修 PufferLib Muon 1-D bug（无 weight decay 无限动量）+ KL>0.05/vf>0.5 skip；anchor training 无益/有害 | Muon 2e-3→2e-4 |

结论：top 队没有我们这套组合（单冻结 C1 + 常数 ent 0.01 + Muon 全 2D + 300M aggregate）。最接近的是 Top1（固定卡组 + 多级冻结对手 + 小熵）与 Top4（小 KL）；Top3 的 AdamW 固定卡组长期训练是一个强存在性证明。

## 2. 当前实现审计

### 2.1 优化器

- `native/src/muon.cu` `muon_step`：按 `param_alloc->regs` 顺序遍历；`ndim >= 2` → `muon_nesterov` + 5 步 Newton–Schulz；`ndim < 2` → `muon_nesterov_ema` + `--muon-wd-1d`。**没有 AdamW 路径**。
- 参数表（`native/tools/native_weights.py`）中以下 2D 参数全部走 Muon：
  - 词嵌入：`card_id_emb`、`attack_id_emb`、`skill_id_emb`、`zone_emb`、`owner_emb`、`sel_type_emb`、`ctx_emb`、`opt_type_emb`；
  - 输出/打分头：`score_mlp`（最终 policy logits）、`value_mlp`（critic）、`q_mlp`、`dec_mlp`；
  - hidden：`blocks.*`、`card_mlp` hidden、`token_proj`、`global_mlp`、`bcast`、`attack_mlp`、`skill_mlp`、`opt_card`、`opt_attack`。
- 1-D 参数走 EMA + wd；这修了 top2 的 PufferLib Muon 1-D 崩坏路径，但**不是 AdamW**。
- 状态文件只存 Muon momentum（`docs/training.md` §8），没有 AdamW m/v。

### 2.2 C2 约束现状

- ref-KL 已实现：`--ref-kl-coef`、`--ref-kl-bank`、`--ref-kl-decay-frac`；loss 只对 learner 行生效，日志有 `refkl`；成本约 +17% SPS（`docs/training.md` §10）。
- entropy 退火已支持：`--anneal-ent-coef`、`--min-ent-coef-ratio`（`native/train_native.py:813`）；默认关闭。
- **V20 smoke 结果（重要）**：old 616 pool、C1 热启动、WALLY 固定、league 1.0、lr 1e-4、`ent 0.001`、`ref-KL 0.05`、`decay 0`：

| epoch | lgw | ent | refkl |
|---|---|---|---|
| 1 | 0.766 | 0.359 | 0.010 |
| 5 | 0.570 | 0.537 | 0.863 |
| 8 | 0.262 | 0.650 | 1.895 |
| 10 | 0.131 | 0.679 | 2.245 |

ref-KL 在上升但政策仍在漂移：0.05 的系数不足以对抗 entropy/PG 的累积。这支持“优化器分组是下一根杠杆”，而不是继续只调 KL。

### 2.3 数据池

- C1 在 **616 副旧池**训练（`decks/pool_617_archive` 去掉 r4d；`native/ptcg_tables_616.bin` 是本次导出的 616 blob）。
- `decks/pool` 现在是 **1369 副 famfix2**（1134 coverage / 58 real / 40 patch / 34 weakness / 32 archetype / 28 benchdamage / 26 anti_ex / 14 self_combo / 1 submission + 2 手工）。
- 新旧池重名只有 2 副（WALLY 与 r4d）；C1 没见过新池的牌。
- rollout 探针（C1 热启动、league 1.0、WALLY 固定）：
  - 同卡组 vs 同卡组：lgw 0.551（bank 正常）
  - 旧 616 uniform：lgw 0.742
  - 新 1369 uniform：lgw 0.982（C2f 实际配置，优势饱和）
  - 新 1369 tier 加权（real 60%）：lgw 0.008（反向饱和）
- 结论：新池必须先修 family/tier 采样，并让 C1 在新池上重训/适配；否则冻结 C1 在新池上不是“同分布对手”。

### 2.4 当前 V20

- `gates_v20_c2.sh` 已修 gate 解析（summary 行不再假 PASS），配置 = old 616 pool + ref-KL + entropy anneal，baseline 0.6502，50M aggregate 裁决。
- 当前 `v19_c2_kl_b005` 是 10-epoch smoke，不是正式 V20 300M/100M；正式 V20 是否发车取决于系数 sweep。

## 3. V21 方案

### 3.1 参数分组（核心改动）

| 组 | 参数 | 优化器 |
|---|---|---|
| hidden 2D | `blocks.*`（qkv/proj/ffn）、`card_mlp.1/3`、`token_proj`、`global_mlp`、`dec_mlp`、`bcast`、`attack_mlp`、`skill_mlp`、`opt_card`、`opt_attack`、`q_mlp` | **Muon** |
| 输入 embedding | `card_id_emb`、`attack_id_emb`、`skill_id_emb`、`zone_emb`、`owner_emb`、`sel_type_emb`、`ctx_emb`、`opt_type_emb` | **AdamW** |
| 输出头 | `score_mlp`（policy logits）、`value_mlp`（critic） | **AdamW** |
| 1-D | 所有 LayerNorm gain/bias、bias、`readout` | **AdamW** |

- `q_mlp` 是候选 query 的 hidden 投影；先留在 Muon。若 A/B 显示 head 仍漂，再把它移到 AdamW。
- `dec_mlp` 是 hidden 选择上下文投影；留 Muon。
- `opt_card`/`opt_attack` 是 option 特征投影；先留 Muon，A/B 可切换。

### 3.2 超参与实现

- 新增 CLI：`--adamw-lr`（初始 3e-4，或 1e-4）、`--adamw-beta2 0.95`、`--adamw-wd 0.01`、`--optimizer-split`（默认 off，保证旧 run 逐位不变）。
- Muon lr 不变：C1 1e-3 cosine；C2 1e-4 cosine。
- `muon.cu`：给 `AllocEntry` 加 `use_adamw` 标记（或在 `ptcg_model.cu` 注册时写平行 mask）；`muon_step` 对 `use_adamw` 走新 AdamW kernel（m/v、bias correction、decoupled wd），保留 gate 语义；其余路径 bit-identical。
- `ptcg_model.cu`：按参数名/角色标记分组；`native_weights.py` 的 PT_PARAMS 顺序不变。
- 状态：`state_latest.bin` 增加 AdamW m/v 段（版本号 bump 或追加 section）；旧 checkpoint 无 AdamW 状态，warm-start 时 AdamW 从 0 开始；resume 需兼容。
- 测试：小矩阵 AdamW 单测；无 split 时与旧 path 逐位一致；SPS/VRAM 复核。

### 3.3 训练配方

- **V21 C1 重训**：新 famfix2 池 + `sample_weight` 权重文件（`--deck-weights-file`，把 tier 概率转成 per-deck 权重，归一化 mean 1）+ 混合优化器；预算先做 **1B aggregate pilot**，胜者再跑完整 2B（与 C1 可比）。entropy 0.01 + anneal；C1 from scratch 不需要 ref-KL。
- **V21 C2**：新 C1 冻结；learner WALLY 固定；对手随机新池（tier/family/weakness 采样）；ref-KL 到新 C1（系数 sweep 0.05→1.0，decay 0–0.5）；entropy 1e-3→1e-4 anneal；预算 ~100M aggregate（Top8 500k games 天花板），checkpoint 选择而非跑到 300M。
- **对手阶梯（可选，Top1）**：C1 + C2-minimal/C2b/中期快照作为多 bank（`--league-ckpts` ≤16，总 pct ≤1.0）；至少保留 10–20% mirror/self-play 或历史快照，避免只学打一个 generalist（Top8 的“中等飞行员”批评）。
- **采样顺序**：先修 pool 的 family/tier 标注，再生成 weights 文件；不要再用 uniform 1369。

### 3.4 排期与成本

| 阶段 | 内容 | 成本 | 依赖 |
|---|---|---|---|
| A0（现在） | V20 ref-KL/ent sweep + 优化器分组 smoke（old 616，warm-start C1，10 epoch） | 每跑 ~10 分钟 | 无 C1 重训 |
| A1（A0 通过） | V20 正式 C2（old 616，ref-KL/entropy，100M aggregate） | ~3h | A0 |
| B1（数据池修复后） | V21 C1 pilot：新池 + 混合优化器，1B aggregate，A/B 旧优化器 | 每臂 ~24h（11.5K SPS） | 新 pool 定稿 |
| B2（B1 通过） | V21 C1 完整 2B（胜者配置） | ~48h | B1 |
| C1（B2 后） | V21 C2 specialist（新 C1 冻结，100M aggregate） | ~3h | B2 |

- C1 重训是长杆；B1 的 1B pilot 可以先回答“混合优化器是否更好”，避免直接烧 2B×2。
- 如果 B1 的收益不明确，V21 暂缓，保留 V20 的 C2 约束版本。

## 4. 指标与闸门

- **训练健康**：ent 不出现单调失控（warm-start C1 签名 ~0.36–0.5；红线趋势 >0.8）；`refkl` 不应无限增长（< ~0.5 且趋稳）；lgw 不持续下行；klskip/vfskip < 5%；SPS/VRAM 正常。
- **外部场**：arena vs 3 meta 对手（alakazam/dragapult/blaziken），baseline 0.6502；50M aggregate 裁决 <0.55 停；终局 ≥0.65。
- **C1 重训**：新 C1 vs 旧 C1 在外场/ battery / matrix 不回退（battery macro 0.5687、wally exp 0.7717 为参照）。
- **C2**：wally exp ≥0.8459（C2 minimal 的 +7.4pt）或更高；battery macro 允许 specialist 退化，但需记录（C2 minimal −12.8pt 是已知代价）。
- **预算**：C2 ≤100M aggregate（≈500k games）；超过 Top8 天花板无收益。

## 5. 风险与回滚

- **优化器分组实现 bug**：参数 mask 错、AdamW 状态未存、NCCL 梯度不一致。用无 split 的 bit-identical 测试 + 小矩阵单测 + 10-epoch smoke 防守。
- **aux lr 不匹配**：3e-4 对 2.3M 模型可能过大/过小；先 sweep {1e-4, 3e-4}，看 entropy/lgw。
- **ref-KL 过约束**：0.05 已不足，但 1.0 可能阻碍专精；用 decay + 小系数 sweep，观察 wally exp 与 refkl。
- **新池 OOD**：C1 未见新牌；必须 C1 重训/适配后再做 V21 C2。
- **成本**：C1 重训 2 天/臂；先 1B pilot，避免盲目烧卡。
- **回滚**：`--optimizer-split` 默认 off；旧 run/旧 checkpoint 逐位不变；状态格式兼容旧文件。

## 6. 待决问题

1. `q_mlp` / `opt_card` / `opt_attack` 是否也归 AdamW？先按表分组，A/B 决定。
2. AdamW aux lr 用 1e-4 还是 3e-4？是否与 Muon 共享 cosine？
3. V21 C1 是 1B pilot 还是直接 2B？是否先在新池上 fine-tune 旧 C1 作为中间对照？
4. 新池采样：tier `sample_weight`（real 60%）vs family-uniform vs weakness boost；C2 的对手分布需要用户最终拍板。
5. 是否在 C2 加入 10–20% mirror/历史快照（Top1/Top3/Top8），还是严格“单冻结 C1”（用户原设计）？
6. V21 C1 是否也用 ref-KL 到 BC/S1？from-scratch 没有自然 reference；暂不做。

## 7. 参考

- 知乎《Muon: 新一代神经网络优化器》：https://zhuanlan.zhihu.com/p/30895340275
- Keller Jordan, Muon：https://github.com/KellerJordan/Muon ；https://kellerjordan.github.io/posts/muon/
- Moonlight: Scaling Muon to 3B/16B MoE：https://arxiv.org/abs/2502.16982
- Kimi K2 Technical Report：https://arxiv.org/abs/2507.20534
- 本项目 top 文档：`/root/autodl-tmp/pokemon_RL/docs/top8_writeups_raw/01..08`
- 当前实现：`native/src/muon.cu`（muon_step）、`native/train_native.py`（hypers/args）、`native/src/pufferlib.cu`（ref-KL loss）、`docs/training.md` §9/§10
- 当前证据：`logs/v19_c2_kl_b005`（10-epoch ref-KL smoke）、`logs/v19_c2f.log`（C2f collapse）、`logs/v20_c2_probes.md`（V20 gate 预留）

## 8. V20 归档记录与 2026-11-19 最新验证

### 8.1 决策

- **V20 C2 不正式发车**；C2f 保持归档。`v19_c2f`（新 1369 池 uniform）和 V20 的旧池 ref-KL 系列都不再续跑。
- V20 的 ref-KL/entropy 代码保留，作为 V21 C2 的工具，但**不再指望 KL 单独解决漂移**。
- 根因判定：策略漂移对 `ent_coef`（0.01→0.001）、ref-KL（β≤10）、10% mirror、learner-only advantage 归一化**都不敏感**；下一根杠杆是 **Muon 参数分组**（§3.1），而不是继续调 KL/熵。

### 8.2 验证矩阵（全部 C1 热启动、WALLY 固定、lr 1e-4、无 LR 退火）

| # | 配置 | 时长 | lgw | ent | 结论 |
|---|---|---|---|---|---|
| 1 | 同卡组 vs 同卡组（bank 公平性，rollout-only） | 1 epoch | **0.551** | — | 冻结 bank 路由/合并正常 |
| 2 | 旧 616 池 uniform（rollout-only） | 1 epoch | **0.742** | — | 可训练区间 |
| 3 | 新 1369 池 uniform（rollout-only，C2f 实际） | 1 epoch | **0.982** | — | 优势饱和 |
| 4 | 新 1369 池 tier 加权 real 60%（rollout-only） | 1 epoch | **0.008** | — | 反向饱和 |
| 5 | 旧 616 池，`ent 0.01`，无 ref-KL（基线） | 10 epoch | 0.708→**0.147** | 0.358→**0.684** | 崩 |
| 6 | 旧 616 池，`ent 0.001`，无 ref-KL | 10 epoch | 0.766→**0.131** | 0.359→**0.679** | ent 10× 降低**无效果** |
| 7 | 旧 616 池，`ent 0.001`，ref-KL **0.05** | 10 epoch | 0.766→**0.131** | 0.359→0.679 | refkl 0.010→**2.245** |
| 8 | 旧 616 池，`ent 0.001`，ref-KL **10** | 4 epoch | 0.750/0.710/0.629/0.658 | 0.352→0.492 | refkl 0.008/0.091/0.237/0.455；与 #7 的 4-epoch（0.766/0.697/0.613/0.659，refkl 0.010/0.099/0.251/0.500）**几乎逐行相同** |
| 9 | 旧 616 池，`league-pct 0.9`（10% mirror），`ent 0.001` | 4 epoch | 0.769/0.730/0.605/0.649 | 0.352→0.468 | 10% mirror **无效果** |
| 10 | 旧 616 池，learner-only advantage 归一化，`ent 0.001` | 10 epoch | 0.747/0.757/0.605/0.676/0.534/0.458/0.390/0.235/0.184/**0.125** | 0.363→**0.686** | 与基线几乎相同，**无效果** |

- **SPS**：ref-KL 开启 ~9.5K vs 关闭 ~11.5K（-17%）；`PTCG_REFKL_DEBUG` 的 device printf 只在调试时开。
- **ref-KL 确实进入梯度**：`[ppo] ref_coef=10.00000 inv_L=0.00177 d_ref=0.01770 frz=0 tag=0`，与 PG 同阶；β=10 仍不足以对抗 PG 的累积漂移。β~100 可以钉住，但会阻止专精。

### 8.3 解释（写入 V21 避雷）

1. **漂移不是熵项驱动的**：#6（ent 0.001）与 #5（ent 0.01）同轨；熵上升是策略漂移的**症状**，不是驱动。
2. **不是 mirror 驱动的**：#9 的 10% mirror 没有改变 e1–e4 轨迹。
3. **不是 advantage 归一化驱动的**：#10 的 learner-only 归一化没有改变 e10（0.125 vs 0.147）。
4. **是策略头/优化器驱动的**：native 的 `muon_step` 对所有 `ndim>=2` 参数（含 `score_mlp` 输出头、`value_mlp`、embedding）做 Newton–Schulz 正交化；小梯度也被归一化成固定谱范数步长，策略头在 128 minibatch/epoch 的同向更新下累积漂移。Top4 的 KL 5e-4 是在 AdamW/其 loss 尺度下的系数，不能直接搬；本 loss 的有效 β 尺度远大于它。
5. **正确修法是参数分组**（§3.1）：hidden 2D→Muon；embedding、`score_mlp`/`value_mlp` 输出头、所有 1-D→AdamW。KL 是辅助，不是主修。
6. **V21 C2 的对手分布要按“初始 lgw”校准**，不能盲用 tier 概率：新池 uniform=0.982（太弱）、real 60%=0.008（太强）；目标是初始 lgw ~0.65–0.75。新池必须先重训/适配 C1（C1 只见过 616 池，新旧池仅 2 副重名）。

### 8.4 代码与工件状态（归档）

- 新增 flags：`--ref-kl-coef`、`--ref-kl-bank`、`--ref-kl-decay-frac`、`--anneal-ent-coef`、`--min-ent-coef-ratio`（`native/train_native.py`；默认关闭，旧 run 行为不变）。
- `native/src/pufferlib.cu`：frozen bank 参考前向、fp32 logits 拷贝、`LOSS_REF_KL` 项与梯度、learner-only `ppo_var_mean`。
- `native/src/bindings.cu`：新 hypers 解析 + `losses/ref_kl` 日志键。
- `docs/training.md`：§10 flag 表 + reference-KL 段落。
- `gates_v20_c2.sh`：修复 summary 行解析（已用 `gate25000000_alakazam.json` 验证：`meanof=0.0067` 而不是空串；判 FAIL 而不是假 PASS）。
- 旧池资产：`decks/pool_616_c1`（616 副，WALLY anchor index 297）、`native/ptcg_tables_616.bin`（616 + 87 alt）。
- 构建：`CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH bash native/build_native.sh`。
- 归档日志：`probe_c2/kl_b005.log`、`kl_b10.log`、`mix90.log`、`advnorm.log`、`logs/v19_c2f.log`。
- 注意：当前 `.so` 已编译 learner-only 归一化 + `PTCG_REFKL_DEBUG` 门控打印；ref-KL 默认 0，C1 训练路径逐位不变。

### 8.5 V21 A0 验证清单（发车前）

1. **优化器分组**（§3.1）：hidden 2D→Muon；embedding/输出头/1-D→AdamW；`--optimizer-split` 默认 off；无 split 时与旧 path 逐位一致。
2. **10-epoch 旧池 smoke**（C1 热启动、WALLY 固定、league 1.0）：`lr` {1e-5, 1e-4} × aux `adamw_lr` {1e-4, 3e-4}；成功判据 ent ≤0.5、lgw ≥0.65 @e10。
3. **ref-KL with split**：β {0.05, 0.5}，decay {0, 0.5}；若分组改变了尺度，重新校准；不要用 β>10（钉死）。
4. **advantage 归一化 A/B**：learner-only（当前实现）vs 全元素；目前 10-epoch 无差异，V21 可保留 learner-only 作为正确性修正。
5. **新池采样校准**：tier mix（real 20/40/60%）、family-uniform、weakness boost；用**初始 lgw**选，目标 0.65–0.75；不要盲用 sample_weight 的 real 60%。
6. **C1 重训**：新 famfix2 池 + 混合优化器，1B aggregate pilot；旧 C1 只作 control/warm start。
7. **闸门**：`gates_v20_c2.sh` 的 summary 解析；50M aggregate 裁决 `< baseline-0.10` 停；终局 ≥0.65。
8. **预算**：C2 ≤100M aggregate（≈500k games，Top8 天花板）；checkpoint 选择/早停，不要跑到 300M。
9. **对手阶梯**：优化器修好后，再测 Top1 式渐进阶梯（C1 + C2-minimal/C2b/中期快照）和/或 10–20% mirror；单冻结 C1 是 Top8 明确指出的“中等飞行员”风险，不要作为唯一方案。

### 8.6 数据池 family 审计（2026-11-19，CPU-only）

对 `decks/pool_616_c1`（旧，C1 训练池）与 `decks/pool`（新 famfix2）逐副 60 张卡表做 Jaccard 审计：

| 指标 | 旧 616 | 新 1369 |
|---|---|---|
| 唯一 60 张卡表 | 616 / 616 | 1369 / 1369 |
| 现有 family 数 | **105 个名字前缀** | **26**（25 + 2 manual-real） |
| family 大小 中位/最小/最大 | 3 / 1 / 31 | 36 / 2 / 167 |
| 小于 5 副的 family | **68** | 1 |
| 小于 10 副的 family | **84** | 5 |
| 单副 family | **35** | 0 |
| family 内平均 Jaccard | 0.60–0.88（按大小） | 0.60–0.75 |
| 跨 family Jaccard≥0.9 的对 | **418** | 未逐对统计（coverage 族内部高相似） |
| 数据驱动聚类数 | 34@0.5 / 60@0.6 / 103@0.7 / 183@0.8 | — |
| 卡表里出现的 card id | **417** | **1428** |
| 出现率 <1% 的 card id | 159 | **1217** |
| 与对方池的完全同列表 | 17 | 17 |

**结论（直接回答“family 划分”问题）：**

1. **旧 105 个名字前缀不是可用的 family**：35 个单副族、68 个 <5 副、84 个 <10 副；更严重的是有 **418 对跨前缀 Jaccard≥0.9**（同一份卡表被拆到不同“family”，例如 Team Rocket 的 `__ariana`/`__energy`/`__proton` 前缀有 1.000 的同列表）。作为 MoE expert 会出现大量数据饥饿 + 同族重复专家。
2. **新 26 个 family 太粗且 coverage 占 83%**（1134/1369）；family 大小 2–167；新池还引入 1428 个 card id（旧池只有 417），C1 没见过——**新池不能直接做 C2 的冻结对手**，必须先 C1 重训/适配。
3. **数据驱动聚类给出自然的两级粒度**：0.5 阈值 34 族、0.6 阈值 60 族、0.7 阈值 103 族（≈旧前缀粒度）。建议：
   - **MoE expert 级：16–24 组**（从 34@0.5 合并小簇/稀有族进 shared/other；每组 ≥20 副）；
   - **family 采样/分析级：49–60 族**（0.6 聚类，或 `meta_families.csv` 的 49 个真实牌表族）；
   - **variant 分析级：105 前缀/616 副**（只做 deck-level 分析，不做 expert）。
4. **V21 plain C2 基线用旧 616 池**（C1 同分布，uniform 0.742），family 重标注只是元数据，**不需要为换标签重训 C1**；新 1369 池留到 C1 适配之后再上。
5. **C2 对手采样**：基线保持旧 616 **uniform**（C1 的分布），不要和优化器改动同时换采样；family-balanced 与 weakness boost 作为后续单变量 A/B（见 §9.4）。

> 审计脚本为一次性 CPU 分析；复现命令：对 `decks/pool_616_c1/*.csv` 与 `decks/pool/*.csv` 逐对算 Jaccard，阈值 {@0.5,0.6,0.7,0.8} 的连通分量即上表聚类数。

## 9. 用户 V21 规划对齐与架构补充（2026-11-19）

### 9.0 用户规划（摘要）

1. 训练池用旧的（扩充过的），检查生成卡组分布；主要改 family 划分，更细，为可能的 MoE 铺垫。
2. 优化器 + KL 是 V20 失败的可能根源，修改由实现侧负责。
3. C1 其余沿用；先朴素 C2（learner 固定卡组且更新，对手随机卡组、固定 C1 权重，采样沿用 C1）；基线确保后，在 C1/C2 之间加 family MoE 提高 C2 对手质量；C2 对手策略不随 learner 更新；若 MoE 有效可考虑 learn 侧自我更新，但绝不与 learn 侧共享。

### 9.1 对齐结论（与 §2/§3/§8 一致/需修正）

- **一致**：plain C2 配方（固定 WALLY、frozen C1、随机旧池、无 mirror、learner 正常更新）已被本轮的 bank/rollout 探针验证（同卡组 0.551、旧池 0.742）；V20 崩坏是优化器，不是该配方。C2 对手不随 learner 更新 = native league 的 frozen bank，已满足。
- **需修正 1（family 粒度）**：不要直接用旧 105 名字前缀。§8.6 显示 35 个单副族、418 对跨前缀近重复；MoE expert 用 **16–24 组**（从 34@0.5 合并小簇，每组 ≥20 副），family 采样/分析用 **49–60 族**（0.6 聚类或 meta_families 的 49 族），variant 分析用 105 前缀。Jaccard 聚类需人工语义校验（0.5 阈值的 143 副大簇可能混合了不同 archetype）。
- **需修正 2（C1 是否重训）**：用旧 616 池、只换 family 标签 → **C1 不需要重训**（标签只是元数据，策略分布不变）。换新 1369 池 → C1 必须先重训/适配（新池 1428 card ids vs 旧池 417；仅 17 副同列表）。不要把“换标签”和“换池”混为一谈。
- **需修正 3（C2 采样）**：plain C2 基线保持旧 616 **uniform**（C1 的分布）；family-balanced/weakness boost 作为后续单变量 A/B，不要与优化器修复同时上。
- **优化器/KL**：Muon 分组是主修；ref-KL 辅助。C1 用 Muon 在 self-play 下稳定，不必为 split 重训 C1；split 先只用于 C2 warm-start（A0 smoke）。

### 9.2 架构层补充问题（用户规划里未展开/未提到的）

1. **MoE 与 native frozen bank 的架构不兼容（第一 blocker）**：`WeightBank`/`load_frozen_bank` 假设 frozen opponent 与 primary 同一 `PT_PARAMS` 布局；真 MoE（多 expert + router）参数更多，当前 bank 加载不了。可选：(a) MoE 只作为训练期对手的独立前向路径；(b) 给 bank 增加 arch 变体；(c) 先做 **family-conditioned 单模型**（加 family embedding/router），而不是真 MoE。建议先 (c)/(a)，把真 MoE 留到 V22。
1b. **league bank 是 row-static，不能按 episode/deck 选 expert（第二 blocker）**：`env->tag` 在启动时按 row 固定（`my_vec_init`/`bank_layout`），一行永远打同一个 bank；而 C2 的对手 deck 是每局随机，family-MoE 需要**按对手 deck 的 family 逐局选 expert**。现有 league 做不到。要么每行固定 family（对手 deck 也按 family 固定，失去随机性），要么新增 per-episode tag/bank 选择路径，要么改成一个 **family-conditioned 单模型**（family id 作为输入，逐局给对手 family）。
1c. **`PT_MAX_BANKS=16` 的硬上限**：即使 per-episode 选择能实现，16–24 个 expert 超过 16 个 bank；expert 数要么压到 ≤16，要么用单模型/独立对手路径。
1d. **务实替代（推荐 V21 先做）**：family-conditioned 单模型（同一 `PT_PARAMS` + 一个 family embedding/id 输入）作为 C2 对手；训练期 env 已知对手 deck 的真实 family，逐局 stamp 到 obs（类似 frozen tag 列），推理期若提交再用公开卡推断。它避开了 1b/1c，且与当前 bank 兼容（learner 也用同一模型，family=WALLY）。真 MoE（多 expert 联合训练）留到 V22。
2. **Router 输入**：训练期对手 deck 已知（env 的 `deck_idx`），router 可用真 family；若未来提交 MoE，推理期 deck 隐藏，需要从公开卡推断 family（Top6 posterior / Top8 core Pokémon）。训练期与提交期要分开定义。
3. **Expert 数据充足性**：旧 616 按 105 前缀有 35 个单副族；按 16–24 expert 每组约 25–40 副，可用；但每个 expert 的**对局数**取决于采样，MoE 训练时要按 expert 分层统计，稀有族进 shared/other。
4. **MoE 训练 recipe**：per-family expert 从 C1 fine-tune（Top8 的 pin 法）还是 router over C1 snapshots？前者便宜、与 C1 同分布；后者需要多 C1 快照。建议先做 **per-family C1 experts + 真 deck family router（训练期）**，评估每 family 的对手质量，再决定真 MoE。
5. **C2 对手质量评估不能只看平均 lgw**：要按 family/deck 分层的胜率矩阵（`--deck-matrix-every` 或 arena battery）。MoE 的目标是**弱 family 的对手质量**，不是平均分。Top8 的教训是“打 generalist 场 = 打中等飞行员”。
6. **对手自我更新的工程缺口**：当前 `load_frozen_bank` 只在启动时；若未来让对手随训练更新，需要 mid-run bank swap 或独立 opponent trainer。V21 不做；C2 严格 frozen C1/MoE。
7. **C1 不要加 ref-KL**：C1 从零训练，没有自然 reference；ref-KL 只用于 C2 warm-start。ref-KL 的系数尺度在 native loss 下与 Top4 的 5e-4 不可直接比（§8.2/8.3），C2 要重新 sweep。
8. **优化器 split 的工程细节**：AdamW state 的 checkpoint 格式、NCCL 梯度、`max_grad_norm` 的全局裁剪、`--optimizer-split off` 的逐位一致、warm-start 无 AdamW state 的 transient。A0 单测逐项过。
9. **C2 的 family 级评估缺自动化**：当前 gate 只有 arena 平均；V21 应加 per-family win-rate 报告（deck matrix + anchor deck WR），否则无法判断 MoE 是否真的提升了弱 family。
10. **预算/排期分离**：plain C2 100M aggregate；MoE 训练另算（per-family experts × games）；C1 重训（若新池）1B pilot + 2B full。不要把三者塞进一个 gate。
11. **“旧池扩充”语义**：旧 616 是 C1 池（扩充过）；新 1369 是 famfix2（coverage 83%）。两者不是同一分布。V21 baseline 用旧池；新池只在 C1 适配后使用。不要让 C2 的 frozen C1 直接面对新池（OOD）。

### 9.3 建议的 V21 顺序（对齐用户）

- **A0（无 C1 重训）**：优化器 split smoke（旧 616、C1 热启动、WALLY 固定、league 1.0、10 epoch；lr {1e-5,1e-4} × aux lr {1e-4,3e-4}）；成功判据 ent ≤0.5、lgw ≥0.65。
- **A1**：plain C2（旧 616、frozen C1、uniform 对手、无 mirror；100M aggregate；ref-KL 0.05–0.5 + entropy anneal；50M gate）。
- **A2**：plain C2 评估（arena + wally exp + per-family matrix）。
- **B0**：family taxonomy 定稿（16–24 expert / 49–60 family / 105 variant；Jaccard 聚类 + 人工语义校验）。
- **B1**：per-family C1 experts（或 family-conditioned 单模型）训练 + 每 family 质量评估；作为 C2 的 frozen opponent。
- **B2**：C2 with MoE opponent（旧 616、learner WALLY 固定、对手 family MoE 冻结；100M；per-family 评估）。
- **B3**：MoE A/B（vs frozen C1）决定是否保留；真 MoE 联合训练留到 V22。
- **C1 重训/新 1369 池**：只在决定上 1369 池时做；与 A/B 解耦。

### 9.4 待用户拍板

1. 池：V21 baseline 用旧 616（推荐）还是新 1369（需 C1 重训）？
2. family 粒度：expert 16–24 / family 49–60 / variant 105（推荐两级）；MoE 的 expert 数最终取多少？
3. C1：只换 family 标签不重训（推荐）；还是借 V21 一起用 hybrid 优化器重训？
4. MoE 形态：per-family C1 experts + router（推荐，工程小）还是真 MoE（多 expert 联合训练，工程大）？
5. C2 对手采样：baseline uniform（推荐）；family-balanced / weakness boost 作为后续 A/B？
6. C2 成功后是否要“对手自我更新”（需要 mid-run bank swap，V22）？

## 10. 用户 2026-11-19 决策与修订（覆盖 §9.4）

### 10.1 决策

1. **V21 用新 1369 池 + C1 重训**。旧 C1 只作 control；Muon 的影响由 C1 重训（hybrid 优化器）一起补偿。family 用用户新修改的划分（不是旧 105 前缀）。
2. **MoE 门先固定按 family 划分**；每个 expert 的 policy/value 网络独立；固定门有效后再考虑自由/学习门。
3. **C2 对手采样与 C1 保持一致**，只改 family 划分；不是 uniform 新池。C2f 的 uniform 是 bug，不是设计。
4. **C2 基线成功后再做 MoE 实验**；根据 MoE 结果再讨论对手是否自我更新（V22）。

### 10.2 随之修订的顺序

- **A0 诊断（不重训 C1）**：优化器 split smoke（旧 C1/旧 616，10 epoch）——只为快速验证 split，不作为 V21 生产基线。
- **B1/B2 新 C1**：新 1369 池 + 用户 family 采样 + hybrid 优化器；1B pilot（A/B 旧 Muon vs hybrid）→ 2B full；**重新测 arena/battery/matrix 基线**（旧 0.6502/0.5687/0.7717 不能直接套新 C1）。
- **A1 plain C2**：新 C1 热启动；learner WALLY 固定；对手 = 冻结新 C1 + **与 C1 相同的 family 采样**；无 mirror；100M aggregate；ref-KL/entropy 约束按 §8 的教训设置。
- **A2 评估**：arena + wally exp + per-family matrix；用新 C1 的 baseline 判负。
- **B0 family 定稿**：对用户新 family 划分做 §8.6 同款 Jaccard/大小/覆盖率审计（见 10.3）。
- **B3 MoE**：固定 family 门 + 独立 policy/value expert；作为 C2 的 frozen opponent；per-family 评估。
- **B4**：固定门有效后再考虑自由门；对手自我更新留到 V22。

### 10.3 新 family 划分的审计要求（用用户文件）

需要用户提供新 family 划分文件（deck id -> family），审计项：
1. **互斥且全覆盖**：每个 deck 恰好一个 family；无 UNKNOWN/空。
2. **大小分布**：<5/<10/<20 副的 family 数；单副族。固定门 + 独立 expert 时，小 family 的 expert 会数据饥饿；建议把 <N 副的族并入 shared/other，或做 **两级固定门**（coarse expert 16–24 → fine family conditioning）。
3. **族内一致性**：族内平均 Jaccard；若某个 family 内 Jaccard 低（<0.5），说明混了不同 archetype，应拆；若跨族 Jaccard≥0.9 多，说明该合并。
4. **tier/coverage 构成**：新池 coverage 占 83%；若 C1 采样是 tier 60% real，则 family 划分要与 tier 交叉检查；若 C1 采样是 family-balanced，则每个 family 的权重相等，coverage 族会主导 expert 训练（20/26 族），real 族 expert 数据少。
5. **与 C1 采样规则一致**：C2 继承 C1 的采样；family 划分改动只应在“family-balanced”采样下改变分布；若 C1 用 tier 权重，family 重标不改变 tier 概率，但会改变 family 内的分配。

> 结论：**新 family 划分是否合理，要看它是否满足“互斥/全覆盖 + 每 expert 有足够 deck 多样性 + 族内 Jaccard 一致”**。当前 famfix2 的 26 族语义上比旧 105 前缀好，但 12/26 族 <20 副、coverage 83%；如果用户的新划分更细，小族 expert 的数据饥饿风险更高，需要用 shared/other 或两级门兜底。

### 10.4 固定门 + 独立 policy/value expert 的架构补充（重要）

1. **per-row family 输入**：C2 的对手 deck 每局随机，expert 的 policy/value 必须按**该行 actor 的 deck family** 选择。env 已知双方 deck，需在 obs 里 stamp 一个 **per-row family id**（不是只给 learner 的 WALLY，也不是只给 opponent 的 tag）。否则：
   - 对手行的 value 会用 learner 的 WALLY expert 价值，而不是对手 family expert 的价值 → GAE 在跨越对手决策时估值错误；
   - 现有“primary critic prices every row”只在“对手=同一个 C1”时成立；MoE 对手下不成立。
2. **GAE/value 路径**：独立 expert 的 value 必须对 frozen 行也生效（对手 family 的 value）；当前 `ptcg_puff_advantage` 只用 primary 的 values，需要 per-row family 条件化后 primary 的 value 才是对手 expert 的 value。否则要么 merge frozen value（改“values 不合并”的现有约定），要么确保 primary 的 per-row family 输入正确。
3. **ref-KL 的 reference family**：ref-KL 必须锚到 **learner 的 family（WALLY）**，不是对手的随机 family。现有 `ref_forward_bank` 用 minibatch obs；per-row family 输入后，learner 行是 WALLY，frozen 行是对手 family；ref-KL 只取 learner 行，OK。实现时要确认 reference bank 的 family 列与 learner 行一致。
4. **单 MoE 模型 vs 多 bank**：固定门 + 独立 expert 建议做成**单个 MoE 模型**（family id 输入 + 固定 gate + 每 expert 独立 policy/value），C2 learner 也用同一架构（family=WALLY），这样 frozen bank 的 `PT_PARAMS` 才能加载；避免 §9.2 的 row-static/16-bank 两个 blocker。代价：checkpoint/state/NCCL/optimizer 要支持 MoE 参数；learner 只更新 WALLY expert（policy masked 的其它 expert 只受 value loss/weight decay 影响，需确认是否符合预期）。
5. **expert 数据**：固定门要求每个 expert 的 family 有足够 deck 多样性；小 family 并入 shared/other，或两级门（coarse expert + fine family conditioning）；否则 expert 会过拟合少数 deck。

### 10.5 C1/C2 采样：之前是什么、现在有哪些选项（具体数字）

**先纠正事实：native v19_c1 的“之前设定”是 uniform，不是 family 采样。**

- 旧 C1（`native-v19_c1`）：`--deck-pool decks/pool` 616 副；**没有 `--deck-weights-file`** → 每副 1/616 = **0.1623%**；`--mix-self 0.98`（98% mirror self-play，2% random-legal opponent）；6 个 anchor 只做遥测（`--p-anchor-*` 全 0）；random-legal 对手的座位还会从 87 副 coverage alt 池抽；训练池仍是 616。
- “参考 top 文档后 C1 按 family 采样”是**目标/另一套 ptcg_bc V18 的 `family_wr`**，不是 native C1 的实际运行。famfix2 报告 `family_fix2_20261119.md` 的 C2 用法是 `PPO_DECK_SAMPLING=family_wr`（ptcg_bc trainer），native trainer 没有 `family_wr`。
- native trainer 只有两种采样：**uniform** 或 `--deck-weights-file`（静态 per-deck 权重，归一化 mean 1）。所以 famfix2 的 family 采样要在 native 落地，只能用静态 weights 文件近似。

**选项与具体分布**（新池 1369 = JSON 1367 + WALLY + r4d；25 family + 2 manual）：

1. **Uniform（旧规则）**：每副 1/1369 = **0.0730%**。tier 构成 = real 4.38%（60 副，含 2 manual）/ coverage 82.8%（1134）/ patch 2.92% / archetype 2.34% / self_combo 1.02% / weakness 2.48% / anti_ex 1.90% / benchdamage 2.05% / submission 0.073%。
   - 优点：完全沿用旧 C1 的规则；实现零改动。
   - 缺点：新池 uniform 是 **83% coverage**；real meta 只有 4.4%；C1/C2 会被 coverage 主导，真实场（alakazam/dragapult/blaziken 类）覆盖不足。C2f 的 0.982 是“旧 C1 OOD + uniform 新池”，新 C1 在同一 uniform 上重训会 in-distribution，但仍是 coverage-heavy，C1 的 arena 对真实 meta 可能偏弱。
2. **Tier `sample_weight`（JSON 自带）**：每副概率 = `tier_prob / tier_count`：
   - real 60% / 58 副 → 每副 **1.0345%**；
   - coverage 20% / 1134 → 每副 **0.01764%**；
   - patch 6% / 40 → 0.150%；archetype 5% / 32 → 0.156%；self_combo 3% / 14 → 0.214%；weakness 3% / 34 → 0.0882%；anti_ex 1.5% / 26 → 0.0577%；benchdamage 1.5% / 28 → 0.0536%；submission 0。
   - 优点：字段像真实场（real 60%）；small tiers 有最低配额；C1/C2 的 arena 对 meta 更有保障。
   - 缺点：coverage 只 20%，coverage family 的 MoE expert 数据更少；C2 的固定 WALLY（real）对 real 60% 的对手，初始 lgw 可能偏低（我们用旧 C1 的 OOD 探针是 0.008，但新 C1 在同一 tier 上重训后不再 OOD，需实测）。
3. **Family-equal（报告 `family_wr` 的静态近似）**：25 family 各 **4.00%**，族内 uniform；每副 = 4%/族大小：
   - cov_dragon_box 167 → 0.0240%；cov_mega_venusaur 132 → 0.0303%；cov_other 127 → 0.0315%；
   - dragapult_spread 37 → 0.1081%；abra 37 → 0.1081%；excadrill 36 → 0.1111%；
   - ogerpon 17 → 0.2353%；crustle 16 → 0.2500%；grookey 14 → 0.2857%；lopunny 13 → 0.3077%；
   - benchdamage-grimmsnarl 12 → 0.3333%；antiex-crustle 10 → 0.4000%；benchdamage-munkidori 7 → 0.5714%；tech_toolbox 6 → **0.6667%**。
   - family 份额：coverage 10/25 = 40%，其余 15 family = 60%（real 族在其余里，按族大小分）。
   - 优点：每个 family 必定出现；小 family 每副概率高（expert 数据够）；报告的 C2 用法就是这个族等概率表。
   - 缺点：忽略真实场份额；coverage family 仍占 40%；C1/C2 的 meta 场覆盖取决于 15 个非 cov family；固定 WALLY 对 family-equal 的初始 lgw 需实测。
4. **`family_wr` 动态（ptcg_bc）**：按 family 的 win-rate/field 概率，不是静态；native 没有这个 sampler。若一定要动态，需要给 native 加 sampler 或改用 ptcg_bc trainer。

**用户确认（2026-11-19）：real 60% + 其余 40% family-equal 的混合采样。**

- **60% 质量**：real tier 58 副 + WALLY + r4d = **60 副，每副 1.000%**（mean-1 weights 13.69）。
- **40% 质量**：非 real 的 **1309 副按 25 family 等分（每 family 1.6%）**，族内 uniform；real 副不参与这 40%（它们在 60% 里）。
- **每 family 总质量** = 该 family real 数/60 × 60% + 1.6%（有 non-real 时）：
  - cov_dragon_box 1.60%；cov_mega_venusaur/Vaporeon/azumarill/emboar/probopass/slaking/pikachu/mawile 各 1.60%；cov_other 3.60%（2 real）；
  - dragapult_spread 16.60%（15 real）；abra_kadabra_mill 18.60%（17 real）；excadrill 6.60%（5 real）；shuppet 5.60%（4 real）；
  - crustle 5.60%（4 real，含 r4d）；ogerpon 6.60%（5 real）；grookey 4.60%（3 real）；lopunny 4.60%（3 real，含 WALLY）；
  - benchdamage-grimmsnarl / antiex-crustle / antiex-sylveon / antiex-farigiraf / benchdamage-froslass / benchdamage-munkidori 各 1.60%（0 real）；tech_toolbox 3.60%（2 real）。
- **每副 non-real 概率**（族内 uniform）：cov_dragon_box 0.0096%；cov_mega_venusaur/Vaporeon 0.0121%；cov_other 0.0128%；cov_azumarill 0.0137%；cov_mega_emboar 0.0151%；cov_probopass 0.0168%；cov_slaking 0.0176%；cov_pikachu 0.0211%；cov_mega_mawile 0.0242%；dragapult 0.0727%；abra 0.0800%；excadrill 0.0516%；shuppet 0.0696%；crustle 0.1231%；ogerpon 0.1333%；grookey/lopunny 0.1455%；benchdamage-grimmsnarl 0.1333%；antiex-crustle/sylveon 0.1600%；antiex-farigiraf/benchdamage-froslass 0.2000%；benchdamage-munkidori 0.2286%；tech_toolbox 0.4000%。
- **coverage = 合成/构造的 coverage 卡组**（1134 副，`cov_*` 10 family + 其它 family 里的 coverage 副）；real = 真实 meta 牌表（58 + WALLY + r4d）；其余 40% 覆盖 coverage + patch/archetype/self_combo/weakness/anti_ex/benchdamage/submission 的**非 real** 副。
- **submission `v11_r055`**（tier `submission`，`sample_weight` 0）在混合下按 crustle family 的 non-real **0.1231%** 采样；如果应并入 real 60% 或排除，告诉我。
- **权重文件**：per-deck weight = probability × 1369（mean 1）：real 13.69；cov_dragon_box 0.1312；cov_mega_venusaur/Vaporeon 0.1659；cov_other 0.1752；cov_azumarill 0.1872；…；non-real 按上表。WALLY/r4d 显式列入；C1/C2 共用 `--deck-weights-file`。
- **C1 2B（≈13.3M games）下每 family 对局**：cov_dragon_box 1.6%≈213k；dragapult 16.6%≈2.2M；abra 18.6%≈2.5M；tech_toolbox 3.6%≈479k；MoE expert 数据充足。
- **C2f 的 uniform 0.982 不复用**；C2 用同一混合权重；新 C1 重训后实测初始 lgw。
- 这是对报告 `family_wr`（动态 family win-rate）的**静态混合近似**；若要动态 family win-rate，需要给 native 加 sampler。
- **Tier `sample_weight` 作为 A/B**：如果 C1 的 arena 对真实 meta 不够，再切 tier（real 60%）；但 tier 和 family-equal 是两个采样规则，不要同时改。
- **Uniform 只作为旧规则对照**：新池 uniform 83% coverage，不建议作为 V21 生产。
- **weakness boost**：已被 40% family 采样覆盖（weakness/anti_ex/benchdamage 各族 1.6%）；若要额外 boost，plain C2 基线之后单独 A/B，不要混入 C1 重训。
- **2 个 manual deck 的 family**：r4d → `crustle_kangaskhan_wallstall`（16→17）；WALLY → `lopunny_froslass_tempo`（13→14，按 Jaccard 最近的 lopunny 族）或新建 `lopunny_wally` family（family-equal 下 4% 给 1 副，会过度采样；建议并入 lopunny 族）。samples weights 文件要把这两副列入，否则 `default` 地板会破坏 family 份额。
- **anchor 列表**：旧 6 个 anchor 名字不在新池（新旧池只有 WALLY/r4d 重名），`resolve_anchor_indices` 会 assert 失败；新 C1 的 `--anchor-decks` 要么用新池 real 名，要么空（只失去遥测）。C2 的 anchor 用 WALLY（新池 index 由 resolve 决定）。
- **C2f 的 uniform 0.982 不可复用**：那是旧 C1 + 新池 OOD；新 C1 在同一采样上重训后必须重新测初始 lgw。

## 11. C1 重训的 2B/早停与吞吐优化

### 11.1 2B + 早停（按用户偏好；低频评估、省磁盘）

- 旧 C1 已跑满 2B aggregate（final step 666,763,264/rank ≈ 2B/3；`state_latest` 完整），后期接近收敛。新 C1 用新池 + 新采样 + hybrid 优化器，**仍按 2B budget**，加**早停**；1B 只作为优化器 A/B 的 checkpoint，不作为最终版本。
- **评估/保存口径（按用户 2026-11-19：不要频繁，磁盘紧张）**：
  - `--checkpoint-interval 200`（约 25 个 weights ckpt，~225MB）；`STATE_HISTORY` 10→3（~54MB state）；新 C1 总工件 ~300MB。不再每 50 epoch 存 state。
  - C1 评估只在 **1B / 1.5B / 2B**（或每 1000 epoch）做一次 battery/arena；不做每 100–150 epoch。
  - `--deck-matrix-every 0`（训练内 matrix 关闭）；`PTCG_EVENT_LOG` 关闭；日志保留。
  - 早停：1.5B→2B 评估无提升（<0.3–0.5pt）即停；2B 是上限，不追 2B 之后的收益。
  - 旧 C1 的 0.6502/0.5687/0.7717 是旧池/旧 C1；新 C1 的 baseline 在 1B/1.5B/2B 的评估里重新测。
- **2B wall time**：旧 C1 11.6K SPS ≈ 48h；SPS 到 14–15K ≈ 37–40h；早停（~1.5–1.8B）再省 10–15%。
- **磁盘清理（待用户批准）**：旧 C1 1.1G 只保留 `final.bin` + `state_latest.bin`（中间 ckpt/state 可删）；失败 C2 系列（`c2f`/`c2_v4`/`c2_v3_erosion`/`c2_dbg6`/`lgw_probe*` 等，约 1.5–2G）保留日志、删 ckpt/state；`c2_minimal_done`/`c2b` 的 final 保留（C2 对照），中间删。当前 `autodl-tmp` 41G 可用，清理后更宽裕。
- **BC/依赖（回答“只重训 C1 就行吗”）**：
  - native V21 路径（C1 → plain C2 → family MoE experts）**只需要重训 C1**；C2 和 MoE experts 都从新 C1 派生，不依赖 BC。
  - V18 的 BC 是独立资产，**旧池只有 417 个 card id，新池 1428 个**；如果 V21 用 V18 BC/ptcg_bc 训练（例如 MoE 走 V18），BC 会 OOD，需要按新池重新生成/适配；native 路径不用它。
  - 新 C1 是 from scratch（native，无 BC 初始化）；旧 C1 只作 control。

### 11.2 吞吐 profiling（先测再改）

- **2026-11-19 实测（新池 + 混合 weights；3-epoch 诊断被用户取消，但 weights 已加载）**：`deck weights -> v1, 1369/1369 active, ESS 156.4, min 0.131, max 13.7`。ESS 低是因为 real 60% 是 60 副各 1%，非 real 40% 按 25 family 各 1.6%；2B（≈13.3M games）下每个 real deck ~133k games，cov_dragon_box 167 副只有 ~1.3k/deck。若 C2 初始 coverage competence 不够，60/40 是 A/B 杠杆，不是优化器。
- **先澄清 11.6K vs 15.1K 不是“C1 的 per-step 更慢”**：C1 ep_len ~150、C2-minimal ep_len ~200；按 games/s 算两者都约 75–77，SPS（decisions/s）差异主要来自短局每局固定开销（engine start/obs setup）摊薄更少。2B decisions = ~13.3M games；新池 family-equal 的 real 局更长，SPS 可能自然上升。

候选 A/B（每项单独跑 3–5 epoch，记录 SPS/VRAM）：

1. `--cudagraphs -1` vs `10`：C1 无 league；C2f 的 `-1` 是 league bank 的保守选择。C1 可试 cudagraphs；若 capture 干净，rollout/train 的 launch overhead 可省。
2. `--minibatch 1024` vs `2048`：minibatch_segments 16→32，optimizer step 数 128→64；GPU 利用率/launch 开销；VRAM 复核。
3. `--mix-self 0.98` vs `1.0`：去掉 2% random-legal 的 CPU 回答/alt 池；`win_vs_rand` 改由单独 eval 提供。注意 2% random 也是 health telemetry 的来源。
4. `--deck-matrix-every 0` vs 20000：旧 C1 没有 deckmat；若早停要用训练内 matrix，可低频开；否则关。
5. `--num-threads` 16/32/48：机器 208 核，3 rank × 32 = 96；CPU 线程与 GPU 的配比；看 profile 的 eval_env/rollout 占比。
6. `--total-agents 2048` vs 3072：旧 C1 38.4G/49G，有约 10G 余量；rollout/obs buffer 放大；若 OOM 回退。
7. `--num-buffers 2` vs 4：buffer 并行；VRAM/线程。
8. `PTCG_EVENT_LOG` 关闭（若开）；`--profile` 打开看 rollout/train/eval 的分解。
9. `--checkpoint-interval` / state 保存：每 50 epoch 的 15MB state 保存，影响很小，但可调。

- 建议先跑 `--profile` 的 3–5 epoch，再决定前 3 项；不要把“2B”与“吞吐优化”同时作为结论，先固定配置测 SPS。
- 新 C1 的 2B 仍按用户偏好；早停是止损，不是把预算改成 1B。

### 11.3 C2 固定卡组选择（按用户决策）

- C2 的固定卡组**不再锁定 WALLY**；等新 C1 的评测（battery / matrix / arena）出来后，从新 C1 的 deck matrix/关键 matchup 里选：
  - 选一个在新 C1 评测里**有代表性/有提升空间**的卡组（不是只看平均最強；要看对 meta 的 matchup 和 C2 想验证的 family）；
  - 该卡组必须在 C2 训练池里（1369 + 2 manual），并且能作为 `--anchor-decks` 解析；
  - 该卡组的 family 决定 C2 learner 的 expert family（MoE 阶段）和 ref-KL 的 reference family；
  - WALLY 只是旧 C1 的候选之一，保留为对照。
- C2 的固定卡组选择要在 plain C2 发车前定；不要用旧 C1 的 wally exp 0.8459 直接套新 C1。

### 11.4 新 C1 的 anchor/weights 文件落地细节

- 新池 `--anchor-decks`：旧 6 个 anchor 名不在新池（仅 WALLY/r4d 重名），必须换新池 real 名或传空；`resolve_anchor_indices` 会在启动时 assert。
- `--deck-weights-file`：按 §10.5 的选项生成 1369 行；WALLY/r4d 必须显式给权重（否则 `default` 地板破坏 family 份额）；`weight_vector` 会拒绝任何不认识的 deck 名。
- `--deck-pool decks/pool`（1369）+ `PTCG_TABLES=native/ptcg_tables.bin`（1369 blob）；旧 `native/ptcg_tables_616.bin` 只用于旧池对照。
- C2 的 `--deck-pool`/`PTCG_TABLES`/weights 文件与 C1 完全相同；C2 的 anchor 用选定的固定卡组名。

## 12. V21 执行记录（2026-11-19 晚）

### 12.0 BC 线挂起（用户指令）

BC 权重无法直接初始化 native trainer：`v17_bc.pt` 是旧架构（key 与 PT_PARAMS 0 重合）；`v18bc*` 是 ptcg_bc V18 栈的 npz 分片，需专门转换器，且为旧 417-id 分布、对新 1369 池 OOD（与 §11.1 一致）。按用户指令挂起，不投工程。

### 12.1 optimizer split：实现完成，A0 第一轮未过闸（aux 量级）

- 实现/验证/审查：`BUILD_LOG.md` Step 23.1/23.2/23.4、`docs/v21_split_review_20261119.md`（0 P0）。`--optimizer-split` 默认 off 逐位一致；AdamW 单测 PASS；PTCGSTA2 state 兼容；SPS -0.5%。
- A0 第一轮（§8.5 #2 网格 lr{1e-4,1e-5}×aux{1e-4,3e-4}）：四臂 e10 lgw 0.020/0.039/0.022/0.035、ent 0.768–0.806，全部 FAIL 且劣于不分组对照（0.131/0.679）。e1 全部健康（lgw 0.759–0.859）。
- 根因 = 扫描网格量级设计错误：AdamW 每元素定步长无 Muon 的 sqrt(R/C) 谱缩放，aux 3e-4 ≈ 旧头步长 ~48×；lr 1e-5 臂与 1e-4 臂崩速相同 → 头部移动由 aux 主导、与 trunk lr 无关。**第二轮 aux {3e-6,1e-5,3e-5}（Muon:aux 33–100×）待拍板**；另欠 split+cudagraphs capture smoke（审查 P1-3）。
- 对 §3.4 的修正认知：A0（热启动诊断）与 B1（from-scratch 重训）是两个 regime；若用户假设"C1 被旧优化器训坏"，则 B1 的 from-scratch A/B 才是直接检验，A0 只校准 C2 阶段的 aux 尺度。

### 12.2 B0 family 审计：PASS，可定稿（详见 BUILD_LOG 23.5）

- 互斥全覆盖 ✓（1369/1369）；25 族无单副、无 <5，<10 共 4 族；族内 Jaccard 22/25 ≥0.5，旗标 3 族（excadrill 0.362 混原型疑似、ogerpon 0.496 边缘、tech_toolbox 0.303 按定义混装）；跨族 J≥0.9 仅 2 对（旧池 418 对）；混合采样逐族质量与 §10.5 零偏差、real 60.00%。
- **拍板结果（2026-11-19 用户确认）**：① excadrill **不拆**——36 副呈"22 副 0.6-连通紧核（含全部 5 real）+ 14 副松散 coverage 卫星"结构，与 cov_* 族同构；拆分会制造 14 个 1–3 副小族（§8.6 数据饥饿坑）；tech_toolbox 按定义混装可接受、ogerpon 边缘无碍。② anchor 名单采纳提案：WALLY（`dudunsparce_mega_lopunny_ex_wally_s_compassion__02`）+ r4d_003（abra）/ r4d_004（dragapult）/ r4d_002（excadrill）/ r4d_000（shuppet）/ r4d_001（grookey）。**数据侧定稿**：family 25 族维持 famfix2 v2、`deck_weights_v21_hybrid.req` v1 不变、`native/ptcg_tables.bin`（1369 blob）不变。
