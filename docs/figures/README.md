# PTCG agent — 三张示意图（对齐 `klay-curry/Pokemon_agent@main`）

三张 1600×1000、200 dpi 的图，每张同时输出 PNG 与 SVG（SVG 可直接放大 / 印刷 / 进 README）：

| # | 文件 | 讲什么 |
|---|---|---|
| 1 | `out/fig1_feature.png` / `.svg` | **两条编码通道**：数值观测（12,992）+ 文本 embedding → PCA → 卡表，在 token_proj 汇合 |
| 2 | `out/fig2_model.png` / `.svg` | 模型：卡表 → token → 共享主干 → 演员 / 盲评论家 / oracle 评论家；HL-Gauss 与参数预算 |
| 3 | `out/fig3_pipeline.png` / `.svg` | **卡组池的用法**：中间一个圆柱池，左边学习方、右边对手方，双方从池里抽卡组；下面是一局 → PPO / 评测 → 回流 |

设计系统沿用上一版草稿（`pkit.py`，白底 / 发丝灰线 / 居中节点 / 彩色折线箭头 / 编号圆标），
只改内容与版式：所有数字都能对到仓库里的某一行代码或文档。

## 三张图分别想回答什么

**图 1｜编码为什么是两条**：观测行里只有 id 和数值标量（C 编码器每步现算，可手算、可对齐），
卡面文字走另一条完全离线的通道（text-embedding-3-large 3072 → PCA 白化 → 卡 128 / 招式 64 / 特性 64），
冻结成 1448 行卡表。两条通道唯一的接触点是 `card_id`：观测里它只是个整数，向量在卡表里。

**图 2｜网络怎么搭**：d128/8 头/FFN 512/4 层，2,306,568 参数（+HL 13,107）；
演员在 64 个等价类上打分，盲评论家发布用、oracle 评论家只在训练时当 baseline，
v22 把 value 换成 51-bin HL-Gauss 直方图；MoE 5 专家头是关闭的可选分支。

**图 3｜池子怎么被两边用起来**（不含提交）：中间是圆柱形的 `decks/pool`（1500+ 副、每副 60 行、
deck id = 排序位置、采样权重、coverage 池 4096+、append-only 热增）。左右两侧各自从池里**按权重抽一副**：
学习方（策略 π、梯度只走这一侧）与对手方（98% 镜像 / league bank ≤16 / 2% 随机）。
中间的“一局”把两个座位跑完，reward 去左侧的 PPO，deck 级胜负去右侧的评测（anchor 逐副胜率、
deck 矩阵、bench、arena）；最后新权重回到 π、采样权重与新卡组回流到池。

## 相对第 3 版草稿（`G:\code\Remotion\ptcg_figs`）改了什么

上一版是照上游 release（`_poke4src`）的文档画的；仓库 main 上的代码已经和它分叉，这一版逐条对齐：

| 位置 | 草稿 | 本版（代码里的事实） |
|---|---|---|
| 图 1 结构 | 单一“编码器 → 一行” | 拆成**数值通道 / 文本通道**两条，并画出汇合（324 → 128） |
| 图 1 卡表 | 只在脚注写“1448×128” | 单独一条链路：原始文本 → 3072 维 embedding → PCA 白化 → 128/64/64 → `embed_pca.npz` → `PTCGTAB1` 文本段 |
| 图 1 token 字段 | 只有分组名 | 换成 `buffers.py` 的真实下标区间 0–4 / 5–17 / 18–28 / 29–43 / 44–53 / 54–55 |
| 图 1 oracle | 24 / 24 / 8 槽 | 标明 24×2 + 24×3 + 8×2 = 136（`ORACLE_F`） |
| 图 1 tracker | 只有“编码器 · 规则 + 追踪” | 单列 InfoTracker / EffectTracker / OracleLedger，写明“没有历史帧” |
| 图 1 卡图 | 抽象卡片图标 | 换成真实卡图：Dragapult ex / Mega Lopunny ex / Ultra Ball |
| 图 2 value 头 | 线性头回归 ±1 标量 | v22 主 run 用 **51-bin HL-Gauss** 直方图头（`PTCG_HL=1`），V = Σ p·z；`make_hl_init.py` 零成本迁移 |
| 图 2 参数 | 只有一个 2,306,568 | 同值已按 `PT_PARAMS` 逐项核对；新增 +HL 13,107 → 2,319,675，MoE 可选 +1,476,490 |
| 图 2 参数构成 | 无 | 新增按模块的预算条（主干 884,752 / 静态卡表 420,916 / 辅助头 372,265 / oracle 281,729 / 演员 229,777 / 盲 V+决策 117,129） |
| 图 2 辅助头 | 只说“训练时挂” | 真实系数：`0.1 × PTCG_AUX_HAND(0.3)`、`0.1 × PTCG_AUX_PRIZE(0.3)` |
| 图 2 MoE | 无 | 单列“可选、主力关”：5 个专家头 +1,476,490 参数，v21 七小时对照无族内增益 |
| 图 3 主题 | 数据 → BC → 阶段一 → 选卡组 → 阶段二 → 提交 | 换成**池子为中心的对称布局**：圆柱池在中间，左学习 / 右对手各抽一副，下接一局 → PPO / 评测 → 回流（去掉提交） |
| 图 3 采样 | “98% 镜像 + 2% 随机”一句 | 拆成四步：对手种类 → 座位抽牌 → 权重 → anchor 覆盖，并说明 coverage 池只喂随机对手 |
| 图 3 id 规则 | 无 | deck id = sorted glob 位置；`PT_DECK_ALT_BASE = 4096`；改名/插队 = 全体重编号 |
| 图 3 统计 | “macro / nash / floor 三个统计量” | 仓库里只有 `macro_mean`、逐格 win_rate、anchor 的 `deck_wr_<name>`；nash 无实现，删掉 |
| 图 3 回流 | 无 | 权重请求（每 20 epoch）/ 热增池（append-only + 版本 allreduce + zz 前缀）/ 生成端在 decklab |

## 事实出处（想核某句话时按这张表找）

| 图里的说法 | 出处 |
|---|---|
| 一行 12,992、token 61、选项 64、oracle 136、预留 2,048 | `ptcg/rl/buffers.py`（`VERSION = 6`、`OFFSETS`、`ORACLE_F`） |
| token 56 个浮点的下标含义 | `ptcg/rl/buffers.py` 的 `TokF` / `OptF` / `GlobF` |
| zone 14 类、词表 1448 / 1772 / 496 | `buffers.py` 的 `Zone` / `N_CARDS` / `N_ATTACKS` / `N_SKILLS` |
| 编码器步骤、oracle 写块、mask、pinned buffer | `native/ptcg/encoder.c`、`obs.c`、`oracle.c`、`tables.c`、`docs/training.md` §1 |
| tracker 三件套 | `native/ptcg/tracker.c`、`ptcg/tracker.py`、`ptcg/effects.py` |
| 文本 → embedding → PCA → 128/64/64 | `tools/embed_card_texts.py`、`tools/build_embed_pca.py`、`data/README.md` |
| 文本段进 blob | `native/tools/export_tables.py`（`PTCGTAB1`，15 段） |
| d 128 / 8 头 / FFN 512 / 4 层、各层维度、参数量 | `native/tools/native_weights.py`（`D, HEADS, FFN, LAYERS, DC, DA` + `param_order()`）、`native/model/ptcg_model.cu` |
| HL-Gauss 51 bin、软标签、迁移 | `native/model/ptcg_model2.cu`（`PT_VBINS`）、`ptcg/rl/model.py`、`native/tools/make_hl_init.py` |
| 辅助头系数 | `ptcg_model.cu` 的 `AUX_*_COEF` × `PTCG_AUX_HAND/PRIZE` |
| MoE 5 专家头 | `native_weights.py`（`moe.*`）、`docs/design_moe5_v1_20261121.md`、`docs/moe5_7h_report_20261122.md` |
| 每副 60 行、deck id = 排序位置、4096 基址 | `docs/training.md` §2、`docs/deck-pool.md` §1–2、`native/ptcg/ptcg_env.h`（`PT_DECK_ALT_BASE`） |
| 采样（对手种类 / 座位 / coverage / anchor） | `native/ptcg/env.c` 的 `game_start`、`docs/deck-pool.md` §4、`native/train_native.py`（`resolve_anchor_indices`） |
| 权重请求格式与 0 权重语义 | `docs/training.md` §3、`docs/deck-pool.md` §6、`read_weight_request` / `weight_vector` |
| 热增池（append-only / 版本 allreduce / zz 前缀） | `docs/training.md` §3、`docs/deck-pool.md` §5、`native/ptcg/tables.c`（`pt_reload_decks`） |
| anchor 逐副胜率、deck 矩阵 | `native/train_native.py`（`deck_wr_` / `PTCG_DECK_MATRIX`）、`docs/training.md` §11 |
| bench / macro_mean / arena / selfeval_batched | `ptcg/rl/eval_watch.py`、`ptcg/rl/arena.py`、`tools/selfeval_batched.py`、`docs/eval.md` |

## 卡图

`assets/` 里三张卡（仅作示意，来源 `images.pokemontcg.io`，版权归 Pokémon / 任天堂）：

| 文件 | 卡 | 来源 |
|---|---|---|
| `dragapult.png` | Dragapult ex（TWM 130/167） | https://images.pokemontcg.io/sv6/130_hires.png |
| `lopunny.png` | Mega Lopunny ex（ME2 084/094） | https://images.pokemontcg.io/me2/84_hires.png |
| `ultraball.png` | Ultra Ball（SV1 196/198） | https://images.pokemontcg.io/sv1/196_hires.png |

换图：把同尺寸的 PNG 放到 `assets/` 并改脚本里的文件名即可（渲染时按宽度等比缩放）。

## 重画

```powershell
cd G:\code\pokemon\docs\figures
python fig1_feature.py    # -> out/fig1_feature.png + .svg
python fig2_model.py      # -> out/fig2_model.png + .svg
python fig3_pipeline.py   # -> out/fig3_pipeline.png + .svg
```

依赖：Python 3 + matplotlib + Pillow（本机 3.14 / mpl 3.10 / Pillow 13 已验证）。
字体走系统里的 Segoe UI（拉丁）+ Microsoft YaHei（中文）+ Consolas（等宽），无需额外安装。

## 写图时留的几个待确认点

1. **训练池的实际副数**：README 只说“1500+ 副、内容不入库”，图里不写具体数字。
2. **`docs/deck-pool.md` 在 main 上 404**：机制描述取自上游同名文档，并通过本仓库
   `ptcg_env.h`（`PT_DECK_ALT_BASE = 4096`）、`tables.c`、`train_native.py` 的常量/函数名核对。
3. **权重生成端**：仓库里只有消费端（`read_weight_request` / `weight_vector`），
   把矩阵变成权重的 decklab 在训练机本地，图里按“仓库外”标注。
4. **上游 README 说布局“v4 冻结”，代码 `VERSION = 6`**：v6 只是尾部追加两个全零预留块（偏移不变），
   图 1 标注 `VERSION 6`。
