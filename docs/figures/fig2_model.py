"""Figure 2 — 模型：卡表 → token → 共享主干 → 演员 / 评论家（v2，对齐 Pokemon_agent@main）。

相对草稿的更新：
- 参数量按 native/tools/native_weights.py 的 PT_PARAMS 精确拆分（2,306,568）；
- value 头补上 v22 的 HL-Gauss 51-bin 直方图头（+13,107）；make_hl_init 的迁移写清楚；
- 辅助头给出实际系数（AUX 0.1 × env 0.3）；
- MoE 5 专家头作为「可选、主力关」的分支单独标出，并给参数开销。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pkit as P

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "fig2_model.png")

PARAMS = [("主干（4 层 + token/bcast）", 884752, 0.384),
          ("静态卡表编码", 420916, 0.182),
          ("辅助头（训练时）", 372265, 0.161),
          ("Oracle 评论家", 281729, 0.122),
          ("演员头", 229777, 0.100),
          ("盲评论家 + 决策编码", 117129, 0.051)]


def attn_box(ax, x, y, w, h):
    P.group(ax, x, y, w, h, title="自注意力", lw=0.9)
    n = 5
    xs = [x + 22 + i * (w - 44) / (n - 1) for i in range(n)]
    y1, y2 = y + h - 42, y + h - 12
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            hi = (i == 1 and j == 2)
            ax.plot([xs[i], xs[j]], [y1, y2],
                    color=P.A["purple"] if hi else "#D7D2EC",
                    lw=1.3 if hi else 0.6, zorder=6)
    for i in range(n):
        ax.add_patch(P.Circle((xs[i], y1), 2.8, facecolor=P.A["blue"],
                              edgecolor="white", lw=0.6, zorder=7))
        ax.add_patch(P.Circle((xs[i], y2), 2.8, facecolor=P.A["purple"],
                              edgecolor="white", lw=0.6, zorder=7))


def hl_panel(ax, x, y, w, h):
    P.tint(ax, x, y, w, h, "green")
    P.t(ax, x + 16, y + 20, "value 头：标量回归 → 51-bin 直方图（PTCG_HL=1）",
        fs=8.8, c=P.NAVY, bold=True, ha="left")
    P.t(ax, x + 16, y + 38, "旧：线性头回归一个标量。新：在 [−1.2, +1.2] 上等距 51 格预测一个分布。",
        fs=6.2, c=P.BODY, ha="left")

    base = y + 110
    x0, x1 = x + 30, x + 296
    n = 25
    xs = np.linspace(x0, x1, n)
    z = np.linspace(-1.2, 1.2, n)
    p = np.exp(-((z - 0.30) ** 2) / (2 * 0.62 ** 2))
    p = p / p.sum()
    tgt = np.exp(-((z - 0.30) ** 2) / (2 * 0.30 ** 2))
    tgt = tgt / tgt.sum()
    scale = 58.0 / max(p.max(), tgt.max())
    hp, ht = p * scale, tgt * scale
    ax.plot([x0, x1], [base, base], color=P.HAIR, lw=0.8, zorder=4)
    for i in range(n):
        ax.add_patch(P.Rectangle((xs[i] - 4, base - hp[i]), 8, hp[i],
                                 facecolor=P.A["green"], alpha=0.45,
                                 edgecolor="none", zorder=5))
    ax.plot(xs, base - ht, color=P.A["orange"], lw=1.6, zorder=7)
    ez = float((p * z).sum())
    ex = x0 + (ez + 1.2) / 2.4 * (x1 - x0)
    ax.plot([ex, ex], [base, base - ht.max() - 4], color=P.A["red"], lw=1.0,
            linestyle=(0, (2, 2)), zorder=9)
    P.t(ax, x0 + 2, base + 13, "−1.2", fs=5.6, c=P.FAINT, ha="left")
    P.t(ax, x1, base + 13, "+1.2", fs=5.6, c=P.FAINT, ha="right")
    P.t(ax, x + 330, base - 54, "蓝柱 = 预测分布 p（51 格）", fs=5.8, c=P.A["green"],
        ha="left")
    P.t(ax, x + 330, base - 40, "橙线 = 高斯软标签 t（σ=0.02）", fs=5.8, c=P.A["orange"],
        ha="left")
    P.t(ax, x + 330, base - 26, "红线 = 期望 V = Σ p·z", fs=5.6, c=P.A["red"], ha="left")
    P.t(ax, x + 330, base - 12, "损失 = 交叉熵(p, t) + ∂E 项", fs=5.6, c=P.DIM, ha="left")

    P.t(ax, x + 16, y + h - 56, "期望 V 照旧进 GAE；策略与价值共享主干，梯度相加。",
        fs=6.0, c=P.BODY, ha="left")
    P.t(ax, x + 16, y + h - 40, "make_hl_init.py：旧标量 blob → 新布局，bins 权重 0、",
        fs=6.0, c=P.BODY, ha="left")
    P.t(ax, x + 16, y + h - 26, "bias = −centers²/2σ² —— 迁移零成本、不用重训。",
        fs=6.0, c=P.BODY, ha="left")


def build():
    fig, ax = P.new_canvas()
    P.title_block(
        ax, "模型：一条共享主干，一个演员，一个盲评论家 + 一个 oracle 评论家",
        "卡牌先编码成向量，token 互看 4 层，再对 64 个候选打分；value 在 v22 换成 51-bin HL-Gauss，"
        "辅助头与 oracle 只在训练时挂",
        ["d 128 · 8 heads · FFN 512 · 4 layers", "2,306,568 params · +13,107 (HL)"])
    P.legend(ax, 40, 84, [("teal", "卡表 / 表示"), ("blue", "token"),
                          ("purple", "共享主干"), ("green", "评论家"),
                          ("orange", "只在训练时")], fs=7.2, gap=26)

    # ------------------------------------------------------------- 卡表
    P.group(ax, 40, 104, 260, 316, title="卡表：身份 → 向量")
    P.node(ax, 58, 124, 108, 40, "招式向量", "127 → 64", fs=7.2, subfs=5.8)
    P.node(ax, 174, 124, 110, 40, "特性向量", "112 → 128", fs=7.2, subfs=5.8)
    P.node(ax, 58, 178, 226, 128, "")
    P.t(ax, 171, 196, "卡向量  634 → 128", fs=8.2, c=P.NAVY, bold=True)
    for i, (s, v) in enumerate([("id 查表（1448 行）", "64"),
                                ("静态属性（58 维）", "58"),
                                ("卡面文本 embedding（冻结）", "128")]):
        P.t(ax, 72, 218 + i * 15, s, fs=6.2, c=P.BODY, ha="left")
        P.t(ax, 272, 218 + i * 15, v, fs=5.8, c=P.FAINT, ha="right")
    P.t(ax, 171, 268, "文本：卡 128 · 招式 64 · 特性 64", fs=5.8, c=P.DIM)
    P.t(ax, 171, 284, "+ 招式均值/最大 128 + 特性均值/最大 256", fs=5.8, c=P.DIM)
    P.arrow(ax, (112, 166), (112, 178), accent="teal", lw=1.0)
    P.arrow(ax, (229, 166), (229, 178), accent="teal", lw=1.0)
    P.t(ax, 58, 322, "1448 张卡各一行，每步重建：梯度能回到 id 表；", fs=6.2, c=P.DIM, ha="left")
    P.t(ax, 58, 338, "同一张卡出现在哪里都拿同一个向量。", fs=6.2, c=P.DIM, ha="left")
    P.t(ax, 58, 360, "静态属性 = 类型 / 阶段 / ex / 血量 / 撤退 / 弱抗", fs=5.8, c=P.FAINT, ha="left")
    P.t(ax, 58, 376, "文本通道 = 3072 维 embedding → PCA，没见过的卡也有意义",
        fs=5.8, c=P.FAINT, ha="left")

    # ------------------------------------------------------------- token
    P.group(ax, 318, 104, 230, 316, title="拼成 token")
    P.node(ax, 332, 124, 202, 112, "")
    P.t(ax, 433, 142, "324 → 128", fs=8.2, c=P.NAVY, bold=True)
    for i, (s, v) in enumerate([("卡向量", "128"), ("道具 0 + 道具 1", "128"),
                                ("定长浮点 TokF", "56"), ("zone", "8"), ("owner", "4")]):
        P.t(ax, 346, 164 + i * 15, s, fs=6.2, c=P.BODY, ha="left")
        P.t(ax, 520, 164 + i * 15, v, fs=5.8, c=P.FAINT, ha="right")
    P.t(ax, 433, 246, "→ 线性 324 → 128", fs=5.6, c=P.DIM)
    P.node(ax, 332, 252, 92, 38, "全局", "64 → 128", fs=7.0, subfs=5.6)
    P.node(ax, 438, 252, 96, 38, "问题", "696 → 64", fs=7.0, subfs=5.6)
    P.t(ax, 433, 306, "两者相加 → 192 → 128，加到每个 token 的输入上", fs=5.8, c=P.DIM)
    P.t(ax, 433, 322, "（全局信息不过注意力就到 layer 0）", fs=5.6, c=P.FAINT)
    P.t(ax, 433, 346, "每步 40–90 个（上限 160）", fs=6.2, c=P.DIM)
    P.t(ax, 433, 362, "没有位置编码：token 是集合", fs=6.2, c=P.DIM)
    P.t(ax, 433, 384, "补齐到 160 后，尾部 token 用一个 keep 掩码屏蔽", fs=5.6, c=P.FAINT)
    P.arrow(ax, (300, 210), (330, 210), accent="teal", lw=1.0)
    P.arrow(ax, (378, 250), (378, 238), accent="purple", lw=1.0)
    P.arrow(ax, (486, 250), (486, 238), accent="purple", lw=1.0)

    # ------------------------------------------------------------- 主干
    P.group(ax, 566, 104, 330, 316, title="共享主干  × 4 层", accent="purple")
    for i in range(4):
        P.node(ax, 588, 124 + i * 32, 286, 25, "", accent="purple", r=5)
        P.t(ax, 602, 137 + i * 32, "LN → QKV → 注意力 → 残差", fs=6.0, c=P.BODY, ha="left")
        P.t(ax, 860, 137 + i * 32, "LN → FFN 512 → 残差", fs=5.8, c=P.DIM, ha="right")
    attn_box(ax, 588, 254, 286, 70)
    P.t(ax, 866, 266, "8 头 · 无位置编码", fs=5.6, c=P.DIM, ha="right")
    P.t(ax, 588, 342, "每个 token 依据内容决定看谁：主动牌看到对手能量，", fs=6.2, c=P.BODY, ha="left")
    P.t(ax, 588, 358, "手牌看到进化链 —— 权重 = softmax(QK^T/√d)。", fs=6.2, c=P.BODY, ha="left")
    P.t(ax, 588, 380, "同一主干同时喂演员和评论家：参数共享、梯度相加。", fs=6.2,
        c=P.A["purple"], ha="left")
    P.t(ax, 588, 396, "READOUT 作为 CLS 打头，pre-LN，fp32 + TF32 matmul。", fs=5.8,
        c=P.DIM, ha="left")
    P.arrow(ax, (548, 196), (564, 196), accent="blue", lw=1.2)

    # ------------------------------------------------------------- 演员
    P.group(ax, 914, 104, 330, 316, title="演员：给 64 个候选打分")
    P.node(ax, 930, 124, 140, 44, "候选编码 q", "408 → 128", fs=7.4, subfs=5.8)
    P.node(ax, 1088, 124, 138, 44, "打分", "576 → 256 → 1", fs=7.4, subfs=5.8)
    P.arrow(ax, (1070, 146), (1086, 146), accent="blue", lw=1.0)
    P.t(ax, 930, 186, "q = [类型 16 ‖ h[ptr1] ‖ h[ptr2] ‖ 卡 64 ‖ 招 64 ‖ 8]",
        fs=5.6, c=P.BODY, ha="left", mono=True)
    P.t(ax, 930, 200, "score = MLP([q ‖ r ‖ g ‖ d ‖ q*r])", fs=5.6, c=P.BODY,
        ha="left", mono=True)
    P.t(ax, 930, 216, "h[ptr] 指向动作对应的那个 token（指针式）", fs=5.8, c=P.DIM, ha="left")
    P.dotgrid(ax, 930, 232, 12, 2, size=9, gap=5, hi=7, on_n=18, accent="blue",
              hi_accent="orange")
    P.t(ax, 1104, 240, "屏蔽不合法 → softmax → 采 1 个", fs=6.2, c=P.BODY, ha="left")
    P.t(ax, 930, 278, "r = READOUT，g = 全局，d = 问题向量", fs=5.8, c=P.FAINT, ha="left")
    P.t(ax, 930, 298, "同名卡合成 1 类：logprob / 熵 / 比率都算在类上，", fs=5.8,
        c=P.DIM, ha="left")
    P.t(ax, 930, 312, "四张同样的卡不会把策略梯度稀释掉。", fs=5.8, c=P.DIM, ha="left")
    P.t(ax, 930, 334, "q*r 显式告诉网络「这个候选和这个局面搭不搭」。", fs=5.8,
        c=P.FAINT, ha="left")
    P.arrow(ax, (896, 146), (926, 146), accent="purple", lw=1.2)

    # ------------------------------------------------------------- 评论家
    P.group(ax, 1262, 104, 298, 316, title="评论家：发布用 + 训练用", accent="green")
    P.node(ax, 1276, 124, 270, 58, "", accent="green")
    P.t(ax, 1411, 140, "盲评论家（发布、搜索叶子用）", fs=7.4, c=P.NAVY, bold=True)
    P.t(ax, 1411, 156, "V = MLP([r ‖ g]) 256 → 256 → 1", fs=5.8, c=P.BODY, mono=True)
    P.t(ax, 1411, 172, "看不到真值；只喂策略也看得到的东西", fs=5.6, c=P.DIM)

    P.node(ax, 1276, 192, 270, 66, "", accent="orange", dashed=True)
    P.t(ax, 1411, 208, "Oracle 评论家（PPO 的 baseline）", fs=7.4, c=P.NAVY, bold=True)
    P.t(ax, 1411, 224, "56 个 oracle 槽 → 投影 134 → 1 层注意力 → V", fs=5.6, c=P.BODY)
    P.t(ax, 1411, 240, "价值能看真值，策略不能 —— 策略头到真值块没有通路", fs=5.4, c=P.DIM)

    P.node(ax, 1276, 268, 270, 56, "", accent="green", dashed=True)
    P.t(ax, 1411, 284, "HL-Gauss 51-bin 头（v22）", fs=7.4, c=P.NAVY, bold=True)
    P.t(ax, 1411, 300, "挂在 oracle 特征上：51 bins → V = Σp·z", fs=5.6, c=P.BODY)
    P.t(ax, 1411, 314, "只换输出层，主干与前向不变", fs=5.4, c=P.DIM)

    P.t(ax, 1276, 344, "两份都是线性头（不加 tanh）：回归 ±1 的胜负。", fs=5.8, c=P.DIM, ha="left")
    P.t(ax, 1276, 360, "训练：oracle-V 算优势，盲 V 用同一批回报做回归；", fs=5.8, c=P.DIM, ha="left")
    P.t(ax, 1276, 374, "发布：只带盲 V（+ HL 头可选）。", fs=5.8, c=P.DIM, ha="left")
    P.t(ax, 1276, 396, "价值看真值不是作弊：它只是把「赢没赢」估得更准。", fs=5.6,
        c=P.FAINT, ha="left")

    # trunk -> critic 路由（从主干顶绕到评论家）
    ax.plot([890, 890, 1420], [112, 92, 92], color=P.A["purple"], lw=1.1, zorder=6)
    P.arrow(ax, (1420, 92), (1420, 120), accent="purple", lw=1.1)

    # ------------------------------------------------------------- 设定
    P.t(ax, 40, 452, "关键设定", fs=11.5, c=P.NAVY, bold=True, ha="left")
    cards = [
        (40, 360, "尺寸与容量", "gray",
         ["d 128 · 8 头 · FFN 512 · 4 层", "d_c 128 · d_a 64 · G 128",
          "2,306,568 参数（9.2 MB fp32）", "HL 头 +13,107 → 2,319,675"]),
        (420, 360, "稀疏奖励", "red",
         ["只有终局：赢 +1 / 输 −1", "中间每一步 reward = 0",
          "3000 引擎步截断：没有终止旗标", "→ 截断自举，不当失败"]),
        (800, 360, "折扣与优势", "green",
         ["γ = 1.0（不折扣，胜负只看结果）", "GAE λ = 0.95",
          "negamax：读 SEAT_COL，换座位乘 −1", "优势在 minibatch 内标准化"]),
        (1180, 360, "PPO / 优化", "purple",
         ["clip 0.2 · vf-clip 0.2 · grad-clip 1.5", "ent 0.01（可退火到 0.1×）· vf 2.0",
          "lr 1e-3：Muon 管隐藏 2-D，AdamW 管嵌入/头/1-D", "KL 守卫 0.05 / 值损失守卫 0.5"]),
    ]
    for x, w, title, acc, lines in cards:
        P.group(ax, x, 466, w, 112, title=title, lw=1.0)
        for i, s in enumerate(lines):
            P.t(ax, x + 12, 494 + i * 19, s, fs=6.2, c=P.BODY, ha="left")

    # --------------------------------------------------- HL-Gauss / 参数 / 辅助
    hl_panel(ax, 40, 596, 520, 194)

    P.tint(ax, 572, 596, 372, 194, "gray")
    P.t(ax, 588, 616, "参数预算 2,306,568", fs=8.8, c=P.NAVY, bold=True, ha="left")
    P.t(ax, 588, 632, "按 PT_PARAMS 的权重表逐项加起来", fs=5.8, c=P.DIM, ha="left")
    for i, (name, v, frac) in enumerate(PARAMS):
        yy = 650 + i * 21
        P.t(ax, 588, yy, name, fs=5.8, c=P.BODY, ha="left")
        ax.add_patch(P.Rectangle((742, yy - 5), max(2, frac * 178), 7,
                                 facecolor=P.A["purple"] if "主干" in name else
                                 P.A["gray"], edgecolor="none", zorder=6))
        P.t(ax, 928, yy, f"{v:,}", fs=5.6, c=P.FAINT, ha="right", mono=True)
    P.t(ax, 588, 782, "+ MoE 5 专家头 1,476,490（PTCG_MOE=1，可选）",
        fs=5.8, c=P.A["orange"], ha="left")

    P.tint(ax, 960, 596, 600, 194, "orange")
    P.t(ax, 976, 616, "辅助头：只在训练时，目标来自同一行的 oracle 块",
        fs=8.8, c=P.NAVY, bold=True, ha="left")
    P.t(ax, 976, 640, "① aux_hand：1448 类 softplus，猜对手手里各有多少张 · L2",
        fs=6.2, c=P.BODY, ha="left")
    P.t(ax, 1000, 655, "coef = 0.1 × PTCG_AUX_HAND（主力 run 用 0.3）", fs=5.8, c=P.DIM,
        ha="left")
    P.t(ax, 976, 676, "② aux_prize：每个「我没见过的 token」一个 logit，猜它是否在奖品区 · BCE",
        fs=6.2, c=P.BODY, ha="left")
    P.t(ax, 1000, 691, "coef = 0.1 × PTCG_AUX_PRIZE（主力 run 用 0.3）", fs=5.8, c=P.DIM,
        ha="left")
    P.t(ax, 976, 712, "作用：不额外前向、不回流策略，逼主干把「对手有什么」编码出来。",
        fs=6.2, c=P.BODY, ha="left")
    P.t(ax, 976, 734, "MoE（可选，v21 实验）：5 个按卡组族路由的专家头，各带一套", fs=6.0,
        c=P.A["orange"], ha="left")
    P.t(ax, 976, 748, "score / oracle-V / q；7 小时对照无族内增益 → 主力 run 未启用。",
        fs=6.0, c=P.A["orange"], ha="left")
    P.t(ax, 976, 770, "代价是 +1.48 M 参数与一次族路由；收益没跑出来，就关掉。", fs=5.8,
        c=P.DIM, ha="left")

    # -------------------------------------------------- 训练栈 vs 推理栈
    P.group(ax, 40, 810, 960, 132)
    P.t(ax, 56, 826, "同一个网络，两份实现", fs=8.0, c=P.NAVY, bold=True, ha="left")
    P.t(ax, 56, 844, "训练：C 环境 + 手写 CUDA 前向/反向（native/model/ptcg_model{1,2,3,4}.cu ·"
        " kernels.cuh + Muon），rollout 不跨 Python 边界。", fs=6.2, c=P.BODY, ha="left")
    P.t(ax, 56, 862, "推理 / 评测：torch 版 PTCGTransformer（ptcg/rl/model.py）；权重由"
        " native_weights.py 在 .pt 与 fp32 blob 之间转换，native 侧没有随机初始化。", fs=6.2,
        c=P.BODY, ha="left")
    P.t(ax, 56, 880, "对齐靠 model_parity.py / stage_parity.py：同一 batch 比 logits、V 与 KL，"
        "不一致就按 forward stage 二分定位。", fs=6.2, c=P.BODY, ha="left")
    P.t(ax, 56, 898, "所以：训练循环里没有 PyTorch，PyTorch 只在启动、导出、评测三处出现。",
        fs=6.4, c=P.A["purple"], ha="left")
    P.t(ax, 56, 916, "torch 版默认 d 256 只是参考实现的默认值；实际训练与导出都是 d 128"
        "（native_weights.py 的 D/HEADS/FFN/LAYERS）。", fs=5.8, c=P.DIM, ha="left")

    P.group(ax, 1020, 810, 540, 132)
    P.t(ax, 1036, 826, "一次前向吐出什么", fs=8.0, c=P.NAVY, bold=True, ha="left")
    P.t(ax, 1036, 844, "64 个 logits（已按等价类掩码）＋ 价值 V", fs=6.2, c=P.BODY, ha="left")
    P.t(ax, 1036, 862, "训练：采样动作 · logprob · 熵 · V · 两个辅助预测", fs=6.2, c=P.BODY, ha="left")
    P.t(ax, 1036, 880, "评测：temp=0 就是 argmax；V 给搜索做叶子评估", fs=6.2, c=P.BODY, ha="left")
    P.t(ax, 1036, 898, "分布是离散 64 类，不是高斯策略。", fs=6.2, c=P.DIM, ha="left")
    P.t(ax, 1036, 916, "masked softmax 定义在等价类上，四张同样的卡 = 一类。", fs=5.8,
        c=P.FAINT, ha="left")

    P.t(ax, 1180, 974, "klay-curry/Pokemon_agent @ main · 2026-11-24", fs=5.8,
        c=P.FAINT, ha="right")

    P.save(fig, OUT)
    return OUT


if __name__ == "__main__":
    print(build())
