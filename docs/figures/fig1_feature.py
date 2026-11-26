"""Figure 1 — 编码：数值通道 + 文本通道（v3，按“两条通道”重画）。

一条是观测编码：引擎快照 → C 编码器 → 一行 12,992 个 fp32（全是 id 与数值）。
另一条是文本嵌入编码：卡面文字 → text-embedding-3-large → PCA → 静态卡表。
两条通道在 token_proj 处汇合（324 → 128）。牌桌与卡面用了真实卡图。
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pkit as P

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "fig1_feature.png")
ASSETS = os.path.join(HERE, "assets")

TOKF = [
    ("计数 · 血量", 0, 4, 5, "teal", "COPIES=n/4 · HP 比例 · HP · 上限 · 伤害"),
    ("能量", 5, 17, 13, "blue", "附能张数 + 12 种能量类型直方图（各 /3）"),
    ("装备 · 状态", 18, 28, 11, "orange", "道具 / 进化深度 / 5 种异常 / 在场 / 被点名 / 已选"),
    ("攻击 · 对位", 29, 43, 15, "purple", "top-4 招式 ×（能打·伤害·缺能）+ 弱点 / 抵抗 / 撤退"),
    ("持续效果", 44, 53, 10, "red", "8 个残留效果 flag（1.0 当下 / 0.5 待触发）+ 2 个本回合标记"),
    ("保留", 54, 55, 2, "gray", "恒 0"),
]


def card(ax, stem, x, y, w, z=7):
    im = Image.open(os.path.join(ASSETS, stem))
    h = w * im.height / im.width
    im = im.resize((int(w * 2), int(h * 2)), Image.LANCZOS)
    ax.imshow(np.asarray(im), extent=(x, x + w, y + h, y), zorder=z,
              interpolation="lanczos")
    ax.add_patch(P.Rectangle((x, y), w, h, fill=False, edgecolor="#C9CFDA",
                             lw=0.6, zorder=z + 1))
    return h


def head(ax, x, y, num, text, accent):
    ax.add_patch(P.Circle((x + 10, y - 4), 10, facecolor=P.A[accent], zorder=7))
    P.t(ax, x + 10, y - 3.5, str(num), fs=10, c="white", bold=True, z=8)
    P.t(ax, x + 30, y, text, fs=11.4, c=P.NAVY, bold=True, ha="left", z=8)


def build():
    fig, ax = P.new_canvas()
    P.title_block(
        ax, "编码：一条数值通道，一条文本通道，在模型入口汇合",
        "观测行 12,992 个 fp32 里只有 id 和状态数字；卡面文字离线走 embedding + PCA，变成 1448 行的卡表。"
        "两边在 token_proj 拼成 324 → 128。",
        ["obs 12,992 · fp32 · VERSION 6", "buffers.py · encoder.c · cards.py"])

    # ============================================================ ① 数值通道
    head(ax, 40, 96, 1, "数值通道：引擎快照 → 一行 12,992（每步现算）", "teal")
    P.group(ax, 40, 112, 380, 176, title="牌桌（真实卡图）")
    for i, (stem, lab) in enumerate([("dragapult.png", "主动"), ("lopunny.png", "后备"),
                                     ("ultraball.png", "手牌")]):
        card(ax, stem, 56 + i * 92, 134, 72)
        P.t(ax, 92 + i * 92, 252, lab, fs=6.4, c=P.BODY)
    P.t(ax, 230, 272, "只看得见自己的手牌", fs=5.8, c=P.DIM)

    P.t(ax, 432, 146, "ID · 血量 · 区域 · 附加能量 · 状态", fs=6.8, c=P.BODY, ha="left")
    P.t(ax, 432, 166, "引擎只给“当前视角”：看不到对手手牌与牌库", fs=6.4, c=P.DIM, ha="left")
    P.t(ax, 432, 186, "发牌不可复现 —— 所以从记录里直接读 obs，不重放", fs=6.4, c=P.DIM, ha="left")
    P.t(ax, 432, 212, "tracker 吃 logs 增量：我方未见 / 奖品已知 / 对手已知手牌", fs=6.2,
        c=P.A["teal"], ha="left")
    P.t(ax, 432, 232, "没有历史帧：时序只以计数 / flag 标量进观测", fs=6.2, c=P.A["teal"],
        ha="left")
    P.t(ax, 432, 258, "C 侧直接写 pinned buffer，rollout 不跨 Python 边界", fs=6.2,
        c=P.FAINT, ha="left")

    P.node(ax, 40, 300, 750, 42, "C 编码器  encoder.c",
           "读快照 → 查静态表 → 派生特征（弱点/能量/招式）→ 写掩码 → 12,992 维", fs=8.2,
           subfs=6.2)
    blocks = [("实体 token", "160 × 61 = 9,760", "teal"),
              ("候选动作", "64 × 14 + 64 掩码", "blue"),
              ("全局 + 问题", "64 + 24", "purple"),
              ("oracle 真值", "24×2 + 24×3 + 8×2 = 136", "orange")]
    for i, (a, b, acc) in enumerate(blocks):
        P.node(ax, 40 + i * 190, 356, 180, 52, a, b, fs=7.4, subfs=5.8, accent=acc)
    segs = [("实体 token", "9,760", "teal", 0.760),
            ("候选动作", "960", "blue", 0.075),
            ("全局+问题", "88", "purple", 0.042),
            ("真值", "136", "orange", 0.038),
            ("预留", "2,048", "gray", 0.085)]
    x, y, tw, hh = 40, 430, 720, 30
    for s, v, acc, frac in segs:
        w = tw * frac
        ax.add_patch(P.Rectangle((x, y), w - 2, hh, facecolor=P.A[acc],
                                 edgecolor="none", zorder=6))
        big = frac > 0.2
        P.t(ax, x + w / 2, y + (11 if big else 12), s, fs=6.3 if big else 5.6, c="white", z=7)
        P.t(ax, x + w / 2, y + 23, v, fs=5.8 if big else 5.6, c="white", z=7)
        x += w
    P.t(ax, 770, 444, "= 12,992", fs=8.0, c=P.NAVY, bold=True, ha="left")
    P.arrow(ax, (500, 466), (500, 536), accent="teal", lw=1.1)
    ax.plot([1195, 1195, 762], [466, 502, 502], color=P.A["purple"], lw=1.1, zorder=5)
    P.arrow(ax, (762, 502), (762, 536), accent="purple", lw=1.1)

    # ============================================================ ② 文本通道
    head(ax, 830, 96, 2, "文本通道：卡面文字 → embedding → PCA → 卡表（离线，只算一次）", "purple")
    P.group(ax, 830, 112, 730, 176, title="卡面文本：1431 张卡的招式 / 特性 / 规则框")
    card(ax, "dragapult.png", 846, 132, 72)
    P.group(ax, 936, 128, 608, 140, accent="gray", lw=0.8)
    P.t(ax, 950, 146, "Jet Headbutt   70", fs=6.6, c=P.NAVY, ha="left")
    P.t(ax, 950, 166, "Phantom Dive   200 —— 在对手后备区放 6 个伤害指示物", fs=6.4,
        c=P.BODY, ha="left")
    P.t(ax, 950, 186, "Ability：只要这只宝可梦在后备区，受到的招式伤害全部防止", fs=6.4,
        c=P.BODY, ha="left")
    P.t(ax, 950, 210, "文本原样进 embedding：招式名 / 伤害 / 费用 / 效果，特性，规则框，", fs=6.0,
        c=P.DIM, ha="left")
    P.t(ax, 950, 226, "弱点 / 抵抗 / 撤退 —— 不手工写规则，让向量自己带语义", fs=6.0,
        c=P.DIM, ha="left")
    P.t(ax, 950, 250, "文本里的“防止全部伤害”这类话，最后变成主轴里的一维", fs=5.8,
        c=P.FAINT, ha="left")

    P.node(ax, 830, 300, 340, 46, "tools/embed_card_texts.py",
           "text-embedding-3-large · 3072 维 · 一次性离线", fs=7.6, subfs=5.8)
    P.node(ax, 1200, 300, 360, 46, "tools/build_embed_pca.py",
           "PCA + 白化 + 行 L2 → 降到 128 / 64 / 64", fs=7.6, subfs=5.8)
    P.arrow(ax, (1174, 323), (1196, 323), accent="purple", lw=1.2)

    P.node(ax, 830, 360, 730, 46, "文本向量：卡 128 · 招式 64 · 特性 64",
           "data/embed_pca.npz（入库；行号 = 模型表里的 index）", fs=8.0, subfs=6.0,
           accent="purple")
    P.node(ax, 830, 414, 730, 40, "打包进静态表  native/ptcg_tables*.bin",
           "PTCGTAB1 的文本段：卡表 1448 行 × 128 维 · 招式 1772 × 64 · 特性 496 × 64",
           fs=7.8, subfs=5.8, accent="purple")



    # ============================================================== ③ 汇合
    head(ax, 40, 516, 3, "汇合：两条通道拼成一个 token（324 → 128）", "navy")
    chips = [("卡向量 128", 196, "purple"), ("道具向量和 128", 186, "purple"),
             ("56 个浮点", 150, "teal"), ("zone 8", 58, "blue"), ("owner 4", 52, "blue")]
    x = 40
    for s, w, acc in chips:
        ax.add_patch(P.Rectangle((x, 540), w, 30, facecolor=P.A[acc], edgecolor="none",
                                 zorder=6))
        P.t(ax, x + w / 2, 555, s, fs=6.4, c="white", z=7)
        x += w + 5
    P.arrow(ax, (x + 4, 555), (x + 40, 555), accent="navy", lw=1.4)
    P.node(ax, x + 44, 534, 150, 42, "token_proj", "324 → 128", fs=7.8, subfs=6.2)
    P.t(ax, x + 208, 546, "紫色 = 文本通道来的向量（查表，不随局面变）", fs=6.0,
        c=P.A["purple"], ha="left")
    P.t(ax, x + 208, 562, "青色 = 数值通道来的标量（每步现算）", fs=6.0, c=P.A["teal"],
        ha="left")
    P.t(ax, 40, 592, "两条通道唯一的接触点是 card_id：观测里它只是个整数，"
        "对应的 128 维向量存在卡表里 —— 所以文本永远不会污染数值特征，数值也不用等文本。",
        fs=6.4, c=P.BODY, ha="left")

    # ============================================================ ④ token 解剖
    P.t(ax, 40, 630, "一个 token = 61 个数：5 个整数查身份，56 个浮点说形状", fs=11.0,
        c=P.NAVY, bold=True, ha="left")
    for i, s in enumerate(["card_id", "zone", "owner", "tool0", "tool1"]):
        P.node(ax, 40 + i * 84, 650, 78, 24, s, fs=6.2)
    P.t(ax, 480, 662, "→ card_id 查卡表；tool0/1 查道具卡向量并相加；zone/owner 是 8 + 4 维 one-hot",
        fs=6.2, c=P.BODY, ha="left")

    P.t(ax, 330, 686, "下标", fs=5.4, c=P.FAINT)
    P.t(ax, 382, 686, "个数", fs=5.4, c=P.FAINT)
    for i, (name, lo, hi, n, acc, desc) in enumerate(TOKF):
        yy = 702 + i * 27
        P.t(ax, 40, yy + 4, name, fs=6.6, c=P.NAVY, ha="left")
        P.dotgrid(ax, 136, yy - 2, min(n, 13), 1, size=8, gap=3.4, accent=acc)
        P.t(ax, 330, yy + 4, f"{lo}–{hi}", fs=5.6, c=P.FAINT, mono=True)
        P.t(ax, 382, yy + 4, str(n), fs=5.8, c=P.FAINT)
        P.t(ax, 420, yy + 4, desc, fs=5.9, c=P.BODY, ha="left")

    P.t(ax, 40, 872, "56 = 5 + 13 + 11 + 15 + 10 + 2　·　棋盘上的宝可梦最富，同一张手牌只带身份",
        fs=6.2, c=P.DIM, ha="left")
    P.t(ax, 40, 890, "同一区域里同 id 的卡并成 1 个 token（COPIES = n/4），所以 160 个槽够用。", fs=6.0,
        c=P.DIM, ha="left")
    P.t(ax, 40, 908, "定长 / 无序 / 可手算：偏移是编译期常量，token 是集合，除数固定。"
        "改一个字段就要 VERSION++，同时作废 checkpoint、C 编码器和静态表。", fs=6.0,
        c=P.BODY, ha="left")

    # 右下：为什么分成两条
    P.group(ax, 1000, 636, 560, 250, title="为什么非要拆成两条通道")
    notes = [("数值通道必须能“手算”", "C 里直接算、逐位可复现，parity 测试才可比；"
              "所以只放标量与整数，不放 embedding。"),
             ("文本通道只需要“算一次”", "3072 维文本向量离线跑一次就冻结，训练时只是查表，"
              "省掉推理开销。"),
             ("两边解耦", "换文本模型 = 重导一张表；换观测布局 = VERSION++。互不牵连。"),
             ("代价", "卡表静态：换 embedding 或重算 PCA 都要重导 blob；文本语义进不了局面特征。")]
    for i, (a, b) in enumerate(notes):
        yy = 664 + i * 56
        P.t(ax, 1016, yy, a, fs=7.0, c=P.NAVY, bold=True, ha="left")
        P.t(ax, 1016, yy + 14, b, fs=6.0, c=P.BODY, ha="left")

    P.t(ax, 1180, 974, "klay-curry/Pokemon_agent @ main · 2026-11-24", fs=5.8,
        c=P.FAINT, ha="right")
    P.t(ax, 40, 962, "卡图：Dragapult ex (TWM 130) · Mega Lopunny ex (ME2 84) · Ultra Ball (SV1 196)，"
        "取自 images.pokemontcg.io（仅作示意）。", fs=5.6, c=P.FAINT, ha="left")

    P.save(fig, OUT)
    return OUT


if __name__ == "__main__":
    print(build())
