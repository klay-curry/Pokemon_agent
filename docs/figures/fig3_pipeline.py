"""Figure 3 — 一个卡组池，两个座位，一套权重（v4）。

按反馈重画：中间是一个圆柱形的卡组池，左边学习方、右边对手方，双方都从池里抽卡组；
中间下方是“一局”，再往下分成 PPO 更新与评测回流两条出口。
"""
import os
import sys

import numpy as np
from PIL import Image
from matplotlib.patches import Arc, Ellipse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pkit as P

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "fig3_pipeline.png")
ASSETS = os.path.join(HERE, "assets")


def card(ax, stem, x, y, w, z=7):
    im = Image.open(os.path.join(ASSETS, stem))
    h = w * im.height / im.width
    im = im.resize((int(w * 2), int(h * 2)), Image.LANCZOS)
    ax.imshow(np.asarray(im), extent=(x, x + w, y + h, y), zorder=z,
              interpolation="lanczos")
    ax.add_patch(P.Rectangle((x, y), w, h, fill=False, edgecolor="#C9CFDA",
                             lw=0.6, zorder=z + 1))
    return h


def cylinder(ax, x, y, w, h, ry=26, accent="teal", z=4):
    """数据库圆柱：池子的形状。"""
    cx = x + w / 2
    col = P.A[accent]
    ax.add_patch(P.Rectangle((x, y + ry), w, h - 2 * ry, facecolor=P.TINT[accent],
                             edgecolor="none", zorder=z))
    ax.plot([x, x], [y + ry, y + h - ry], color=col, lw=1.1, zorder=z + 2)
    ax.plot([x + w, x + w], [y + ry, y + h - ry], color=col, lw=1.1, zorder=z + 2)
    ax.add_patch(Arc((cx, y + h - ry), w, 2 * ry, theta1=180, theta2=360,
                     edgecolor=col, lw=1.1, zorder=z + 2))
    for k in (0.28, 0.56, 0.84):
        ax.add_patch(Arc((cx, y + ry + (h - 2 * ry) * k), w, 2 * ry * 0.86,
                         theta1=196, theta2=344, edgecolor=col, lw=0.6,
                         alpha=0.30, zorder=z + 2))
    ax.add_patch(Ellipse((cx, y + ry), w, 2 * ry, facecolor=P.TINT[accent],
                         edgecolor=col, lw=1.1, zorder=z + 3))
    return cx


def build():
    fig, ax = P.new_canvas()
    P.title_block(
        ax, "一个卡组池，两个座位，一套权重",
        "每局开始时，左右两侧各自从池里抽一副卡组（按采样权重）；打完一局，左侧拿到梯度、"
        "右侧逼出弱点，结果再回流成权重与新的卡组。",
        ["decks/pool · deck_weights_*.req", "PT_DECK_ALT_BASE = 4096 · 每副 60 行"])

    # ------------------------------------------------------------ 左：建池
    P.group(ax, 120, 100, 420, 220, title="建池：一副牌 = 60 行卡牌 ID")
    for i, stem in enumerate(["dragapult.png", "lopunny.png", "ultraball.png"]):
        card(ax, stem, 134 + i * 54, 126, 46)
    for i, s in enumerate([
            "decks/pool/*.csv，每副正好 60 行，行序无意义",
            "sorted glob 的下标就是 deck id（0..n）",
            "改名 / 插队 = 后面全体重编号 —— 所以矩阵、",
            "权重表都必须和“池的版本”绑在一起"]):
        P.t(ax, 134, 210 + i * 19, s, fs=6.2, c=P.BODY, ha="left")
    P.arrow(ax, (544, 210), (594, 210), accent="teal", lw=1.2)

    # ------------------------------------------------------------ 右：入口
    P.group(ax, 1060, 100, 420, 220, title="池的三个入口")
    ent = [("① 建池", "pool/*.csv → export_tables.py → PTCGTAB1"),
           ("② 改权重", "deck_weights_*.req：每 20 epoch 重读"),
           ("③ 热增", "--deck-pool-file：append-only，zz 前缀")]
    for i, (a, b) in enumerate(ent):
        P.t(ax, 1076, 136 + i * 34, a, fs=7.2, c=P.NAVY, bold=True, ha="left")
        P.t(ax, 1076, 150 + i * 34, b, fs=6.2, c=P.BODY, ha="left")
    P.t(ax, 1076, 246, "删除不是删文件：权重 0 = 禁用但保留 id 与历史", fs=6.0,
        c=P.A["red"], ha="left")
    P.t(ax, 1076, 264, "覆盖池 pool_alt 单独编号，只喂随机对手的座位", fs=6.0,
        c=P.DIM, ha="left")
    P.t(ax, 1076, 288, "每次加卡组都要重导 blob；没报错 = 未必生效，看日志的 vN", fs=6.0,
        c=P.FAINT, ha="left")
    P.arrow(ax, (1056, 210), (1006, 210), accent="teal", lw=1.2)

    # ------------------------------------------------------------ 池（圆柱）
    cylinder(ax, 600, 100, 400, 240, ry=30, accent="teal")
    P.t(ax, 800, 148, "卡组池  decks/pool", fs=11.0, c=P.NAVY, bold=True)
    rows = [("1500+ 副 · 每副 60 行卡牌 ID", P.BODY),
            ("deck id = 排序位置（不变量）", P.BODY),
            ("采样权重 w(deck)（可热改）", P.BODY),
            ("coverage 池：id ≥ 4096，只给随机对手", P.DIM),
            ("append-only：只追加，不删改", P.DIM)]
    for i, (s, c) in enumerate(rows):
        P.t(ax, 800, 176 + i * 24, s, fs=6.6, c=c)
    P.t(ax, 800, 372, "池子的多样性 = 训练分布的上限", fs=7.0, c=P.A["teal"], bold=True)

    # -------------------------------------------------- 两侧：学习 / 对手
    P.group(ax, 120, 400, 420, 160, title="学习方  learner（策略 π）", accent="blue")
    for i, s in enumerate(["网络：d128 · 4 层 · 2.31 M 参数", "卡组：从池里按权重抽一副",
                           "梯度只从这一侧的决策回传", "更新完立刻成为“镜像对手”"]):
        P.t(ax, 136, 434 + i * 22, s, fs=6.6, c=P.BODY, ha="left")
    P.t(ax, 136, 528, "π 是唯一被训练的实体", fs=6.2, c=P.A["blue"], bold=True, ha="left")

    P.group(ax, 1060, 400, 420, 160, title="对手方  opponent（三种来源）", accent="orange")
    for i, s in enumerate(["98% 镜像：同一份 π，不额外加载", "league bank ≤16：冻结的历史权重",
                           "2% 随机合法：唯一能抽 coverage 的座位", "各 bank 固定占 1.9% 的行"]):
        P.t(ax, 1076, 434 + i * 22, s, fs=6.6, c=P.BODY, ha="left")
    P.t(ax, 1076, 528, "对手分布决定了“练到多强”", fs=6.2, c=P.A["orange"], bold=True, ha="left")

    # 池 → 两侧（抽牌）
    P.arrow(ax, (596, 296), (520, 396), accent="teal", lw=1.3, rad=-0.12)
    P.arrow(ax, (1004, 296), (1080, 396), accent="teal", lw=1.3, rad=0.12)
    P.t(ax, 512, 352, "抽一副（按权重）", fs=6.0, c=P.A["teal"], ha="right")
    P.t(ax, 1088, 352, "抽一副（按权重）", fs=6.0, c=P.A["teal"], ha="left")

    # ------------------------------------------------------------ 一局
    P.group(ax, 620, 400, 360, 160, title="一局  env（两个座位）", accent="purple")
    for i, s in enumerate(["game_start → 两个座位各抽一副（独立）",
                           "同一环境跑两个座位，一行 obs 12,992",
                           "打到终局：赢 +1 / 输 −1",
                           "3000 步截断：自举，不当失败"]):
        P.t(ax, 636, 434 + i * 22, s, fs=6.6, c=P.BODY, ha="left")
    P.t(ax, 636, 528, "镜像局也可能抽到不同卡组", fs=6.2, c=P.A["purple"], bold=True, ha="left")
    P.arrow(ax, (542, 480), (616, 480), accent="blue", lw=1.2)
    P.arrow(ax, (1056, 480), (984, 480), accent="orange", lw=1.2)

    # ------------------------------------------------- 下排：PPO / 评测
    P.group(ax, 120, 660, 600, 150, title="PPO 更新（左侧的账）", accent="purple")
    for i, s in enumerate(["GAE：negamax，换座位时优势取负",
                           "clip 0.2 · lr 1e-3 · Muon(2-D) + AdamW(1-D)",
                           "KL 守卫 0.05 / 值损失守卫 0.5：超了跳过这次更新",
                           "每行只带 deck id，卡组内容不进梯度"]):
        P.t(ax, 136, 692 + i * 21, s, fs=6.4, c=P.BODY, ha="left")

    P.group(ax, 880, 660, 600, 150, title="评测与回流（右侧的账）", accent="orange")
    for i, s in enumerate(["anchor 逐副胜率 deck_wr_<name> / deck_n_<name>（≤16）",
                           "deck 矩阵：每副 × 每副 的局数/胜数，并行攒",
                           "bench：6 副 × 对手 × 16 局；arena 多进程打外场",
                           "结论 → 采样权重；新卡组 → 热增回池"]):
        P.t(ax, 896, 692 + i * 21, s, fs=6.4, c=P.BODY, ha="left")

    # 一局 → 两侧的账
    P.arrow(ax, (660, 564), (660, 656), accent="purple", lw=1.3)
    P.t(ax, 668, 600, "reward ±1", fs=6.0, c=P.PURPLE if False else P.A["purple"], ha="left")
    P.arrow(ax, (940, 564), (940, 656), accent="orange", lw=1.3)
    P.t(ax, 948, 600, "deck 级胜负", fs=6.0, c=P.A["orange"], ha="left")

    # PPO → 学习方（新权重，同时也是镜像）
    P.arrow(ax, (330, 656), (330, 566), accent="purple", lw=1.3)
    P.t(ax, 322, 612, "新权重", fs=6.0, c=P.A["purple"], ha="right")

    # 评测 → 池（回流）
    ax.plot([1480, 1540, 1540, 800], [742, 742, 78, 78], color=P.A["red"], lw=1.2,
            linestyle=(0, (4, 3)), zorder=5)
    P.arrow(ax, (800, 82), (800, 96), accent="red", lw=1.2, ls=(0, (4, 3)))
    ax.text(1552, 420, "采样权重 / 新卡组", rotation=90, fontsize=6.2,
            color=P.A["red"], ha="center", va="center", family="Microsoft YaHei",
            zorder=8)

    # ------------------------------------------------------------ 底部注
    P.t(ax, 120, 852, "回流有两条线：① 新权重 → π（训练的自然出口，同时也是下一轮的镜像对手）；"
        "② 采样权重 / 新卡组 → 池（改变下一轮抽到什么牌）。",
        fs=6.8, c=P.NAVY, ha="left")
    P.t(ax, 120, 878, "池和策略是解耦的：同一个池可以换策略重训，同一份策略也可以换池重训 —— "
        "deck id 是两者之间唯一的接口。", fs=6.6, c=P.BODY, ha="left")
    P.t(ax, 120, 906, "常量：每副 60 行 · PT_DECK_ALT_BASE 4096（训练池上限 = 矩阵宽度上限）· "
        "PT_MAX_ANCHORS 16 · --pool-check-every 20 epoch · --mix-self 0.98", fs=6.0,
        c=P.DIM, ha="left")
    P.t(ax, 120, 950, "所有数字取自 klay-curry/Pokemon_agent@main（2026-11-24 快照）：README.md ·"
        " docs/training.md · docs/deck-pool.md · native/train_native.py · native/ptcg/{env.c,tables.c}；"
        "卡图取自 images.pokemontcg.io（仅作示意）。", fs=5.8, c=P.FAINT, ha="left")
    P.t(ax, 1180, 972, "klay-curry/Pokemon_agent @ main", fs=5.8, c=P.FAINT, ha="right")

    P.save(fig, OUT)
    return OUT


if __name__ == "__main__":
    print(build())
