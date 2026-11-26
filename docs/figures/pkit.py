"""pkit — design system for the reference style (flat, hairline, white).

Reference vocabulary: hairline white nodes with centered labels, small caps
group titles, thin coloured elbow arrows, 4-cell vector chips, dot grids,
tinted row panels, numbered badges. No shadows, no heavy fills.
"""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, PathPatch
from matplotlib.path import Path

FIG_W, FIG_H = 16.0, 10.0
W, H = 1600, 1000
DPI = 200
PPU = W / (FIG_W * 72.0)

WHITE = "#FFFFFF"
NAVY = "#16243D"
BODY = "#41506B"
DIM = "#8A93A6"
FAINT = "#B4BCC9"
HAIR = "#E3E6EC"

A = {
    "teal": "#2F8F86",
    "blue": "#2E6FE0",
    "purple": "#7A5FCB",
    "orange": "#DD8A3B",
    "green": "#2E9A63",
    "red": "#C0473F",
    "navy": "#1D2B45",
    "gray": "#98A2B3",
}
TINT = {
    "teal": "#F1F8F7",
    "blue": "#F2F6FD",
    "purple": "#F7F4FE",
    "orange": "#FEF8F1",
    "green": "#F1F9F4",
    "red": "#FDF4F3",
    "gray": "#F7F8FA",
    "navy": "#F2F4F8",
}


def new_canvas():
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor(WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    _fonts()
    return fig, ax


def _fonts():
    for name in ("segoeui.ttf", "seguisb.ttf", "msyh.ttc", "msyhbd.ttc", "consola.ttf"):
        p = os.path.join(r"C:\Windows\Fonts", name)
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
            except Exception:
                pass
    # matplotlib >= 3.6 falls back per glyph: Latin from Segoe UI, CJK from YaHei
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["font.monospace"] = ["Consolas", "DejaVu Sans Mono"]
    plt.rcParams["axes.unicode_minus"] = False


def t(ax, x, y, s, fs=8, c=BODY, ha="center", va="center", bold=False,
      mono=False, z=6, alpha=1.0, italic=False, wrap=False):
    # pick one family that actually has the glyphs: Segoe UI for Latin,
    # YaHei when the string carries CJK (its Latin is a clean grotesque too).
    if mono and not any(ord(ch) > 0x2E80 for ch in s):
        fam = "Consolas"
    elif any(ord(ch) > 0x2E80 for ch in s):
        fam = "Microsoft YaHei"
    else:
        fam = "Segoe UI"
    ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va, zorder=z,
            family=fam, alpha=alpha,
            fontweight="semibold" if bold else "normal",
            fontstyle="italic" if italic else "normal",
            linespacing=1.35)


def node(ax, x, y, w, h, title, sub=None, fs=8.6, subfs=6.8, r=7,
         accent=None, dashed=False, z=5, fill=WHITE, bold=False, lw=1.0):
    ec = A[accent] if accent else HAIR
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw, edgecolor=ec, facecolor=fill, zorder=z,
        linestyle=(0, (3, 2)) if dashed else "solid"))
    cy = y + h / 2 - (5 if sub else 0)
    t(ax, x + w / 2, cy, title, fs=fs, c=NAVY, bold=bold, z=z + 1)
    if sub:
        t(ax, x + w / 2, y + h / 2 + 7, sub, fs=subfs, c=DIM, z=z + 1)


def group(ax, x, y, w, h, title=None, accent=None, r=10, dashed=False,
          z=4, fill=WHITE, lw=1.0, sub=None):
    ec = A[accent] if accent else HAIR
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw, edgecolor=ec, facecolor=fill, zorder=z,
        linestyle=(0, (3, 2)) if dashed else "solid"))
    if title:
        t(ax, x + w / 2, y + 12, title, fs=7.4, c=NAVY, z=z + 1)
    if sub:
        t(ax, x + w / 2, y + 25, sub, fs=6.4, c=DIM, z=z + 1)


def tint(ax, x, y, w, h, accent="gray", r=10, z=3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=0, facecolor=TINT[accent], zorder=z))


def badge(ax, x, y, num, accent="navy", d=22, fs=11.5, z=7):
    ax.add_patch(FancyBboxPatch(
        (x, y), d, d, boxstyle="round,pad=0,rounding_size=6",
        linewidth=0, facecolor=A[accent], zorder=z))
    t(ax, x + d / 2, y + d / 2 + 0.5, str(num), fs=fs, c=WHITE, bold=True, z=z + 1)


def heading(ax, x, y, text, badge_num=None, accent="navy", fs=13.5, sub=None,
            z=7):
    if badge_num is not None:
        badge(ax, x, y - 11, badge_num, accent=accent, d=22, fs=11.5)
        x += 32
    t(ax, x, y, text, fs=fs, c=NAVY, bold=True, ha="left", z=z)
    if sub:
        t(ax, x + _w(text, fs) + 12, y + 1, sub, fs=7.0, c=DIM, ha="left", z=z)


def _w(s, fs):
    em = fs * PPU
    n = 0.0
    for ch in s:
        n += 1.0 if ord(ch) > 0x2E80 else 0.55
    return n * em


def chipbar(ax, x, y, w=46, h=12, cells=4, accent="teal", z=6, dim=None):
    """A vector representation: 3-4 adjacent cells, like the reference bars."""
    gap = 2.0
    cw = (w - gap * (cells - 1)) / cells
    col = A[accent]
    for i in range(cells):
        alpha = 1.0
        if dim is not None and i == dim:
            alpha = 0.25
        ax.add_patch(Rectangle((x + i * (cw + gap), y), cw, h,
                               facecolor=col, edgecolor="none", alpha=alpha,
                               zorder=z))


def dotgrid(ax, x, y, cols, rows, size=8, gap=5, hi=None, accent="blue",
            z=6, dim=0.22, on_n=None, hi_accent=None):
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if on_n is None:
                on = (hi is None) or (idx == hi)
            else:
                on = idx < on_n
            face = A[accent] if on else A["gray"]
            if hi is not None and idx == hi and hi_accent:
                face = A[hi_accent]
            ax.add_patch(FancyBboxPatch(
                (x + c * (size + gap), y + r * (size + gap)), size, size,
                boxstyle="round,pad=0,rounding_size=1.6", linewidth=0,
                facecolor=face, alpha=1.0 if on else dim, zorder=z))
            if hi is not None and idx == hi:
                ax.add_patch(FancyBboxPatch(
                    (x + c * (size + gap) - 2.6, y + r * (size + gap) - 2.6),
                    size + 5.2, size + 5.2,
                    boxstyle="round,pad=0,rounding_size=2.6", linewidth=1.3,
                    edgecolor=A[hi_accent or "orange"], facecolor="none",
                    zorder=z + 1))


def arrow(ax, p, q, accent="gray", lw=1.2, z=6, head=True, ls="-",
          rad=0.0, alpha=1.0, ms=8):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>" if head else "-",
        connectionstyle=f"arc3,rad={rad}", linewidth=lw, color=A[accent],
        linestyle=ls, mutation_scale=ms, shrinkA=0, shrinkB=1, zorder=z,
        alpha=alpha, capstyle="round", joinstyle="round"))


def elbow(ax, p, q, accent="gray", lw=1.2, z=6, mid=None, head=True, ls="-",
          ms=8):
    """Orthogonal connector: p -> (corner) -> q, arrowhead at q."""
    if mid is None:
        if abs(q[0] - p[0]) >= abs(q[1] - p[1]):
            mx = (p[0] + q[0]) / 2
            pts = [p, (mx, p[1]), (mx, q[1]), q]
        else:
            my = (p[1] + q[1]) / 2
            pts = [p, (p[0], my), (q[0], my), q]
    else:
        pts = [p] + list(mid) + [q]
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(PathPatch(Path([pts[0], pts[1], pts[2], pts[3]],
                                [Path.MOVETO, Path.LINETO, Path.LINETO,
                                 Path.LINETO]),
                           fill=False, edgecolor=A[accent], lw=lw,
                           linestyle=ls, zorder=z, capstyle="round",
                           joinstyle="round"))
    if head:
        arrow(ax, pts[-2], q, accent=accent, lw=lw, z=z + 1, ms=ms, ls=ls)


def card_icon(ax, x, y, w=30, h=42, accent="teal", z=6, band=True):
    """A tiny card: coloured art band, white text band, rounded frame."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=3",
                                linewidth=0.8, edgecolor="#C9CFDA",
                                facecolor=WHITE, zorder=z))
    if band:
        ax.add_patch(Rectangle((x + 2.5, y + 2.5), w - 5, h * 0.52,
                               facecolor=A[accent], alpha=0.75, zorder=z + 1))
    ax.add_patch(Rectangle((x + 4, y + h * 0.66), w - 8, 2.0,
                           facecolor="#D6DBE4", zorder=z + 2))
    ax.add_patch(Rectangle((x + 4, y + h * 0.72), w - 12, 2.0,
                           facecolor="#E4E8EF", zorder=z + 2))


def legend(ax, x, y, items, fs=7.2, gap=26):
    cx = x
    for accent, label in items:
        ax.add_patch(Rectangle((cx, y - 3.5), 7, 7, facecolor=A[accent],
                               edgecolor="none", zorder=7))
        t(ax, cx + 12, y, label, fs=fs, c=BODY, ha="left", z=7)
        cx += 12 + _w(label, fs) + gap


def title_block(ax, title, sub=None, right=None):
    t(ax, 40, 30, title, fs=15.5, c=NAVY, bold=True, ha="left", z=8)
    if sub:
        t(ax, 40, 52, sub, fs=8.0, c=DIM, ha="left", z=8)
    if right:
        for i, s in enumerate(right):
            t(ax, W - 40, 26 + i * 15, s, fs=7.0, c=DIM, ha="right", z=8)


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=WHITE)
    fig.savefig(os.path.splitext(path)[0] + ".svg", facecolor=WHITE)
    plt.close(fig)
    return path
