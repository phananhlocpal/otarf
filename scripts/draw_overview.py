#!/usr/bin/env python3
"""Figure 2 (model overview), one left-to-right flow on an aligned grid. Schematic: curves are synthetic."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle

root = Path(__file__).resolve().parents[1]
folder = root/'results/figures'; folder.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'mathtext.fontset': 'dejavusans',
                     'hatch.linewidth': 0.45, 'hatch.color': '#8d98a1'})
W, H = 5.17, 2.6                        # inches: text width of the template, so font sizes are final
XM, YM = 10.34, 5.2
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, XM), ylim=(0, YM)); ax.axis('off')
INK, LIGHT = '#1f2d3d', '#9aa5ad'
BLUE, BLUE_L, BLUE_F = '#2a78d6', '#dceaf3', '#eef5fa'
GOLD, GOLD_L, GOLD_F = '#c98a1b', '#f5e4c5', '#fbf3e4'
GRAY, CROSS = '#ececec', '#b23b3b'
FS = 7.0


def arrow(a, b, color=INK, lw=0.9, conn='arc3,rad=0'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=7, color=color, linewidth=lw,
                                 connectionstyle=conn, zorder=5, shrinkA=0, shrinkB=0))


def box(x, y, w, h, text, face, size=FS):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.1', facecolor=face,
                                edgecolor=INK, linewidth=0.8, zorder=3))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=size, color=INK, zorder=4, linespacing=1.2)


def frame(x, y, w, h, face, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.15', facecolor=face,
                                edgecolor=LIGHT, linewidth=0.7, zorder=0))
    ax.text(x+0.12, y+h-0.1, title, ha='left', va='top', fontsize=FS, color=INK, weight='bold', zorder=4)


def anchor_dot(x, y):
    ax.add_patch(Circle((x, y), 0.06, facecolor='white', edgecolor=INK, lw=0.9, zorder=6))


# ---------------- synthetic signals (one value scale K for all forecast sketches) ----------------
hrs = np.arange(96); hist = 0.25*np.sin(hrs/9.0)+0.12*np.sin(hrs/3.3)
b = hist[71]
h = np.arange(24); u = h/23
z_local = b + 0.06*np.sin(np.pi*u) - 0.03*u         # base forecast: stays near the anchor level, no strong trend
corr = 0.40*np.sin(np.pi*h/24) - 0.24*u**2          # signed correction: 0 at the origin, up, then slightly down
z_hat = z_local + corr
K = 1.2                                             # drawing units per value unit

# ---------------- rows of the grid ----------------
Y_T = 4.0                    # target row (local branch)
Y_S = [2.05, 1.45, 0.85]     # source rows (cross-variable branch); centre 1.45
Y_C = 1.45

# ---------------- input: target ----------------
x0, x1 = 0.35, 3.05
hx = lambda t: x0+(x1-x0)*t/96
hy = lambda v: Y_T+K*(v-b)
ax.add_patch(Rectangle((hx(72), Y_T-0.7), hx(96)-hx(72), 1.35, facecolor=GRAY, edgecolor='none', zorder=0))
ax.plot(hx(hrs[:72]), hy(hist[:72]), '-', color=BLUE, lw=1.0, zorder=3)
ax.plot(hx(hrs[:72:4]), hy(hist[:72:4]), 'o', color=BLUE, ms=1.8, zorder=3)
ax.plot([hx(71), hx(96)], [Y_T, Y_T], '--', color=INK, lw=0.9, zorder=3)
anchor_dot(hx(71), Y_T)
ax.text(hx(71)-0.05, Y_T+0.1, '$b$', fontsize=FS+0.5, ha='right', va='bottom', color=INK)
ax.annotate('', xy=(hx(96), Y_T-0.5), xytext=(hx(71), Y_T-0.5),
            arrowprops=dict(arrowstyle='<->', color=INK, lw=0.7, shrinkA=0, shrinkB=0))
ax.text((hx(71)+hx(96))/2, Y_T-0.45, '$d$', fontsize=FS+0.5, ha='center', va='bottom', color=INK)
ax.text(x0, Y_T+0.78, 'biến đích $i$', fontsize=FS, color=INK, va='bottom')
ax.add_patch(Rectangle((x0, Y_T-1.02), hx(72)-x0, 0.17, facecolor=BLUE_L, edgecolor=INK, lw=0.6, zorder=2))
ax.add_patch(Rectangle((hx(72), Y_T-1.02), hx(96)-hx(72), 0.17, facecolor=GRAY, edgecolor=INK, lw=0.6, hatch='////', zorder=2))
ax.text(x0-0.08, Y_T-0.935, '$M$', fontsize=FS+0.5, ha='right', va='center', color=INK)

# ---------------- input: other channels ----------------
ax.text(x0, Y_S[0]+0.36, 'kênh khác $j$', fontsize=FS, color=INK, va='bottom')
rows = [('PM$_{10}$ cùng trạm', [1, 1, 0, 1]), ('nhiệt độ', [1, 1, 1, 1]), ('NO$_2$ trạm bên', [0, 1, 1, 1])]
pw, gap, ph = 0.31, 0.05, 0.3; sx = x1-4*pw-3*gap
for (name, m), yc in zip(rows, Y_S):
    y = yc-ph/2
    ax.text(sx-0.08, yc, name, fontsize=FS-0.5, ha='right', va='center', color=INK)
    for k, v in enumerate(m):
        px = sx+k*(pw+gap)
        ax.add_patch(Rectangle((px, y), pw, ph, facecolor=GOLD_L if v else GRAY, edgecolor=INK, lw=0.6,
                               hatch=None if v else '////', zorder=2))
        if not v:
            ax.plot([px+0.03, px+pw-0.03], [y+0.03, y+ph-0.03], color=CROSS, lw=1.0, zorder=3)
            ax.plot([px+0.03, px+pw-0.03], [y+ph-0.03, y+0.03], color=CROSS, lw=1.0, zorder=3)

def panel(x0, x1, y0, y1, label):
    """White chart box with a time axis arrow, so a sketch reads as a small plot and not as a connector."""
    ax.add_patch(FancyBboxPatch((x0, y0), x1-x0, y1-y0, boxstyle='round,pad=0,rounding_size=0.06',
                                facecolor='white', edgecolor=LIGHT, linewidth=0.7, zorder=2))
    ax.annotate('', xy=(x1-0.06, y0+0.09), xytext=(x0+0.08, y0+0.09),
                arrowprops=dict(arrowstyle='-|>', color=LIGHT, lw=0.6, mutation_scale=5, shrinkA=0, shrinkB=0), zorder=3)
    ax.text(x0, y1+0.05, label, fontsize=FS+0.5, color=INK, va='bottom', ha='left')


def ref_line(x0, x1, y):
    ax.plot([x0+0.08, x1-0.08], [y, y], color=LIGHT, lw=0.6, ls=(0, (1, 1.5)), zorder=3)


# ---------------- local branch ----------------
FX0, FX1 = 3.4, 7.7
frame(FX0, 2.95, FX1-FX0, 2.1, BLUE_F, 'Nhánh cục bộ')
box(3.65, Y_T-0.45, 1.2, 0.9, 'Mã hóa\nlịch sử', BLUE_L)
arrow((3.12, Y_T-0.3), (3.65, Y_T-0.3))
LX0, LX1, LY0, LY1 = 5.25, 7.45, Y_T-0.45, Y_T+0.45      # base-forecast chart
panel(LX0, LX1, LY0, LY1, r'dự báo nền $\tilde{Z}$')
LB = Y_T+0.02                                           # anchor level inside the chart
ref_line(LX0, LX1, LB)
tl = np.linspace(LX0+0.18, LX1-0.15, 24)
ax.plot(tl, LB+K*(z_local-b), color=BLUE, lw=1.4, zorder=4)
anchor_dot(tl[0], LB)
arrow((4.85, Y_T), (LX0, Y_T))

# ---------------- cross-variable branch ----------------
frame(FX0, 0.15, FX1-FX0, 2.65, GOLD_F, 'Nhánh liên biến')
box(3.65, Y_S[2]-0.3, 1.1, Y_S[0]-Y_S[2]+0.6, 'Mã hóa\nđoạn\n$(k,\\,u)$', GOLD_L)
for yc in Y_S:
    arrow((3.12, yc), (3.65, yc), lw=0.8)
AX0, AX1 = 5.0, 6.0
box(AX0, Y_C-0.45, AX1-AX0, 0.9, 'Attention', GOLD_L)
arrow((4.75, Y_C), (AX0, Y_C))
QX = (AX0+AX1)/2                                        # query: straight down from the base-forecast chart
arrow((QX, LY0), (QX, Y_C+0.45), color=BLUE, lw=0.8)
ax.text(QX+0.08, 3.25, '$q$', fontsize=FS+0.5, color=BLUE, ha='left', va='center')
box(QX-0.3, 0.32, 0.6, 0.36, '$\\varnothing$', 'white', size=FS+1)
arrow((QX, 0.68), (QX, Y_C-0.45), lw=0.7)
BX0, BX1, BY0, BY1 = 6.25, 7.45, Y_C-0.5, Y_C+0.6     # signed-correction chart, same scale K
panel(BX0, BX1, BY0, BY1, 'hiệu chỉnh $\\pm$')
BZ = Y_C-0.02                                           # zero line
ref_line(BX0, BX1, BZ)
for k, c in enumerate(corr[::2]):
    xk = BX0+0.14+k*(BX1-BX0-0.32)/11
    ax.add_patch(Rectangle((xk, BZ if c >= 0 else BZ+K*c), 0.065, abs(K*c), facecolor=GOLD, edgecolor='none', zorder=4))
arrow((AX1, Y_C), (BX0, Y_C))

# ---------------- sum and output ----------------
CX, CY = 8.15, 2.85
ax.add_patch(Circle((CX, CY), 0.19, facecolor='white', edgecolor=INK, lw=0.9, zorder=5))
ax.text(CX, CY, '+', ha='center', va='center', fontsize=10, color=INK, zorder=6)
arrow((LX1, Y_T), (CX, CY+0.19), conn='angle,angleA=0,angleB=90,rad=0')
arrow((BX1, Y_C), (CX, CY-0.19), conn='angle,angleA=0,angleB=-90,rad=0')
OX0, OX1, OY0, OY1 = 8.6, 10.25, CY-0.6, CY+0.65        # final-forecast chart = base + correction
panel(OX0, OX1, OY0, OY1, r'dự báo cuối $\hat{Z}$')
OB = CY+0.0                                            # anchor level inside the chart
ref_line(OX0, OX1, OB)
to = np.linspace(OX0+0.18, OX1-0.15, 24)
ax.plot(to, OB+K*(z_local-b), color=BLUE, lw=0.9, ls=(0, (3, 2)), zorder=4)
ax.plot(to, OB+K*(z_hat-b), color=INK, lw=1.5, zorder=4)
anchor_dot(to[0], OB)
arrow((CX+0.19, CY), (OX0, CY))

fig.savefig(folder/'otarf_overview.pdf'); fig.savefig(folder/'otarf_overview.png', dpi=220)
print(folder/'otarf_overview.pdf')
