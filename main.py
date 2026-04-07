"""
NatureSim — Main Entry Point
=============================
Usage:
  python main.py                    # animated GUI (default)
  python main.py --headless         # text-only, no window
  python main.py --steps 300        # headless, run 300 ticks
  python main.py --speed 5          # GUI: 5 sim-steps per frame
  python main.py --width 80 --height 80  # larger grid
"""

import argparse
import sys

import numpy as np

from simulation import (
    NatureSim,
    ANIMAL_MAX_COUNT,
    GRID_WIDTH,
    GRID_HEIGHT,
)

# ── CLI arguments ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="NatureSim Forest Ecosystem Simulation")
parser.add_argument("--headless", action="store_true",
                    help="Run without GUI (prints stats to stdout)")
parser.add_argument("--steps",  type=int, default=500,
                    help="Ticks to run in headless mode (default 500)")
parser.add_argument("--speed",  type=int, default=3,
                    help="Simulation steps per animation frame (default 3)")
parser.add_argument("--width",  type=int, default=GRID_WIDTH)
parser.add_argument("--height", type=int, default=GRID_HEIGHT)
args = parser.parse_args()

# ── Headless mode ─────────────────────────────────────────────────────────────

if args.headless:
    sim = NatureSim(args.width, args.height)
    print("=" * 65)
    print("  NatureSim — Forest Ecosystem (headless)")
    print("=" * 65)
    for _ in range(args.steps):
        sim.step()
        if sim.tick % 10 == 0:
            sim.print_summary()
    h = sim.history
    print("\n" + "=" * 65)
    print("  Final Statistics")
    print("=" * 65)
    print(f"  Total ticks       : {sim.tick}")
    print(f"  Avg vegetation    : {np.mean(h['veg_mean']):.3f}")
    print(f"  Peak fire cells   : {max(h['fire_cells'])}")
    print(f"  Peak animal pop.  : {max(h['animal_count'])}")
    print(f"  Final animal pop. : {h['animal_count'][-1]}")
    print(f"  Avg moisture      : {np.mean(h['moisture_mean']):.3f}")
    sys.exit(0)

# ── Visual / animated mode ────────────────────────────────────────────────────

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

sim = NatureSim(args.width, args.height)

# ── Figure layout ─────────────────────────────────────────────────────────────
BG       = "#0f0f1a"
PANEL_BG = "#13131f"
TEXT_COL = "#ccccee"
GRID_COL = "#2a2a44"

fig = plt.figure(figsize=(16, 9), facecolor=BG)
try:
    fig.canvas.manager.set_window_title("NatureSim — Forest Quad-System Simulation")
except Exception:
    pass

gs = GridSpec(
    3, 3,
    figure=fig,
    left=0.03, right=0.97, top=0.94, bottom=0.07,
    wspace=0.40, hspace=0.55,
)

ax_map  = fig.add_subplot(gs[:, :2])   # 2/3 width — main grid
ax_veg  = fig.add_subplot(gs[0, 2])    # vegetation & moisture
ax_fire = fig.add_subplot(gs[1, 2])    # fire cells & risk
ax_pop  = fig.add_subplot(gs[2, 2])    # animal population


def _style_ax(ax, title):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_color(GRID_COL)
    ax.tick_params(colors=TEXT_COL, labelsize=7)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.set_title(title, color=TEXT_COL, fontsize=8, pad=4)


_style_ax(ax_map,  "NatureSim — Forest Ecosystem")
_style_ax(ax_veg,  "Vegetation & Moisture")
_style_ax(ax_fire, "Fire & Drought Risk")
_style_ax(ax_pop,  "Animal Population")

ax_map.axis("off")
fig.suptitle(
    "NatureSim  |  Four-System Forest Simulation",
    color=TEXT_COL, fontsize=11, y=0.985,
)

# ── Map image ─────────────────────────────────────────────────────────────────

img = ax_map.imshow(
    sim.get_rgb(),
    origin="lower",
    interpolation="nearest",
    aspect="equal",
    extent=[0, sim.width, 0, sim.height],
)

# Animal scatter overlay
scat = ax_map.scatter(
    [], [], c="#ffee44", s=14, alpha=0.80, zorder=5, label="Animals",
)

# Color legend (static)
legend_patches = [
    Patch(color=(0.20, 0.85, 0.25), label="Dense vegetation"),
    Patch(color=(0.95, 0.20, 0.05), label="Fire"),
    Patch(color=(0.10, 0.35, 0.55), label="Moist / wet"),
    Patch(color=(0.30, 0.15, 0.05), label="Scorched earth"),
]
animal_handle = plt.Line2D(
    [0], [0], marker="o", color="none",
    markerfacecolor="#ffee44", markersize=6, label="Animals",
)
legend_patches.append(animal_handle)
ax_map.legend(
    handles=legend_patches,
    loc="upper right", fontsize=7,
    facecolor="#000000aa", edgecolor=GRID_COL,
    labelcolor="white",
)

# HUD text (bottom-left of map)
hud = ax_map.text(
    0.01, 0.01, "",
    transform=ax_map.transAxes,
    color="#ffffff", fontsize=8,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#00000099", edgecolor="none"),
)

# ── Time-series plots ─────────────────────────────────────────────────────────

# Vegetation & moisture (both 0–1)
line_veg,   = ax_veg.plot([], [], color="#44dd55", lw=1.5, label="Vegetation")
line_moist, = ax_veg.plot([], [], color="#4499ff", lw=1.0, label="Moisture", alpha=0.75)
ax_veg.set_ylim(0, 1)
ax_veg.set_ylabel("Mean level", fontsize=7, color=TEXT_COL)
ax_veg.legend(fontsize=6, loc="upper right",
              facecolor=PANEL_BG, labelcolor="white", edgecolor=GRID_COL)

# Fire cells (left y-axis) + fire risk (right y-axis)
line_fire, = ax_fire.plot([], [], color="#ff4422", lw=1.5, label="Fire cells")
ax_fire.set_ylim(0, sim.width * sim.height)
ax_fire.set_ylabel("Burning cells", fontsize=7, color="#ff4422")
ax_fire.yaxis.label.set_color("#ff4422")
ax_fire.tick_params(axis="y", colors="#ff4422")

ax_fire2 = ax_fire.twinx()
ax_fire2.set_facecolor(PANEL_BG)
ax_fire2.tick_params(colors="#ffaa22", labelsize=7)
ax_fire2.set_ylim(0, 1)
ax_fire2.set_ylabel("Fire risk", fontsize=7, color="#ffaa22")
line_risk, = ax_fire2.plot([], [], color="#ffaa22", lw=1.0, alpha=0.85, label="Risk")

combined_lines  = [line_fire, line_risk]
combined_labels = ["Fire cells", "Fire risk"]
ax_fire.legend(combined_lines, combined_labels, fontsize=6, loc="upper right",
               facecolor=PANEL_BG, labelcolor="white", edgecolor=GRID_COL)

# Animal population
line_pop, = ax_pop.plot([], [], color="#ffee44", lw=1.5)
ax_pop.set_ylim(0, ANIMAL_MAX_COUNT + 15)
ax_pop.set_ylabel("Count", fontsize=7, color=TEXT_COL)
ax_pop.set_xlabel("Day", fontsize=7, color=TEXT_COL)

# ── Animation ─────────────────────────────────────────────────────────────────

WINDOW = 400   # x-axis rolling window (days shown)


def _xlim(ticks):
    if not ticks:
        return 0, 1
    return max(0, ticks[-1] - WINDOW), max(1, ticks[-1])


def animate(_frame):
    # Advance simulation
    for _ in range(args.speed):
        sim.step()

    h   = sim.history
    tks = h["tick"]
    wx  = sim.weather

    # ── Update map ────────────────────────────────────────────────────────────
    img.set_data(sim.get_rgb())

    if sim.animals:
        xs = [a.x for a in sim.animals]
        ys = [a.y for a in sim.animals]
        scat.set_offsets(np.column_stack([xs, ys]))
    else:
        scat.set_offsets(np.empty((0, 2)))

    weather_label = "RAIN" if wx.raining else f"Drought {wx.drought_days}d"
    hud.set_text(
        f"Day {sim.tick}  |  Animals: {len(sim.animals)}  |  "
        f"Fire: {np.sum(sim.fire > 0.05)} cells\n"
        f"Weather: {weather_label}  |  "
        f"Wind: ({wx.wind_dx:+d},{wx.wind_dy:+d}) ×{wx.wind_speed:.1f}  |  "
        f"Risk: {wx.fire_risk:.2f}"
    )

    if not tks:
        return img, scat, hud

    xmin, xmax = _xlim(tks)

    # ── Vegetation & moisture ─────────────────────────────────────────────────
    line_veg.set_data(tks, h["veg_mean"])
    line_moist.set_data(tks, h["moisture_mean"])
    ax_veg.set_xlim(xmin, xmax)

    # ── Fire & risk ───────────────────────────────────────────────────────────
    line_fire.set_data(tks, h["fire_cells"])
    line_risk.set_data(tks, h["fire_risk"])
    ax_fire.set_xlim(xmin, xmax)
    ax_fire.relim(); ax_fire.autoscale_view(scalex=False)
    ax_fire2.set_xlim(xmin, xmax)

    # ── Population ────────────────────────────────────────────────────────────
    line_pop.set_data(tks, h["animal_count"])
    ax_pop.set_xlim(xmin, xmax)

    return img, scat, hud, line_veg, line_moist, line_fire, line_risk, line_pop


ani = animation.FuncAnimation(
    fig, animate,
    interval=50,          # ms between frames (~20 fps cap)
    cache_frame_data=False,
    blit=False,           # blit=False for twin-axis compatibility
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
