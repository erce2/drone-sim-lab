"""把训练曲线 (TensorBoard) 和评测轨迹画成图。在云端用 matplotlib 跑, 不需要 Kit。

用法:
  python make_plots.py --logdir C:\\IsaacLab\\logs\\rsl_rl\\drone_racing_direct\\<run> --out C:\\work\\drone_lab\\out
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", type=str, required=True)
parser.add_argument("--out", type=str, default="C:/work/drone_lab/out")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)


# ---------------------------------------------------------------- TensorBoard
def read_scalars(logdir):
    """读 TensorBoard 标量。logdir 可以是单个 run, 也可以是包含多个 run 的父目录
    (训练中途 --resume 会新建一个 run 目录, 这里按 step 合并成一条曲线)。"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    dirs = [logdir]
    if os.path.isdir(logdir):
        subs = [d for d in glob.glob(os.path.join(logdir, "*")) if os.path.isdir(d)]
        dirs += [d for d in subs if glob.glob(os.path.join(d, "events.out.tfevents*"))]

    data = {}
    for d in dirs:
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        for t in ea.Tags()["scalars"]:
            data.setdefault(t, []).extend((e.step, e.value) for e in ea.Scalars(t))

    for t in data:
        pts = sorted(set(data[t]))
        data[t] = ([p[0] for p in pts], [p[1] for p in pts])
    return data


def pick(tags, *keys):
    for k in keys:
        for t in tags:
            if k.lower() in t.lower():
                return t
    return None


def plot_training(logdir):
    sc = read_scalars(logdir)
    print("[PLOT] tags:", sorted(sc.keys()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    t = pick(sc, "mean_reward")
    if t:
        axes[0].plot(*sc[t], lw=1.2)
        axes[0].set_title(t)
    axes[0].set_xlabel("iteration")
    axes[0].grid(alpha=0.3)

    t = pick(sc, "mean_episode_length")
    if t:
        axes[1].plot(*sc[t], lw=1.2, color="C1")
        axes[1].set_title(t)
    axes[1].set_xlabel("iteration")
    axes[1].grid(alpha=0.3)

    t = pick(sc, "success_rate")
    if t:
        axes[2].plot(*sc[t], lw=1.2, color="C2", label="success rate")
    t2 = pick(sc, "gates_passed")
    if t2:
        ax2 = axes[2].twinx()
        ax2.plot(*sc[t2], lw=1.0, color="C3", alpha=0.7, label="gates passed")
        ax2.set_ylabel("gates passed", color="C3")
    axes[2].set_title("episode metrics")
    axes[2].set_xlabel("iteration")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    p = os.path.join(args.out, "01_training_curves.png")
    fig.savefig(p, dpi=130)
    print("[PLOT] saved", p)
    return sc


def plot_reward_terms(sc):
    keys = [t for t in sc if "Episode_Reward" in t]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for t in keys:
        ax.plot(*sc[t], lw=1.2, label=t.split("/")[-1])
    ax.set_xlabel("iteration")
    ax.set_ylabel("episode reward (per second)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(args.out, "02_reward_terms.png")
    fig.savefig(p, dpi=130)
    print("[PLOT] saved", p)


def draw_gate_2d(ax, g, hw, hh, horizontal_axis):
    """在 x-y 或 x-z 投影上画门的窗口。"""
    if horizontal_axis == "y":
        ax.add_patch(Rectangle((g[0] - 0.03, g[1] - hw), 0.06, 2 * hw,
                               fill=False, ec="C2", lw=1.5, alpha=0.8))
    else:
        ax.add_patch(Rectangle((g[0] - 0.03, g[2] - hh), 0.06, 2 * hh,
                               fill=False, ec="C2", lw=1.5, alpha=0.8))


def plot_trajectory(name):
    f = os.path.join(args.out, f"traj_{name}.json")
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    traj = d["traj"]
    if len(traj) < 2:
        print("[PLOT] no trajectory recorded for", name)
        return
    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]
    zs = [p[2] for p in traj]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    for g in d["gates"]:
        draw_gate_2d(ax, g, d["half_width"], d["half_height"], "y")
    ax.plot(xs, ys, lw=1.5, color="C0", label="trajectory")
    ax.scatter([xs[0]], [ys[0]], marker="o", color="C0", label="start")
    for i, g in enumerate(d["gates"]):
        ax.annotate(f"gate {i}", (g[0], g[1]), textcoords="offset points", xytext=(4, 6), fontsize=8, color="C2")
    ax.set_xlabel("x [m]  (track direction)")
    ax.set_ylabel("y [m]")
    ax.set_title(f"top view ({name})")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for g in d["gates"]:
        draw_gate_2d(ax, g, d["half_width"], d["half_height"], "z")
    ax.plot(xs, zs, lw=1.5, color="C0")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"side view ({name})")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    p = os.path.join(args.out, f"03_trajectory_{name}.png")
    fig.savefig(p, dpi=130)
    print("[PLOT] saved", p)


def plot_eval_bars():
    files = glob.glob(os.path.join(args.out, "eval_*.json"))
    if not files:
        return
    rows = [json.load(open(f)) for f in files]
    labels = [r["controller"] for r in rows]
    succ = [r["success_rate"] * 100 for r in rows]
    gates = [r["mean_gates_passed"] for r in rows]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = range(len(rows))
    ax.bar([i - 0.2 for i in x], succ, width=0.4, label="success rate [%]", color="C0")
    ax.set_ylabel("success rate [%]")
    ax2 = ax.twinx()
    ax2.bar([i + 0.2 for i in x], gates, width=0.4, label="gates passed", color="C1")
    ax2.set_ylabel("mean gates passed")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title("RL policy vs scripted baseline")
    fig.tight_layout()
    p = os.path.join(args.out, "04_eval_compare.png")
    fig.savefig(p, dpi=130)
    print("[PLOT] saved", p)


def main():
    sc = plot_training(args.logdir)
    plot_reward_terms(sc)
    for name in ["rl", "scripted"]:
        plot_trajectory(name)
    plot_eval_bars()
    print("[PLOT] DONE")


if __name__ == "__main__":
    main()