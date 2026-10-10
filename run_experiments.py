"""Run SA on all test instances several times and collect statistics.

usage: python run_experiments.py [--runs 10] [--time 60] [--workers 8]

Outputs (folder results/):
    runs.csv         - one row per SA run
    summary.csv/.md  - statistics per instance and per instance group
    examples.txt     - start board and best SA board for every instance
    convergence.png  - best cost vs. iteration for selected instances
"""
import argparse
import csv
import glob
import os
import statistics
from multiprocessing import Pool

from iqfit.exact import solve_exact
from iqfit.problem import Instance
from iqfit.sa import SAParams, simulated_annealing

OUT = "results"


def sa_run(job):
    path, seed, time_limit = job
    inst = Instance.load(path)
    res = simulated_annealing(inst, SAParams(time_limit=time_limit, max_iters=10**9), seed=seed)
    return {
        "instance": inst.name, "seed": seed, "solved": int(res.solved),
        "best_cost": res.best_cost, "optimum": inst.optimum,
        "time_s": round(res.time, 4), "iterations": res.iterations,
        "time_to_best_s": round(res.time_to_best, 3), "iter_to_best": res.iter_to_best,
        "cycles": res.cycles, "accepted": res.accepted,
        "history": res.history, "assignment": res.best_assignment,
    }


def exact_run(job):
    path, time_limit = job
    inst = Instance.load(path)
    sol, stats = solve_exact(inst, time_limit=time_limit)
    return inst.name, stats["time"], stats["timeout"], stats["nodes"]


def group_of(name):
    # std_5x10_k08_1 -> std_5x10_k08
    return name.rsplit("_", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--time", type=float, default=60.0, help="SA time limit per run (s)")
    ap.add_argument("--exact-time", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--pattern", default="instances/*.json")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    paths = sorted(glob.glob(args.pattern))
    instances = {Instance.load(p).name: Instance.load(p) for p in paths}
    jobs = [(p, seed, args.time) for p in paths for seed in range(args.runs)]
    print(f"{len(paths)} instances x {args.runs} runs, {args.workers} workers")

    with Pool(args.workers) as pool:
        runs = []
        for r in pool.imap_unordered(sa_run, jobs):
            runs.append(r)
            print(f"  {r['instance']:<18} seed={r['seed']:<3} cost={r['best_cost']} "
                  f"time={r['time_s']:.2f}s")
        exact = {name: (t, to, nodes) for name, t, to, nodes in
                 pool.map(exact_run, [(p, args.exact_time) for p in paths])}

    runs.sort(key=lambda r: (r["instance"], r["seed"]))
    fields = [k for k in runs[0] if k not in ("history", "assignment")]
    with open(os.path.join(OUT, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(runs)

    # ------------------------------------------------ per-instance summary
    by_inst = {}
    for r in runs:
        by_inst.setdefault(r["instance"], []).append(r)
    rows = []
    for name, rs in by_inst.items():
        inst = instances[name]
        solved = [r for r in rs if r["solved"]]
        ex_t, ex_to, ex_nodes = exact[name]
        rows.append({
            "instance": name,
            "board": f"{inst.rows}x{inst.cols}",
            "free_pieces": inst.n_free,
            "free_cells": len(inst.free_cells),
            "search_space": f"{inst.search_space_size():.2e}",
            "runs": len(rs),
            "solved": len(solved),
            "success_rate": round(len(solved) / len(rs), 2),
            "best_cost_min": min(r["best_cost"] for r in rs),
            "best_cost_mean": round(statistics.mean(r["best_cost"] for r in rs), 2),
            "best_cost_max": max(r["best_cost"] for r in rs),
            "optimum": inst.optimum,
            "time_mean_s": round(statistics.mean(r["time_s"] for r in rs), 4),
            "time_solved_mean_s": round(statistics.mean(r["time_s"] for r in solved), 4) if solved else "",
            "time_solved_median_s": round(statistics.median(r["time_s"] for r in solved), 4) if solved else "",
            "iters_solved_mean": int(statistics.mean(r["iterations"] for r in solved)) if solved else "",
            "exact_time_s": f">{args.exact_time:.0f}" if ex_to else round(ex_t, 3),
        })
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------- per-group summary
    groups = {}
    for name, rs in by_inst.items():
        groups.setdefault(group_of(name), []).extend(rs)
    grows = []
    for g, rs in groups.items():
        solved = [r for r in rs if r["solved"]]
        grows.append({
            "group": g,
            "instances": len({r["instance"] for r in rs}),
            "runs": len(rs),
            "success_rate": round(len(solved) / len(rs), 2),
            "best_cost_mean": round(statistics.mean(r["best_cost"] for r in rs), 2),
            "time_solved_mean_s": round(statistics.mean(r["time_s"] for r in solved), 4) if solved else "",
            "time_mean_s": round(statistics.mean(r["time_s"] for r in rs), 4),
        })
    grows.sort(key=lambda g: g["group"], reverse=True)

    def md_table(data):
        keys = list(data[0])
        lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        lines += ["| " + " | ".join(str(d[k]) for k in keys) + " |" for d in data]
        return "\n".join(lines)

    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write(f"# Results\n\nSA time limit per run: {args.time:.0f} s, runs per instance: "
                f"{args.runs}, parallel workers: {args.workers}.\n"
                f"Cost = number of empty cells + number of extra coverings; optimum = 0.\n\n")
        f.write("## Per instance group\n\n" + md_table(grows) + "\n\n")
        f.write("## Per instance\n\n" + md_table(rows) + "\n")

    # ------------------------------------------------- example boards
    with open(os.path.join(OUT, "examples.txt"), "w") as f:
        for name, rs in by_inst.items():
            inst = instances[name]
            best = min(rs, key=lambda r: (r["best_cost"], r["time_s"]))
            f.write(f"=== {name} (seed {best['seed']}, cost {best['best_cost']}, "
                    f"{best['time_s']:.2f}s)\nstart:\n{inst.render()}\nSA result:\n"
                    f"{inst.render(best['assignment'])}\n\n")

    # ------------------------------------------------- convergence plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name in sorted(by_inst):
            if not name.endswith("_1"):
                continue
            rs = sorted(by_inst[name], key=lambda r: r["iterations"])
            med = rs[len(rs) // 2]
            xs = [h[0] for h in med["history"]]
            ys = [h[3] for h in med["history"]]
            ax.step([max(x, 1) for x in xs], ys, where="post", label=name)
        ax.set_xscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("best cost")
        ax.set_title("SA convergence (median run per instance)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "convergence.png"), dpi=150)
    except ImportError:
        print("matplotlib not installed - skipping plot")

    print("\n" + open(os.path.join(OUT, "summary.md")).read())


if __name__ == "__main__":
    main()
