"""Solve one IQ Fit instance with Simulated Annealing.

usage: python solve.py instances/std_5x10_k08_1.json [--seed 1] [--time 60]
"""
import argparse

from iqfit.problem import Instance
from iqfit.sa import SAParams, simulated_annealing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--time", type=float, default=60.0, help="time limit in seconds")
    ap.add_argument("--max-iters", type=int, default=5_000_000)
    ap.add_argument("--t-start", type=float, default=SAParams.t_start)
    ap.add_argument("--t-end", type=float, default=SAParams.t_end)
    ap.add_argument("--cycle", type=int, default=SAParams.cycle_iters)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    inst = Instance.load(args.instance)
    print(f"Instance {inst.name}: board {inst.rows}x{inst.cols}, "
          f"{len(inst.fixed)} pre-placed, {inst.n_free} to place ({''.join(inst.free)}), "
          f"search space {inst.search_space_size():.3e}")
    print("Start board (upper case = pre-placed):")
    print(inst.render())

    params = SAParams(t_start=args.t_start, t_end=args.t_end, cycle_iters=args.cycle,
                      max_iters=args.max_iters, time_limit=args.time)
    res = simulated_annealing(inst, params, seed=args.seed, verbose=args.verbose)

    print(f"\n{'SOLVED' if res.solved else 'NOT SOLVED'}: cost={res.best_cost} "
          f"(optimum {inst.optimum}), iterations={res.iterations}, cycles={res.cycles}, "
          f"time={res.time:.2f}s")
    print("Best board (lower case = placed by SA, '.' = hole, '#' = overlap):")
    print(inst.render(res.best_assignment))
    print(inst.describe(res.best_assignment))


if __name__ == "__main__":
    main()
