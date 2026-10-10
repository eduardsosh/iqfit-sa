"""Generate test instances of different size.

1. A random complete solution of the empty board is found with the exact
   solver (randomised Algorithm X).
2. k pieces are removed from it; the remaining pieces stay pre-placed
   (like the starting positions in the IQ Fit booklet). The algorithm must
   place the k removed pieces.  k = 10 means an empty board.
3. Larger "double" boards (10x10, two piece sets = 20 pieces) are built
   by stacking two random 5x10 solutions.

Since every instance is created from a full solution, the optimal cost is 0.
"""
import argparse
import os
import random

from iqfit.pieces import ALL_PIECES
from iqfit.problem import Instance
from iqfit.exact import solve_exact


def random_full_solution(rows, cols, seed):
    empty = Instance("tmp", rows, cols, [], ALL_PIECES)
    assignment, _ = solve_exact(empty, seed=seed)
    return [(lbl, [(x // cols, x % cols) for x in empty.placements[i][k].cells])
            for i, (lbl, k) in enumerate(zip(empty.free, assignment))]


def solution_grid(rows, cols, pieces):
    grid = [["."] * cols for _ in range(rows)]
    for lbl, cells in pieces:
        for r, c in cells:
            grid[r][c] = lbl
    return ["".join(row) for row in grid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="instances")
    ap.add_argument("--per-size", type=int, default=3, help="instances per size")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    # ---- standard 5x10 board, k free pieces
    for k in (2, 4, 6, 8, 10):
        # the empty board (k = all pieces) is a single instance
        for j in range(args.per_size if k < 10 else 1):
            sol = random_full_solution(5, 10, rng.randrange(10**9))
            free_idx = set(rng.sample(range(10), k))
            fixed = [p for i, p in enumerate(sol) if i not in free_idx]
            free = [p[0] for i, p in enumerate(sol) if i in free_idx]
            name = f"std_5x10_k{k:02d}_{j + 1}"
            inst = Instance(name, 5, 10, fixed, free, 0, solution_grid(5, 10, sol))
            inst.save(os.path.join(args.out, name + ".json"))
            print(name, "free:", "".join(free))

    # ---- double board 10x10 with two piece sets, k free pieces out of 20
    for k in (14, 20):
        for j in range(args.per_size if k < 20 else 1):
            top = random_full_solution(5, 10, rng.randrange(10**9))
            bottom = [(lbl, [(r + 5, c) for r, c in cells])
                      for lbl, cells in random_full_solution(5, 10, rng.randrange(10**9))]
            sol = top + bottom
            free_idx = set(rng.sample(range(20), k))
            fixed = [p for i, p in enumerate(sol) if i not in free_idx]
            free = [p[0] for i, p in enumerate(sol) if i in free_idx]
            name = f"dbl_10x10_k{k:02d}_{j + 1}"
            inst = Instance(name, 10, 10, fixed, free, 0, solution_grid(10, 10, sol))
            inst.save(os.path.join(args.out, name + ".json"))
            print(name, "free:", "".join(free))


if __name__ == "__main__":
    main()
