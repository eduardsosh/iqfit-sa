"""Exact solver (Knuth's Algorithm X, dictionary version).

NOT the optimisation algorithm of this project - it is only used to
generate test instances and to verify that an instance is solvable
(i.e. that the optimal cost 0 is reachable) and to compare run times.
"""
import random
import time


def _exact_cover(X, Y, solution, stats, rng, deadline):
    if not X:
        return list(solution)
    stats["nodes"] += 1
    if deadline is not None and stats["nodes"] % 1000 == 0 and time.perf_counter() > deadline:
        raise TimeoutError
    # choose the column with the fewest candidates (MRV heuristic)
    col = min(X, key=lambda c: len(X[c]))
    rows = list(X[col])
    if rng is not None:
        rng.shuffle(rows)
    for row in rows:
        solution.append(row)
        removed = _select(X, Y, row)
        found = _exact_cover(X, Y, solution, stats, rng, deadline)
        if found is not None:
            return found
        _deselect(X, Y, row, removed)
        solution.pop()
    return None


def _select(X, Y, row):
    cols = []
    for j in Y[row]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols


def _deselect(X, Y, row, cols):
    for j in reversed(Y[row]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)


def solve_exact(instance, seed=None, time_limit=None):
    """Return (assignment or None, stats). assignment[piece_i] = placement index."""
    Y = {}
    for pi, plist in enumerate(instance.placements):
        for k, p in enumerate(plist):
            Y[(pi, k)] = [("piece", pi)] + [("cell", x) for x in p.cells]
    X = {("piece", pi): set() for pi in range(instance.n_free)}
    X.update({("cell", x): set() for x in instance.free_cells})
    for row, cols in Y.items():
        for c in cols:
            X[c].add(row)
    stats = {"nodes": 0}
    rng = random.Random(seed) if seed is not None else None
    deadline = time.perf_counter() + time_limit if time_limit else None
    t0 = time.perf_counter()
    try:
        sol = _exact_cover(X, Y, [], stats, rng, deadline)
        stats["timeout"] = False
    except TimeoutError:
        sol = None
        stats["timeout"] = True
    stats["time"] = time.perf_counter() - t0
    if sol is None:
        return None, stats
    assignment = [0] * instance.n_free
    for pi, k in sol:
        assignment[pi] = k
    return assignment, stats
