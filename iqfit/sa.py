"""Simulated Annealing for the IQ Fit puzzle (own implementation).

Domain
    A state assigns to every free piece one placement from its domain
    (side A/B, rotation, position) - the piece lies completely inside the
    board and does not touch the pre-placed pieces. Pieces may overlap each
    other and leave holes; those are penalised by the cost function.

Evaluation
    For every free board cell with n pieces on it the cell costs |n - 1|:
    an empty cell costs 1, a cell covered by k > 1 pieces costs k - 1.
    cost = sum over free cells; cost == 0  <=>  a valid solution.

Move (neighbourhood)
    One random piece is changed by one of:
      shift    - move by one cell up/down/left/right
      rotate   - rotate 90 degrees (same side)
      flip     - turn to the other side
      relocate - random placement from the whole domain
      fill     - pick an empty cell and put a random piece onto a random
                 placement that covers it
    or two random pieces are changed at once:
      swap     - the pieces exchange their areas (each gets a random placement
                 covering a cell of the other one's current placement)
"""
import math
import random
import time
from dataclasses import dataclass, field

MOVES = ("shift", "rotate", "flip", "relocate", "fill", "swap")


@dataclass
class SAParams:
    t_start: float = 2.0         # initial temperature
    t_end: float = 0.05          # final temperature of one annealing cycle
    cycle_iters: int = 500_000   # iterations per cycle (geometric cooling t_start -> t_end)
    max_iters: int = 2_000_000   # total iteration limit
    time_limit: float = 60.0     # seconds
    restart_from_best: bool = False  # at reheating continue from the best (True) or current state
    move_weights: tuple = (0.30, 0.10, 0.10, 0.10, 0.25, 0.15)  # shift, rotate, flip, relocate, fill, swap
    history_every: int = 1000    # record (iteration, T, cost, best) every N iterations


@dataclass
class SAResult:
    best_assignment: list
    best_cost: int
    iterations: int
    time: float
    solved: bool
    iter_to_best: int
    time_to_best: float
    cycles: int
    accepted: int
    history: list = field(default_factory=list)


def evaluate(instance, assignment):
    """Cost of a complete state computed from scratch."""
    counts = [0] * (instance.rows * instance.cols)
    for pi, k in enumerate(assignment):
        for x in instance.placements[pi][k].cells:
            counts[x] += 1
    return sum(abs(counts[x] - 1) for x in instance.free_cells)


def simulated_annealing(instance, params=None, seed=None, initial=None, verbose=False):
    params = params or SAParams()
    rng = random.Random(seed)
    rand = rng.random
    randrange = rng.randrange

    n = instance.n_free
    placements = instance.placements
    lookup = instance.lookup
    n_orients = [len(instance.orients[lbl]) for lbl in instance.free]
    orients = [instance.orients[lbl] for lbl in instance.free]
    free_cells = instance.free_cells
    covering = instance.covering
    piece_cover = instance.piece_cover

    # cumulative move probabilities
    total_w = sum(params.move_weights)
    cum, acc = [], 0.0
    for w in params.move_weights:
        acc += w / total_w
        cum.append(acc)

    # ---- initial state: random placement for each piece
    if initial is None:
        assign = [randrange(len(placements[i])) for i in range(n)]
    else:
        assign = list(initial)
    counts = [0] * (instance.rows * instance.cols)
    for pi, k in enumerate(assign):
        for x in placements[pi][k].cells:
            counts[x] += 1
    cost = sum(abs(counts[x] - 1) for x in free_cells)

    best_cost, best_assign = cost, list(assign)
    iter_to_best, time_to_best = 0, 0.0
    history = [(0, params.t_start, cost, best_cost)]

    # geometric cooling factor for one cycle
    alpha = (params.t_end / params.t_start) ** (1.0 / params.cycle_iters)
    T = params.t_start
    cycle_pos, cycles, accepted = 0, 1, 0
    t0 = time.perf_counter()
    it = 0

    while it < params.max_iters and best_cost > 0:
        it += 1

        # ---------------------------------------------------- choose a move
        # changes = list of (piece index, new placement index)
        changes = None
        r = rand()
        if r < cum[2]:
            # local moves on one piece: shift / rotate / flip
            pi = randrange(n)
            cur = placements[pi][assign[pi]]
            oi, row, col = cur.orient_idx, cur.row, cur.col
            if r < cum[0]:
                d = randrange(4)
                if d == 0:
                    row -= 1
                elif d == 1:
                    row += 1
                elif d == 2:
                    col -= 1
                else:
                    col += 1
            else:
                o = orients[pi][oi]
                if r < cum[1]:
                    want_side, want_rot = o.side, (o.rot + 1 + randrange(3)) % 4
                else:
                    want_side, want_rot = 1 - o.side, o.rot
                # find orientation with the wanted side/rotation (symmetric pieces
                # have fewer than 8 orientations - then pick any of that side)
                cand = [j for j, oo in enumerate(orients[pi])
                        if oo.side == want_side and oo.rot == want_rot]
                if not cand:
                    cand = [j for j, oo in enumerate(orients[pi])
                            if oo.side == want_side and j != oi]
                    if not cand:
                        cand = [oi]
                new_oi = cand[0] if len(cand) == 1 else cand[randrange(len(cand))]
                # keep the piece's centre roughly in place
                no = orients[pi][new_oi]
                row += (o.height - no.height) // 2
                col += (o.width - no.width) // 2
                oi = new_oi
            # None if the move would leave the board / hit a pre-placed piece
            new_k = lookup[pi].get((oi, row, col))
            if new_k is not None:
                changes = ((pi, new_k),)
        elif r < cum[3]:
            # relocate: random placement from the whole domain
            pi = randrange(n)
            changes = ((pi, randrange(len(placements[pi]))),)
        elif r < cum[4] or n < 2:
            # fill: cover a random empty cell (fallback: overlapping cell)
            holes = [x for x in free_cells if counts[x] != 1]
            x = holes[randrange(len(holes))]
            opts = covering[x]
            changes = (opts[randrange(len(opts))],)
        else:
            # swap: pieces i and j exchange their areas - each is put on a random
            # placement covering a random cell of the other's current placement
            i = randrange(n)
            j = randrange(n - 1)
            if j >= i:
                j += 1
            ci = placements[i][assign[i]].cells
            cj = placements[j][assign[j]].cells
            oi_ = piece_cover[i][cj[randrange(len(cj))]]
            oj_ = piece_cover[j][ci[randrange(len(ci))]]
            if oi_ and oj_:
                changes = ((i, oi_[randrange(len(oi_))]), (j, oj_[randrange(len(oj_))]))

        if changes is not None:
            # ------------------------------------------- incremental evaluation
            delta = 0
            old = []
            for pi, new_k in changes:
                old_k = assign[pi]
                old.append(old_k)
                for x in placements[pi][old_k].cells:
                    c = counts[x]
                    delta += -1 if c >= 2 else 1      # |c-2| - |c-1|
                    counts[x] = c - 1
                for x in placements[pi][new_k].cells:
                    c = counts[x]
                    delta += -1 if c == 0 else 1      # |c| - |c-1|
                    counts[x] = c + 1
                assign[pi] = new_k

            # ------------------------------------------------ acceptance rule
            if delta <= 0 or rand() < math.exp(-delta / T):
                cost += delta
                accepted += 1
                if cost < best_cost:
                    best_cost = cost
                    best_assign = list(assign)
                    iter_to_best = it
                    time_to_best = time.perf_counter() - t0
            else:
                # undo in reverse order
                for (pi, new_k), old_k in zip(reversed(changes), reversed(old)):
                    for x in placements[pi][new_k].cells:
                        counts[x] -= 1
                    for x in placements[pi][old_k].cells:
                        counts[x] += 1
                    assign[pi] = old_k

        # ------------------------------------------------------- cooling
        T *= alpha
        cycle_pos += 1
        if cycle_pos >= params.cycle_iters:
            # reheating: start a new annealing cycle
            cycle_pos = 0
            cycles += 1
            T = params.t_start
            if params.restart_from_best:
                assign = list(best_assign)
                counts = [0] * len(counts)
                for p2, k2 in enumerate(assign):
                    for x in placements[p2][k2].cells:
                        counts[x] += 1
                cost = best_cost
            if verbose:
                print(f"  cycle {cycles}: it={it} best={best_cost} "
                      f"time={time.perf_counter() - t0:.1f}s")
            if time.perf_counter() - t0 > params.time_limit:
                break

        if it % params.history_every == 0:
            history.append((it, T, cost, best_cost))
            if it % 20_000 == 0 and time.perf_counter() - t0 > params.time_limit:
                break

    elapsed = time.perf_counter() - t0
    history.append((it, T, cost, best_cost))
    assert evaluate(instance, best_assign) == best_cost
    return SAResult(best_assign, best_cost, it, elapsed, best_cost == 0,
                    iter_to_best, time_to_best, cycles, accepted, history)
