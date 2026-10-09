"""Problem instance: board, pre-placed pieces, pieces to place and their domains."""
import json

from .pieces import PIECE_SPECS, orientations


class Placement:
    """A concrete placement of a piece: orientation + top-left offset on the board."""

    __slots__ = ("orient_idx", "side", "rot", "row", "col", "cells")

    def __init__(self, orient_idx, orient, row, col, cols):
        self.orient_idx = orient_idx
        self.side = orient.side
        self.rot = orient.rot
        self.row = row
        self.col = col
        # board cells as flat indices r * cols + c
        self.cells = tuple((row + r) * cols + (col + c) for r, c in orient.cells)


class Instance:
    """
    rows x cols board.
    fixed: list of (label, [(r, c), ...]) - pieces placed in advance (the puzzle task)
    free:  list of piece labels that the algorithm must place (duplicates allowed)
    """

    def __init__(self, name, rows, cols, fixed, free, optimum=0, reference_solution=None):
        self.name = name
        self.rows = rows
        self.cols = cols
        self.fixed = [(lbl, [tuple(rc) for rc in cells]) for lbl, cells in fixed]
        self.free = list(free)
        self.optimum = optimum
        self.reference_solution = reference_solution

        self.blocked = set()
        for _, cells in self.fixed:
            for r, c in cells:
                self.blocked.add(r * cols + c)
        self.free_cells = [i for i in range(rows * cols) if i not in self.blocked]

        # Domain of every free piece: all placements fully inside the board that
        # do not touch pre-placed pieces.
        self.orients = {}
        self.placements = []    # placements[piece_i] -> list[Placement]
        self.lookup = []        # lookup[piece_i][(orient_idx, row, col)] -> placement index
        for lbl in self.free:
            if lbl not in self.orients:
                self.orients[lbl] = orientations(lbl)
            plist, lk = [], {}
            for oi, o in enumerate(self.orients[lbl]):
                for r in range(rows - o.height + 1):
                    for c in range(cols - o.width + 1):
                        p = Placement(oi, o, r, c, cols)
                        if not any(x in self.blocked for x in p.cells):
                            lk[(oi, r, c)] = len(plist)
                            plist.append(p)
            self.placements.append(plist)
            self.lookup.append(lk)

        # cell -> [(piece_i, placement_idx)] covering that cell (used by the "fill hole" move)
        # piece_cover[piece_i][cell] -> [placement_idx] of that piece covering the cell
        self.covering = {i: [] for i in self.free_cells}
        self.piece_cover = []
        for pi, plist in enumerate(self.placements):
            pc = {i: [] for i in self.free_cells}
            for k, p in enumerate(plist):
                for x in p.cells:
                    self.covering[x].append((pi, k))
                    pc[x].append(k)
            self.piece_cover.append(pc)

    @property
    def n_free(self):
        return len(self.free)

    def search_space_size(self):
        size = 1
        for plist in self.placements:
            size *= len(plist)
        return size

    # ----------------------------------------------------------------- I/O
    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["name"], d["rows"], d["cols"],
                   [(p["piece"], p["cells"]) for p in d["fixed"]],
                   d["free"], d.get("optimum", 0), d.get("reference_solution"))

    def save(self, path):
        d = {
            "name": self.name,
            "rows": self.rows,
            "cols": self.cols,
            "fixed": [{"piece": lbl, "cells": [list(rc) for rc in cells]} for lbl, cells in self.fixed],
            "free": self.free,
            "optimum": self.optimum,
            "reference_solution": self.reference_solution,
        }
        with open(path, "w") as f:
            json.dump(d, f, indent=1)

    # ----------------------------------------------------------- rendering
    def render(self, assignment=None):
        """Text picture of the board. Upper case = pre-placed, lower case = placed by
        the algorithm, '.' = empty cell, '#' = overlap of several pieces."""
        grid = [[[] for _ in range(self.cols)] for _ in range(self.rows)]
        for lbl, cells in self.fixed:
            for r, c in cells:
                grid[r][c].append(lbl)
        if assignment is not None:
            for pi, k in enumerate(assignment):
                for x in self.placements[pi][k].cells:
                    grid[x // self.cols][x % self.cols].append(self.free[pi].lower())
        lines = []
        for row in grid:
            lines.append(" ".join("." if not c else (c[0] if len(c) == 1 else "#") for c in row))
        return "\n".join(lines)

    def describe(self, assignment):
        out = []
        for pi, k in enumerate(assignment):
            p = self.placements[pi][k]
            name = PIECE_SPECS[self.free[pi]][0]
            out.append(f"{self.free[pi]} ({name}): side {'AB'[p.side]}, rotation {p.rot * 90} deg, "
                       f"position row={p.row} col={p.col}")
        return "\n".join(out)
