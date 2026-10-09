"""IQ Fit piece definitions.

Every IQ Fit piece is a straight row ("base") of 3 or 4 balls with extra
balls sticking out in two perpendicular directions.  The piece can lie on the
board in two ways (side A / side B):

* side A: the "A" balls lie flat next to the base, the "B" balls point up;
* side B: the piece is rotated 90 degrees around its long axis, so the "B"
  balls lie flat on the other side of the base and the "A" balls point up.

The footprint on the board is therefore the base plus the flat balls.
On one side a piece has 1 extra ball, on the other side 2 extra balls.
Pieces can be rotated in the plane (4 rotations) but cannot be mirrored,
because a mirrored 3D piece is a different physical object.
"""

# label: (colour name, base length, ball positions side A, ball positions side B)
PIECE_SPECS = {
    "B": ("Blue",       3, (1,),   (0, 2)),
    "L": ("Lime",       3, (0,),   (0, 2)),
    "G": ("Green",      3, (1,),   (1, 2)),
    "P": ("Purple",     3, (2,),   (0, 1)),
    "Y": ("Yellow",     4, (2, 3), (0,)),
    "R": ("Red",        4, (0,),   (0, 3)),
    "D": ("LightBlue",  4, (0,),   (0, 2)),
    "M": ("Pink",       4, (0, 1), (1,)),
    "O": ("Orange",     4, (1,),   (1, 3)),
    "C": ("Cyan",       4, (1,),   (1, 2)),
}

ALL_PIECES = list(PIECE_SPECS)


def side_cells(label, side):
    """Footprint (list of (row, col)) of a piece lying on the given side (0=A, 1=B)."""
    _, length, balls_a, balls_b = PIECE_SPECS[label]
    if side == 0:
        # base in row 0, A-balls below it in row 1
        return [(0, x) for x in range(length)] + [(1, x) for x in balls_a]
    # base in row 1, B-balls above it in row 0 (opposite side of the base)
    return [(1, x) for x in range(length)] + [(0, x) for x in balls_b]


def normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def rotate90(cells):
    """Rotate cells 90 degrees clockwise: (r, c) -> (c, -r)."""
    return normalize([(c, -r) for r, c in cells])


class Orientation:
    """One fixed shape of a piece: a side and a rotation."""

    __slots__ = ("label", "side", "rot", "cells", "height", "width")

    def __init__(self, label, side, rot, cells):
        self.label = label
        self.side = side
        self.rot = rot
        self.cells = cells
        self.height = max(r for r, _ in cells) + 1
        self.width = max(c for _, c in cells) + 1

    def __repr__(self):
        return f"Orientation({self.label}, side={'AB'[self.side]}, rot={self.rot * 90})"


def orientations(label):
    """All distinct orientations of a piece (max 2 sides x 4 rotations = 8)."""
    result, seen = [], set()
    for side in (0, 1):
        cells = normalize(side_cells(label, side))
        for rot in range(4):
            if cells not in seen:
                seen.add(cells)
                result.append(Orientation(label, side, rot, cells))
            cells = rotate90(cells)
    return result


def render_shape(cells):
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    s = set(cells)
    return "\n".join("".join("#" if (r, c) in s else "." for c in range(w)) for r in range(h))


if __name__ == "__main__":
    for lbl, (name, *_rest) in PIECE_SPECS.items():
        print(f"{lbl} ({name})")
        for side in (0, 1):
            print(f" side {'AB'[side]}:")
            print("  " + render_shape(normalize(side_cells(lbl, side))).replace("\n", "\n  "))
        print(f" orientations: {len(orientations(lbl))}\n")
