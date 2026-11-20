"""How S2 numbers the Earth with a Hilbert curve, as a Manim explainer.

Render (1080p30):
    manim --resolution 1920,1080 --frame_rate 30 03b_s2_hilbert_curve_video.py HilbertS2
"""
from manim import *
import numpy as np

# S2's own Hilbert tables (s2coords.h). An orientation is 2 bits:
# bit 1 swaps i and j (mirror across the diagonal), bit 2 inverts both
# (together with bit 1: mirror across the anti-diagonal).
POS_TO_IJ = [[0, 1, 3, 2], [0, 2, 3, 1], [3, 2, 0, 1], [3, 1, 0, 2]]
IJ_TO_POS = [[0, 1, 3, 2], [0, 3, 1, 2], [2, 3, 1, 0], [2, 1, 3, 0]]
POS_TO_ORIENT = [1, 0, 0, 3]

SERIF = "Georgia"
MONO = "Consolas"
CURVE = "#58C4DD"
ACCENT = "#FFD166"
BAD = "#FC6255"
FAINT = "#33404c"
WIDTH = {1: 7, 2: 5, 3: 4, 4: 3, 5: 2}
config.background_color = "#0b0d10"


class HilbertS2(Scene):
    def construct(self):
        self.cap = None
        self.intro()
        self.quadtree()
        self.zorder()
        self.hilbert_build()
        self.lookup()
        self.locality()
        self.cell_id()

    def intro(self):
        side = 6.4
        curves = [hcurve(n, side, ORIGIN, WIDTH[n]).set_stroke(opacity=0.45) for n in range(1, 6)]
        self.play(Create(curves[0]), run_time=1)
        for a, b in zip(curves, curves[1:]):
            self.play(ReplacementTransform(a, b), run_time=0.9)
        title = Text("Numbering the Earth along one line", font=SERIF, font_size=48)
        sub = Text("the Hilbert curve inside Google's S2", font=SERIF, font_size=30, color=GREY_B)
        sub.next_to(title, DOWN, buff=0.3)
        words = VGroup(title, sub)
        plate = BackgroundRectangle(words, color=config.background_color, fill_opacity=0.85, buff=0.35)
        self.play(FadeIn(plate), Write(title), FadeIn(sub, shift=0.2 * UP), run_time=1.6)
        self.wait(2)
        self.play(FadeOut(VGroup(plate, words, curves[-1])))

    def quadtree(self):
        side, c = 5.0, 0.45 * UP
        self.board = Square(side).move_to(c).set_stroke(GREY_B, 2)
        self.play(Create(self.board))
        self.say("S2 projects the Earth onto a cube, then splits each face\ninto 4 cells, each of those into 4, thirty levels deep.", wait=0.5)
        self.grids = VGroup()
        for n in (1, 2, 3):
            g = grid_lines(n, side, c, only_new=True).set_stroke(GREY_C, 1.5 if n == 1 else 1, opacity=0.8)
            self.play(Create(g), run_time=0.9)
            self.grids.add(g)
        self.wait(0.6)
        self.say("A database index is one sorted line, so every cell needs a number.\nThe quadtree says which cell is inside which, not which child comes first.", wait=3)

    def zorder(self):
        side, c = 5.0, 0.45 * UP
        z = curve_from(lambda p: z_ij(p, 3), 3, side, c).set_stroke(GREY_A, WIDTH[3])
        self.say("One choice is Z-order: the same N-shaped pattern inside every cell.", wait=0)
        self.play(Create(z), run_time=3.5, rate_func=linear)
        # Any step longer than one cell is a jump: neighbours on the line, strangers on the map.
        jumps = VGroup()
        for p in range(63):
            a, b = z_ij(p, 3), z_ij(p + 1, 3)
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1:
                jumps.add(Line(cell_center(*a, 3, side, c), cell_center(*b, 3, side, c)).set_stroke(BAD, 5))
        self.play(Create(jumps, lag_ratio=0.15), run_time=1.5)
        self.say("Each parent still gets one block of numbers, but consecutive\nnumbers keep jumping across the map (the red steps).", wait=3)
        self.play(FadeOut(z), FadeOut(jumps))

    def hilbert_build(self):
        side, c = 5.0, 0.45 * UP
        self.play(FadeOut(self.grids[1:]))
        base = hcurve(1, side, c, WIDTH[1])
        dots, labels = VGroup(), VGroup()
        for p in range(4):
            pt = cell_center(*hilbert_ij(p, 1), 1, side, c)
            dots.add(Dot(pt, radius=0.09, color=ACCENT))
            labels.add(Text(str(p), font=MONO, font_size=34, color=ACCENT).move_to(pt + 0.45 * normalize(pt - c)))
        self.say("Hilbert's 1891 idea: visit the four children in a U,\nso every step moves to a cell that shares an edge.", wait=0)
        self.play(Create(base), run_time=1.6)
        self.play(FadeIn(dots), LaggedStart(*[FadeIn(l, scale=0.6) for l in labels], lag_ratio=0.2))
        self.wait(2)
        self.play(FadeOut(dots), FadeOut(labels), FadeOut(self.grids[0]))
        current = base
        for n in range(1, 5):
            current = self.double(current, n, side, c, explain=(n == 1))
        self.wait(1)
        self.say("Order n is always four copies of order n − 1,\nso one rule draws the curve at every level.", wait=2.5)
        self.play(FadeOut(current), FadeOut(self.board))

    def double(self, current, n, side, c, explain):
        """Build the order n+1 curve from four copies of the order n curve."""
        q = side / 2
        centres = [c + np.array([(hilbert_ij(p, 1)[0] - 0.5) * q, (hilbert_ij(p, 1)[1] - 0.5) * q, 0]) for p in range(4)]
        copies = [current.copy() for _ in range(4)]
        self.remove(current)
        self.add(*copies)
        if explain:
            self.say("To go one level deeper, shrink four copies into the four quadrants.", wait=0)
        self.play(*[cp.animate.scale(0.5).move_to(ctr).set_stroke(width=WIDTH[n + 1]) for cp, ctr in zip(copies, centres)],
                  run_time=1.4 if explain else 0.8)
        if explain:
            self.wait(1.2)
            self.say("Mirror the first copy across its diagonal\nand the last copy across the other diagonal.", wait=0)
            h = q / 2
            d0 = DashedLine(centres[0] + np.array([-h, -h, 0]), centres[0] + np.array([h, h, 0])).set_stroke(GREY_B, 2)
            d3 = DashedLine(centres[3] + np.array([-h, h, 0]), centres[3] + np.array([h, -h, 0])).set_stroke(GREY_B, 2)
            self.play(Create(d0), Create(d3), run_time=0.6)
        # A half-turn about an in-plane axis is a mirror image once it lands.
        diag, anti = np.array([1, 1, 0]) / np.sqrt(2), np.array([1, -1, 0]) / np.sqrt(2)
        self.play(Rotate(copies[0], PI, axis=diag, about_point=centres[0]),
                  Rotate(copies[3], PI, axis=anti, about_point=centres[3]),
                  run_time=1.6 if explain else 0.9)
        if explain:
            self.play(FadeOut(d0), FadeOut(d3), run_time=0.4)
            self.say("Now each copy ends right beside the start of the next one. Join them.", wait=0)
        links = VGroup(*[Line(copies[p].get_end(), copies[p + 1].get_start()).set_stroke(ACCENT, WIDTH[n + 1] + 1) for p in range(3)])
        self.play(Create(links, lag_ratio=0.5), run_time=1.2 if explain else 0.6)
        if explain:
            self.wait(1.2)
        whole = hcurve(n + 1, side, c, WIDTH[n + 1])
        self.play(FadeIn(whole), FadeOut(VGroup(*copies, links)), run_time=0.6)
        return whole

    def lookup(self):
        side, c = 5.0, np.array([-3.7, 0.45, 0])
        ti, tj = 5, 2
        board = Square(side).move_to(c).set_stroke(GREY_B, 2)
        grid = grid_lines(3, side, c).set_stroke(FAINT, 1)
        dim = hcurve(3, side, c, 3).set_stroke(opacity=0.22)
        target = Square(side / 8).move_to(cell_center(ti, tj, 3, side, c)).set_stroke(width=0).set_fill(ACCENT, 0.55)
        self.say("How does S2 number a cell without drawing the curve?\nTake the cell's (i, j) and read one bit of each per level.", wait=0)
        self.play(Create(board), Create(grid), Create(dim), run_time=1.2)
        self.play(FadeIn(target))
        ibits, jbits = to_bits(ti, 3), to_bits(tj, 3)
        irow, jrow = bit_row(f"i = {ti} =", ibits), bit_row(f"j = {tj} =", jbits)
        panel = VGroup(irow, jrow).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        panel.move_to(np.array([0, 2.55, 0])).align_to(np.array([0.2, 0, 0]), LEFT)
        self.play(FadeIn(panel))
        self.wait(1)
        orient, box_c, box_s = 0, c.copy(), side
        log, digits, u_prev, box_prev = VGroup(), [], None, None
        for level in range(3):
            ib, jb = ibits[level], jbits[level]
            u = oriented_u(orient, box_c, box_s, level)
            swap_in = [FadeOut(u_prev)] if u_prev else []
            self.play(irow[1][level].animate.set_color(ACCENT), jrow[1][level].animate.set_color(ACCENT),
                      FadeIn(u), *swap_in, run_time=0.8)
            pos = IJ_TO_POS[orient][(ib << 1) | jb]
            digits.append(pos)
            child_c = box_c + np.array([(ib - 0.5) * box_s / 2, (jb - 0.5) * box_s / 2, 0])
            box = Square(box_s / 2).move_to(child_c).set_stroke(ACCENT, 3)
            note = "next U is mirrored" if POS_TO_ORIENT[pos] else "next U is the same"
            entry = VGroup(
                Text(f"level {level + 1}: bits ({ib}, {jb}) → child {pos}", font=MONO, font_size=21),
                Text(note, font=MONO, font_size=18, color=GREY_B),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
            log.add(entry)
            log.arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(panel, DOWN, aligned_edge=LEFT, buff=0.5)
            box_anim = ReplacementTransform(box_prev, box) if box_prev else Create(box)
            self.play(Indicate(u[1][pos], color=ACCENT), box_anim, FadeIn(entry, shift=0.1 * RIGHT), run_time=1.0)
            self.say((
                "Bits (1, 0) pick the lower-right quadrant, which this U visits as child 3.\nWrite down 3. That copy was mirrored, so the next U is mirrored too.",
                "Inside it the U is mirrored, so bits (0, 1) now mean child 1.\nThat copy wasn't flipped, so the orientation stays.",
                "Last level: bits (1, 0) mean child 3 again.\nThe digits 3, 1, 3 are the cell's place on the curve.",
            )[level], wait=2.5)
            orient ^= POS_TO_ORIENT[pos]
            box_c, box_s, u_prev, box_prev = child_c, box_s / 2, u, box
        value = sum(d * 4 ** (2 - k) for k, d in enumerate(digits))
        result = Text(f"position = {digits[0]}{digits[1]}{digits[2]} in base 4 = {value}", font=MONO, font_size=22, color=ACCENT)
        result.next_to(log, DOWN, aligned_edge=LEFT, buff=0.35)
        self.play(FadeOut(u_prev), FadeIn(result, shift=0.1 * UP))
        pts = [cell_center(*hilbert_ij(p, 3), 3, side, c) for p in range(value + 1)]
        trace = VMobject().set_points_as_corners(pts).set_stroke(ACCENT, 4)
        self.say(f"Walk the curve from 0 and you land on this cell at step {value}.\nThree lookups here; S2 does thirty, one per level.", wait=0)
        self.play(Create(trace), run_time=3, rate_func=linear)
        self.wait(2)
        self.play(FadeOut(VGroup(board, grid, dim, target, panel, log, result, trace, box_prev)))

    def locality(self):
        # 36..67 is a run where Z-order splits into two edge-connected pieces; Hilbert never splits.
        n, side, lo, hi = 4, 4.6, 36, 67
        boards = []
        for fn, cc, name, colour in ((z_ij, np.array([-3.5, 0.55, 0]), "Z-order", GREY_A),
                                     (hilbert_ij, np.array([3.5, 0.55, 0]), "Hilbert", CURVE)):
            frame = Square(side).move_to(cc).set_stroke(GREY_B, 1.5)
            cv = curve_from(lambda p, fn=fn: fn(p, n), n, side, cc).set_stroke(colour, 2.5, opacity=0.55)
            label = Text(name, font=SERIF, font_size=32).next_to(frame, UP, buff=0.2)
            run = VGroup(*[Square(side / 16).move_to(cell_center(*fn(p, n), n, side, cc)).set_stroke(width=0).set_fill(ACCENT, 0.6)
                           for p in range(lo, hi + 1)])
            boards.append((VGroup(frame, cv, label), run, cc))
        self.play(*[FadeIn(b) for b, _, _ in boards])
        self.say(f"Take the same run of 32 consecutive numbers, {lo} to {hi}, on both curves.", wait=0)
        self.play(*[LaggedStart(*[FadeIn(s) for s in run], lag_ratio=0.08) for _, run, _ in boards], run_time=2.5)
        self.wait(0.8)
        self.say("Hilbert's run is always one connected patch; this Z-order run splits in two.\nIn reverse, a map region breaks into fewer runs on Hilbert, so fewer ranges to scan.", wait=3.5)
        self.play(*[FadeOut(run) for _, run, _ in boards])
        ch = boards[1][2]
        marks = VGroup()
        for ij, side_dir in (((7, 0), LEFT), ((8, 0), RIGHT)):
            pt = cell_center(*ij, n, side, ch)
            marks.add(Square(side / 16).move_to(pt).set_stroke(width=0).set_fill(BAD, 0.85))
            label = Text(f"#{hilbert_pos(*ij, n)}", font=MONO, font_size=26, color=BAD)
            marks.add(label.next_to(pt, DOWN, buff=0.35).shift(0.45 * side_dir))
        self.say("The guarantee runs one way. Two cells touching across the middle\ncan still sit far apart on the curve.", wait=0)
        self.play(FadeIn(marks))
        self.wait(2.5)
        self.cover(boards, marks, n, side)

    def cover(self, boards, marks, n, side):
        """A circle across the bad midline, covered by a few cells, read off as a few ranges."""
        zboard, hboard, ch = boards[0][0], boards[1][0], boards[1][2]
        centre, radius = np.array([8.0, 3.0]), 2.6  # in order-4 cell units, straddling the midline
        cells = covering(centre, radius, n, max_level=3)
        unit = side / 2 ** n

        def to_pt(x, y):
            return ch + np.array([(x / 2 ** n - 0.5) * side, (y / 2 ** n - 0.5) * side, 0])

        circle = Circle(radius * unit).move_to(to_pt(*centre)).set_stroke(WHITE, 3)
        squares = VGroup()
        for level, a, b in cells:
            size = 2 ** (n - level)
            squares.add(Square(size * unit).move_to(to_pt((a + 0.5) * size, (b + 0.5) * size))
                        .set_stroke(ACCENT, 2).set_fill(ACCENT, 0.3))
        spans = [(hilbert_pos(a, b, level) * 4 ** (n - level), (hilbert_pos(a, b, level) + 1) * 4 ** (n - level) - 1)
                 for level, a, b in cells]
        ranges = merge(spans)
        width, total, y = 6.0, 4 ** n, 0.3
        bar = Rectangle(width=width, height=0.45).move_to(np.array([-3.5, y, 0])).set_stroke(GREY_B, 1.5)
        x0 = bar.get_left()[0]
        segs = VGroup(*[Rectangle(width=(e + 1 - s) / total * width, height=0.45).set_stroke(width=0).set_fill(ACCENT, 0.9)
                        .move_to(np.array([x0 + (s + e + 1) / 2 / total * width, y, 0])) for s, e in ranges])
        ticks = VGroup(*[Text(str(v), font=MONO, font_size=20, color=GREY_B)
                         .move_to(np.array([x0 + v / total * width, y - 0.5, 0])) for v in (0, 64, 128, 192, 255)])
        title = Text("the same 256 cells as one sorted line", font=SERIF, font_size=26).next_to(bar, UP, buff=0.5)
        self.say("S2 answers a query with a covering: a few cells that together contain the circle.", wait=0)
        self.play(FadeOut(marks), FadeOut(zboard), Create(circle))
        self.play(LaggedStart(*[FadeIn(s) for s in squares], lag_ratio=0.1), run_time=1.5)
        self.wait(1.5)
        self.say(f"Each cell is one unbroken run of numbers, so these {len(cells)} cells\nbecome {len(ranges)} range scans on the sorted index, midline and all.", wait=0)
        self.play(FadeIn(bar), FadeIn(ticks), FadeIn(title))
        self.play(LaggedStart(*[FadeIn(s, scale=0.5) for s in segs], lag_ratio=0.15), run_time=1.5)
        self.wait(3.5)
        self.play(FadeOut(VGroup(hboard, circle, squares, bar, segs, ticks, title)))

    def cell_id(self):
        cells = VGroup(*[Square(0.18).set_stroke(GREY_B, 1) for _ in range(64)]).arrange(RIGHT, buff=0.02).move_to(0.6 * UP)
        for k in range(3):
            cells[k].set_fill(CURVE, 0.9)
        for k in range(3, 63):
            cells[k].set_fill(TEAL if ((k - 3) // 2) % 2 == 0 else GREEN_E, 0.75)
        cells[63].set_fill(ACCENT, 1)
        face = bracket(cells[0], cells[2], "face", up=True)
        position = bracket(cells[3], cells[62], "Hilbert position: 2 bits per level, 30 levels", up=True)
        marker = Text("a final 1", font=SERIF, font_size=22, color=ACCENT).next_to(cells[63], DOWN, buff=0.3).shift(0.3 * LEFT)
        self.say("An S2 cell ID packs it all into 64 bits: the cube face,\nthe Hilbert position, then a 1 that marks the cell's level.", wait=0)
        self.play(LaggedStart(*[FadeIn(s) for s in cells], lag_ratio=0.01), run_time=1.5)
        self.play(FadeIn(face), FadeIn(position), FadeIn(marker))
        example = Text("Ferry Building, San Francisco, level 16 → token 80858066f", font=MONO, font_size=24)
        example.next_to(cells, DOWN, buff=1.0)
        self.play(FadeIn(example, shift=0.1 * UP))
        self.wait(2.5)
        self.say("Places near each other on the curve get IDs near each other,\nso one sorted index keeps neighbouring places together.", wait=3)
        self.play(FadeOut(VGroup(cells, face, position, marker, example, self.cap)))
        self.cap = None
        end = Text("Every place on Earth, numbered along one line.", font=SERIF, font_size=40)
        bg = hcurve(5, 6.4, ORIGIN, 2).set_stroke(opacity=0.25)
        self.play(FadeIn(bg), Write(end), run_time=1.6)
        self.wait(3)
        self.play(FadeOut(end), FadeOut(bg))

    def say(self, text, wait=1.5):
        """Swap the caption at the bottom of the frame."""
        new = Text(text, font=SERIF, font_size=28)
        if new.width > 13:
            new.scale_to_fit_width(13)
        new.to_edge(DOWN, buff=0.4)
        anims = [FadeIn(new, shift=0.15 * UP)]
        if self.cap is not None:
            anims.append(FadeOut(self.cap, shift=0.15 * UP))
        self.play(*anims, run_time=0.6)
        self.cap = new
        if wait:
            self.wait(wait)


def hilbert_ij(p, n):
    """Position p on the order-n curve -> cell (i, j), walking S2's tables top down."""
    orient, i, j = 0, 0, 0
    for level in range(n):
        d = (p >> (2 * (n - 1 - level))) & 3
        ij = POS_TO_IJ[orient][d]
        i, j = (i << 1) | (ij >> 1), (j << 1) | (ij & 1)
        orient ^= POS_TO_ORIENT[d]
    return i, j


def hilbert_pos(i, j, n):
    """Cell (i, j) -> position on the order-n curve: the inverse walk."""
    orient, pos = 0, 0
    for level in range(n - 1, -1, -1):
        p = IJ_TO_POS[orient][(((i >> level) & 1) << 1) | ((j >> level) & 1)]
        pos = (pos << 2) | p
        orient ^= POS_TO_ORIENT[p]
    return pos


def z_ij(p, n):
    """Z-order: the same child order at every level, no orientation state."""
    i = j = 0
    for level in range(n):
        d = (p >> (2 * (n - 1 - level))) & 3
        i, j = (i << 1) | (d >> 1), (j << 1) | (d & 1)
    return i, j


def covering(centre, radius, n, max_level):
    """Quadtree cells (level, a, b) covering a circle: keep a cell once it is inside or at max_level."""
    out = []

    def visit(level, a, b):
        size = 2 ** (n - level)
        x0, y0, x1, y1 = a * size, b * size, (a + 1) * size, (b + 1) * size
        nx, ny = min(max(centre[0], x0), x1), min(max(centre[1], y0), y1)
        if (nx - centre[0]) ** 2 + (ny - centre[1]) ** 2 >= radius ** 2:
            return
        corners = ((x0, y0), (x0, y1), (x1, y0), (x1, y1))
        if level == max_level or all((x - centre[0]) ** 2 + (y - centre[1]) ** 2 <= radius ** 2 for x, y in corners):
            out.append((level, a, b))
            return
        start = len(out)
        for da in (0, 1):
            for db in (0, 1):
                visit(level + 1, 2 * a + da, 2 * b + db)
        # Like S2's normalize step: four kept siblings become their parent.
        if set(out[start:]) == {(level + 1, 2 * a + da, 2 * b + db) for da in (0, 1) for db in (0, 1)}:
            del out[start:]
            out.append((level, a, b))

    for a in (0, 1):
        for b in (0, 1):
            visit(1, a, b)
    return out


def merge(spans):
    """Join position ranges that touch end to start."""
    spans = sorted(spans)
    out = [list(spans[0])]
    for s, e in spans[1:]:
        if s == out[-1][1] + 1:
            out[-1][1] = e
        else:
            out.append([s, e])
    return out


def cell_center(i, j, n, side, c):
    k = 2 ** n
    return c + np.array([((i + 0.5) / k - 0.5) * side, ((j + 0.5) / k - 0.5) * side, 0.0])


def curve_from(ij_of, n, side, c):
    return VMobject().set_points_as_corners([cell_center(*ij_of(p), n, side, c) for p in range(4 ** n)])


def hcurve(n, side, c, width):
    return curve_from(lambda p: hilbert_ij(p, n), n, side, c).set_stroke(CURVE, width)


def grid_lines(n, side, c, only_new=False):
    k, g = 2 ** n, VGroup()
    for t in range(1, k):
        if only_new and t % 2 == 0:
            continue
        x = (t / k - 0.5) * side
        g.add(Line(c + np.array([x, -side / 2, 0]), c + np.array([x, side / 2, 0])))
        g.add(Line(c + np.array([-side / 2, x, 0]), c + np.array([side / 2, x, 0])))
    return g


def oriented_u(orient, bc, s, level):
    """The 2x2 U for one orientation inside a box, with children numbered 0-3."""
    size, radius = (30, 24, 18)[level], (0.3, 0.22, 0.16)[level]
    pts, labels = [], VGroup()
    for p in range(4):
        ij = POS_TO_IJ[orient][p]
        pt = bc + np.array([((ij >> 1) - 0.5) * s / 2, ((ij & 1) - 0.5) * s / 2, 0])
        pts.append(pt)
        disc = Circle(radius).move_to(pt).set_fill(config.background_color, 1).set_stroke(TEAL, 2)
        labels.add(VGroup(disc, Text(str(p), font=MONO, font_size=size).move_to(pt)))
    return VGroup(VMobject().set_points_as_corners(pts).set_stroke(TEAL, 5), labels)


def to_bits(v, n):
    return [(v >> (n - 1 - k)) & 1 for k in range(n)]


def bit_row(label, bits):
    digits = VGroup(*[Text(str(b), font=MONO, font_size=34) for b in bits]).arrange(RIGHT, buff=0.18)
    return VGroup(Text(label, font=MONO, font_size=34), digits).arrange(RIGHT, buff=0.3)


def bracket(first, last, text, up=True):
    """A plain square bracket over a run of bit cells (no LaTeX needed)."""
    sign = 1 if up else -1
    y = (first.get_top()[1] + 0.15) if up else (first.get_bottom()[1] - 0.15)
    l, r = first.get_left()[0], last.get_right()[0]
    shape = VMobject().set_points_as_corners([
        [l, y, 0], [l, y + 0.12 * sign, 0], [r, y + 0.12 * sign, 0], [r, y, 0]]).set_stroke(GREY_B, 2)
    label = Text(text, font=SERIF, font_size=22).next_to(shape, UP if up else DOWN, buff=0.12)
    return VGroup(shape, label)
