"""Activation functions from sigmoid to GeGLU, framed around what modern LLMs use, as a Manim explainer.

Render (1080p30):
    manim --resolution 1920,1080 --frame_rate 30 01_sigmoid_to_geglu_video.py Activations
"""
from manim import *
import math
import numpy as np

SERIF = "Georgia"
MONO = "Consolas"
MATH = "Cambria"  # Georgia and Consolas have no ⊙ or →; Cambria does.
CURVE = "#58C4DD"
ACCENT = "#FFD166"
BAD = "#FC6255"
GOOD = "#83C167"
PURPLE = "#B48EDE"
FAINT = "#33404c"
config.background_color = "#0b0d10"


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def dtanh(x):
    return 1 - math.tanh(x) ** 2


def relu(x):
    return max(0.0, x)


def leaky(x, a=0.1):
    return x if x > 0 else a * x


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def gelu(x):
    return x * Phi(x)


def silu(x):
    return x * sigmoid(x)


class Activations(Scene):
    def construct(self):
        self.cap = None
        self.head = None
        self.intro()
        self.why_nonlinear()
        self.sigmoid_section()
        self.tanh_section()
        self.relu_section()
        self.leaky_section()
        self.gelu_section()
        self.silu_section()
        self.glu_section()
        self.swiglu_section()
        self.geglu_section()
        self.summary()

    def intro(self):
        self.mark("intro")
        ax = make_axes((-4, 4), (-1.2, 3), 9, 4.2, 0.3 * UP)
        faint = VGroup(*[graph(ax, f, -4, 4).set_stroke(c, 3, opacity=0.35)
                         for f, c in ((sigmoid, GREY_B), (math.tanh, TEAL), (relu, BAD), (gelu, CURVE), (silu, ACCENT))])
        self.play(Create(ax), LaggedStart(*[Create(g) for g in faint], lag_ratio=0.25), run_time=2.5)
        title = Text("From sigmoid to GeGLU", font=SERIF, font_size=52)
        sub = Text("the activation functions inside modern LLMs", font=SERIF, font_size=30, color=GREY_B)
        sub.next_to(title, DOWN, buff=0.3)
        words = VGroup(title, sub).move_to(0.3 * UP)
        plate = BackgroundRectangle(words, color=config.background_color, fill_opacity=0.85, buff=0.35)
        self.play(FadeIn(plate), Write(title), FadeIn(sub, shift=0.2 * UP), run_time=1.6)
        self.wait(2.5)
        self.play(FadeOut(VGroup(plate, words, ax, faint)))

    def why_nonlinear(self):
        self.mark("why_nonlinear")
        self.heading("Why a network needs a nonlinearity")
        x = node("x", CURVE)
        w1 = box("W₁", 1.3, 1.1, GREY_B)
        w2 = box("W₂", 1.3, 1.1, GREY_B)
        y = node("y", CURVE)
        row = VGroup(x, w1, w2, y).arrange(RIGHT, buff=1.2).move_to(0.9 * UP)
        arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.12, stroke_width=4, color=GREY_B)
                          for a, b in zip(row, row[1:])])
        self.say("A layer multiplies its input by a matrix of weights.\nStack two layers with nothing in between.", wait=0)
        self.play(FadeIn(row, lag_ratio=0.2), Create(arrows), run_time=1.6)
        self.wait(1.5)
        eq = Text("W₂ (W₁ x)  =  (W₂ W₁) x", font=MATH, font_size=40).move_to(1.0 * DOWN)
        self.play(Write(eq), run_time=1.4)
        merged = box("W₂W₁", 2.0, 1.1, ACCENT).move_to(VGroup(w1, w2).get_center())
        self.say("Matrix products associate, so the two layers collapse into one matrix.\nA hundred linear layers are still just one linear map.", wait=0)
        self.play(ReplacementTransform(VGroup(w1, w2, arrows[1]), merged),
                  arrows[0].animate.put_start_and_end_on(x.get_right() + 0.12 * RIGHT, merged.get_left() + 0.12 * LEFT),
                  arrows[2].animate.put_start_and_end_on(merged.get_right() + 0.12 * RIGHT, y.get_left() + 0.12 * LEFT),
                  run_time=1.5)
        self.wait(4)
        fix = Text("W₂ f(W₁ x)", font=MATH, font_size=40, color=ACCENT).next_to(eq, DOWN, buff=0.45)
        self.say("Put a nonlinear function f between them and the collapse stops.\nThe rest of this video is about which f to use.", wait=0)
        self.play(FadeIn(fix, shift=0.15 * UP))
        self.wait(4.5)
        self.clear_stage()

    def sigmoid_section(self):
        self.mark("sigmoid")
        self.heading("Sigmoid   σ(x) = 1 / (1 + e⁻ˣ)")
        ax = make_axes((-6, 6), (-0.2, 1.2), 7.2, 4.2, np.array([-2.9, 0.15, 0]),
                       xticks=(-6, -3, 0, 3, 6), yticks=(0.25, 0.5, 1))
        s = graph(ax, sigmoid, -6, 6).set_stroke(CURVE, 5)
        self.say("The logistic curve squashes any number into the range 0 to 1.\nIt was the default unit when backpropagation spread in 1986.", wait=0)
        self.play(Create(ax), run_time=1)
        self.play(Create(s), run_time=2)
        self.wait(3)
        d = graph(ax, dsigmoid, -6, 6).set_stroke(ACCENT, 5)
        peak = Dot(ax.c2p(0, 0.25), color=ACCENT, radius=0.08)
        plabel = Text("σ′(0) = 0.25", font=MATH, font_size=26, color=ACCENT).next_to(peak, UR, buff=0.12)
        dl = Text("σ′(x) = σ(x)(1 − σ(x))", font=MATH, font_size=28, color=ACCENT).move_to(ax.c2p(-3.2, 0.85))
        self.say("Its slope is σ(1 − σ). That slope peaks at 0.25, right at zero,\nand is nearly flat once the input moves a few units away.", wait=0)
        self.play(Create(d), FadeIn(dl), run_time=2)
        self.play(FadeIn(peak), FadeIn(plabel))
        self.wait(4)
        rows = VGroup(Text("gradient after n sigmoid layers", font=SERIF, font_size=26, color=GREY_B))
        for n, v in ((1, "0.25"), (2, "0.0625"), (5, "0.00098"), (10, "0.00000095"), (20, "0.0000000000009")):
            rows.add(Text(f"n = {n:>2}   ≤ {v}", font=MONO, font_size=26, color=BAD if n >= 10 else WHITE))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to(np.array([3.9, 0.3, 0]))
        self.say("Backpropagation multiplies one such slope per layer (times the weights).\nIf each factor is at most 0.25, the product shrinks geometrically.", wait=0)
        self.play(FadeIn(rows[0]))
        for r in rows[1:]:
            self.play(FadeIn(r, shift=0.1 * RIGHT), run_time=0.6)
        self.wait(3)
        self.say("This is the vanishing gradient: early layers in a deep sigmoid\nnetwork barely learn, because almost no signal reaches them.", wait=4.5)
        self.ax, self.curve, self.deriv = ax, s, d
        self.play(FadeOut(VGroup(rows, peak, plabel, dl)))

    def tanh_section(self):
        self.mark("tanh")
        ax_old = self.ax
        ax = make_axes((-6, 6), (-1.2, 1.2), 9, 4.4, 0.2 * UP, xticks=(-6, -3, 0, 3, 6), yticks=(-1, 1))
        self.heading("Tanh   tanh(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)")
        t = graph(ax, math.tanh, -6, 6).set_stroke(TEAL, 5)
        dt = graph(ax, dtanh, -6, 6).set_stroke(ACCENT, 5)
        self.play(ReplacementTransform(ax_old, ax), ReplacementTransform(self.curve, t),
                  ReplacementTransform(self.deriv, dt), run_time=1.8)
        self.say("Tanh is a sigmoid stretched to run from −1 to 1, so its outputs\nare centred on zero. LeCun's 1998 \"Efficient BackProp\" recommended it.", wait=0)
        self.wait(3.5)
        peak = Dot(ax.c2p(0, 1), color=ACCENT, radius=0.08)
        plabel = Text("tanh′(0) = 1", font=MATH, font_size=26, color=ACCENT).next_to(peak, UR, buff=0.12)
        self.say("Its slope, 1 − tanh², peaks at 1 instead of 0.25,\nso a gradient can pass through an unsaturated unit unshrunk.", wait=0)
        self.play(FadeIn(peak), FadeIn(plabel))
        self.wait(3.5)
        left = Rectangle(width=ax.c2p(-2.5, 0)[0] - ax.c2p(-6, 0)[0], height=4.4).set_stroke(width=0).set_fill(BAD, 0.15)
        left.move_to(np.array([(ax.c2p(-6, 0)[0] + ax.c2p(-2.5, 0)[0]) / 2, ax.get_center()[1], 0]))
        right = left.copy().move_to(np.array([(ax.c2p(2.5, 0)[0] + ax.c2p(6, 0)[0]) / 2, ax.get_center()[1], 0]))
        sat = VGroup(Text("saturated", font=SERIF, font_size=24, color=BAD).move_to(left.get_top() + 0.3 * DOWN),
                     Text("saturated", font=SERIF, font_size=24, color=BAD).move_to(right.get_top() + 0.3 * DOWN))
        self.say("But both ends still go flat. A unit pushed far from zero\nhas a slope near 0, and the gradient through it vanishes again.", wait=0)
        self.play(FadeIn(left), FadeIn(right), FadeIn(sat))
        self.wait(4.5)
        self.clear_stage()

    def relu_section(self):
        self.mark("relu")
        self.heading("ReLU   max(0, x)")
        ax = make_axes((-4, 4), (-1.5, 4), 7.0, 4.4, np.array([-3.0, 0.2, 0]), xticks=(-4, -2, 0, 2, 4), yticks=(1, 2, 3))
        r = graph(ax, relu, -4, 4).set_stroke(BAD, 5)
        self.say("ReLU passes positive inputs through and zeroes the rest.\nNair and Hinton used it in 2010; AlexNet made it standard in 2012.", wait=0)
        self.play(Create(ax), run_time=1)
        self.play(Create(r), run_time=1.5)
        self.wait(3)
        gax = make_axes((-4, 4), (-0.5, 1.5), 5.0, 2.2, np.array([3.9, 1.0, 0]), xticks=(-4, 0, 4), yticks=(1,))
        glabel = Text("slope", font=SERIF, font_size=24, color=ACCENT).next_to(gax, UP, buff=0.1)
        g0 = Line(gax.c2p(-4, 0), gax.c2p(0, 0)).set_stroke(ACCENT, 5)
        g1 = Line(gax.c2p(0, 1), gax.c2p(4, 1)).set_stroke(ACCENT, 5)
        self.say("For a positive input the slope is exactly 1, at any size, so it never\nsaturates. It is also cheap: one comparison, no exponential.", wait=0)
        self.play(Create(gax), FadeIn(glabel))
        self.play(Create(g0), Create(g1))
        self.wait(4)
        xs = [-1.5, -0.6, 0.4, 1.1, 1.9, 2.7]
        dots = VGroup(*[Dot(ax.c2p(x, relu(x)), radius=0.1, color=GOOD if x > 0 else GREY_B) for x in xs])
        self.say("The catch: each dot is this neuron's input on one example.\nWhere the input is negative, no gradient flows back at all.", wait=0)
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.15))
        self.wait(3.5)
        self.say("One large update can push the bias down so every input is negative.\nThe neuron now outputs 0 everywhere, with slope 0: it is dead.", wait=0)
        self.play(*[d.animate.move_to(ax.c2p(x - 3, 0)).set_color(GREY_B) for d, x in zip(dots, xs)], run_time=2)
        dead = Text("dead neuron: no gradient, no recovery", font=SERIF, font_size=28, color=BAD)
        dead.next_to(gax, DOWN, buff=0.7)
        self.play(FadeIn(dead))
        self.wait(4)
        self.ax, self.curve, self.dots = ax, r, dots
        self.play(FadeOut(VGroup(gax, glabel, g0, g1, dead, dots)))

    def leaky_section(self):
        self.mark("leaky")
        self.heading("Leaky ReLU   max(αx, x)")
        ax = self.ax
        lk = graph(ax, leaky, -4, 4).set_stroke(GOOD, 5)
        self.say("Leaky ReLU keeps a small slope α on the negative side,\nso a neuron below zero still gets some gradient.", wait=0)
        self.play(Transform(self.curve, lk), run_time=1.5)
        note = VGroup(
            Text("α = 0.01 in Maas, Hannun and Ng (2013)", font=SERIF, font_size=26),
            Text("drawn here with α = 0.1 so it is visible", font=SERIF, font_size=22, color=GREY_B),
            Text("He et al. (2015): learn α per channel (PReLU)", font=SERIF, font_size=26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(np.array([3.6, 0.6, 0]))
        self.play(FadeIn(note[0]), FadeIn(note[1]))
        self.wait(4)
        self.play(FadeIn(note[2]))
        self.say("The kink at zero remains. The next step was to smooth it out.", wait=3)
        self.clear_stage()

    def gelu_section(self):
        self.mark("gelu")
        self.heading("GELU   x · Φ(x)")
        ax = make_axes((-4, 3), (-0.6, 3), 8.4, 4.4, 0.2 * UP, xticks=(-4, -2, 0, 2), yticks=(1, 2, 3))
        r = graph(ax, relu, -4, 3).set_stroke(BAD, 3, opacity=0.5)
        phi = graph(ax, Phi, -4, 3).set_stroke(PURPLE, 3)
        phil = Text("Φ(x): chance a standard normal is below x", font=MATH, font_size=24, color=PURPLE)
        phil.move_to(ax.c2p(-1.9, 1.6))
        self.say("Hendrycks and Gimpel (2016): weight x by Φ(x), the Gaussian CDF.\nLarge x is kept, very negative x is dropped, and in between it is gradual.", wait=0)
        self.play(Create(ax), FadeIn(r), run_time=1)
        self.play(Create(phi), FadeIn(phil), run_time=1.5)
        self.wait(3.5)
        g = graph(ax, gelu, -4, 3).set_stroke(CURVE, 5)
        self.play(TransformFromCopy(r, g), run_time=2)
        xm = -0.7518
        dip = Dot(ax.c2p(xm, gelu(xm)), radius=0.08, color=ACCENT)
        dl = Text("minimum ≈ −0.17", font=MATH, font_size=24, color=ACCENT).next_to(dip, DOWN, buff=0.2)
        self.say("The result is a smooth ReLU. Small negative inputs leak through\nas a shallow dip, and the slope changes continuously.", wait=0)
        self.play(FadeIn(dip), FadeIn(dl))
        self.wait(4)
        users = Text("BERT, GPT-2 and GPT-3 use GELU", font=SERIF, font_size=30, color=CURVE).move_to(ax.c2p(1.0, 2.75))
        self.say("GELU became the transformer default: BERT used it in 2018,\nand GPT-2 and GPT-3 kept it.", wait=0)
        self.play(FadeIn(users))
        self.wait(3.5)
        self.ax, self.gelu_curve, self.relu_faint = ax, g, r
        self.play(FadeOut(VGroup(phi, phil, dip, dl, users)))

    def silu_section(self):
        self.mark("silu")
        self.heading("SiLU / Swish   x · σ(x)")
        ax = self.ax
        s = graph(ax, silu, -4, 3).set_stroke(ACCENT, 5)
        legend = VGroup(
            legend_row("GELU   x · Φ(x)", CURVE),
            legend_row("SiLU   x · σ(x)", ACCENT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(ax.c2p(-2.4, 2.4))
        self.say("Swap Φ for the logistic σ and you get SiLU: Elfwing et al. (2017).\nRamachandran et al. found it the same year by search and named it Swish.", wait=0)
        self.play(Create(s), FadeIn(legend), run_time=2)
        self.wait(4)
        xm = -1.2785
        dip = Dot(ax.c2p(xm, silu(xm)), radius=0.08, color=ACCENT)
        dl = Text("minimum ≈ −0.28", font=MATH, font_size=24, color=ACCENT).next_to(dip, DOWN, buff=0.2)
        self.say("The two curves almost coincide. SiLU dips a little deeper,\nand the GELU paper itself approximates GELU as x · σ(1.702 x).", wait=0)
        self.play(FadeIn(dip), FadeIn(dl))
        self.wait(4.5)
        self.say("Both are smooth gates: x times a number between 0 and 1 that\ndepends on x. The next idea makes that gate a separate input.", wait=4)
        self.clear_stage()

    def glu_section(self):
        self.mark("glu")
        self.heading("GLU   (W x) ⊙ σ(V x)")
        x = node("x", CURVE).move_to(np.array([-5.6, 0.5, 0]))
        a = box("W x", 1.6, 0.9, GREY_B).move_to(np.array([-2.8, 1.6, 0]))
        b = box("V x", 1.6, 0.9, GREY_B).move_to(np.array([-2.8, -0.6, 0]))
        sg = box("σ", 0.9, 0.9, PURPLE).move_to(np.array([-0.8, -0.6, 0]))
        prod = node("⊙", ACCENT).move_to(np.array([1.0, 0.5, 0]))
        out = node("out", CURVE).move_to(np.array([3.0, 0.5, 0]))
        arrows = VGroup(
            arrow(x, a), arrow(x, b), arrow(b, sg), arrow(a, prod), arrow(sg, prod), arrow(prod, out))
        self.say("Dauphin et al. (2017) gated their language model's layers:\ntwo projections of the same input, one of them squashed into a gate.", wait=0)
        self.play(FadeIn(x), FadeIn(a), FadeIn(b), Create(arrows[:2]), run_time=1.4)
        self.play(Create(arrows[2]), FadeIn(sg))
        self.play(Create(arrows[3:5]), FadeIn(prod))
        self.play(Create(arrows[5]), FadeIn(out))
        self.wait(2)
        vals = [1.4, -0.8, 2.1, 0.6]
        gates = [0.9, 0.1, 0.5, 0.0]
        vec = VGroup()
        for k, (v, gt) in enumerate(zip(vals, gates)):
            vec.add(VGroup(
                Text(f"{v:+.1f}", font=MONO, font_size=24),
                Text("×", font=MATH, font_size=24, color=GREY_B),
                Text(f"{gt:.1f}", font=MONO, font_size=24, color=PURPLE),
                Text("=", font=MATH, font_size=24, color=GREY_B),
                Text(f"{v * gt:+.2f}", font=MONO, font_size=24, color=ACCENT),
            ).arrange(RIGHT, buff=0.18))
        vec.arrange(DOWN, buff=0.22, aligned_edge=RIGHT).move_to(np.array([5.4, -1.0, 0]))
        hdr = Text("element by element", font=SERIF, font_size=22, color=GREY_B).next_to(vec, UP, buff=0.25)
        self.say("The ⊙ is an element-wise product. Each gate value between 0 and 1\ndecides how much of the matching feature gets through.", wait=0)
        self.play(FadeIn(hdr), LaggedStart(*[FadeIn(r, shift=0.1 * LEFT) for r in vec], lag_ratio=0.25), run_time=2)
        self.wait(4)
        self.say("Unlike SiLU, where x gates itself, the gate here is learned separately,\nso the layer chooses what to pass based on the whole input.", wait=4)
        self.clear_stage()

    def swiglu_section(self):
        self.mark("swiglu")
        self.heading("SwiGLU   W₂ (SiLU(W₁ x) ⊙ W₃ x)")
        x = node("x", CURVE).move_to(np.array([-5.8, 0.6, 0]))
        w1 = box("W₁", 1.2, 0.9, GREY_B).move_to(np.array([-3.7, 1.7, 0]))
        w3 = box("W₃", 1.2, 0.9, GREY_B).move_to(np.array([-3.7, -0.5, 0]))
        act = box("SiLU", 1.4, 0.9, ACCENT).move_to(np.array([-1.6, 1.7, 0]))
        prod = node("⊙", ACCENT).move_to(np.array([0.3, 0.6, 0]))
        w2 = box("W₂", 1.2, 0.9, GREY_B).move_to(np.array([2.1, 0.6, 0]))
        out = node("y", CURVE).move_to(np.array([4.0, 0.6, 0]))
        arrows = VGroup(arrow(x, w1), arrow(x, w3), arrow(w1, act), arrow(act, prod), arrow(w3, prod),
                        arrow(prod, w2), arrow(w2, out))
        dims = VGroup(
            Text("d", font=MATH, font_size=24, color=GREY_B).next_to(x, DOWN, buff=0.15),
            Text("h", font=MATH, font_size=24, color=GREY_B).next_to(w1, UP, buff=0.12),
            Text("h", font=MATH, font_size=24, color=GREY_B).next_to(w3, DOWN, buff=0.12),
            Text("d", font=MATH, font_size=24, color=GREY_B).next_to(out, DOWN, buff=0.15),
        )
        diagram = VGroup(x, w1, w3, act, prod, w2, out, arrows, dims)
        self.say("Shazeer (2020), \"GLU Variants Improve Transformer\", put the gate\ninside the transformer's feed-forward block, with SiLU as the gate.", wait=0)
        self.play(FadeIn(x), FadeIn(dims[0]))
        self.play(Create(arrows[:2]), FadeIn(w1), FadeIn(w3), FadeIn(dims[1:3]), run_time=1.2)
        self.play(Create(arrows[2]), FadeIn(act))
        self.play(Create(arrows[3:5]), FadeIn(prod))
        self.play(Create(arrows[5:]), FadeIn(w2), FadeIn(out), FadeIn(dims[3]))
        self.wait(1.5)
        pulses = VGroup(*[Dot(radius=0.11, color=ACCENT) for _ in range(2)])
        path_a = VMobject().set_points_as_corners([x.get_center(), w1.get_center(), act.get_center(), prod.get_center()])
        path_b = VMobject().set_points_as_corners([x.get_center(), w3.get_center(), prod.get_center()])
        path_c = Line(prod.get_center(), out.get_center())
        self.say("The input goes through two projections in parallel. W₁'s copy\nbecomes the gate after SiLU; W₃'s copy is the value being gated.", wait=0)
        self.add(pulses)
        self.play(MoveAlongPath(pulses[0], path_a), MoveAlongPath(pulses[1], path_b), run_time=2.2, rate_func=linear)
        self.play(FadeOut(pulses[1]), Flash(prod, color=ACCENT), run_time=0.6)
        self.play(MoveAlongPath(pulses[0], path_c), run_time=1.2, rate_func=linear)
        self.play(FadeOut(pulses[0]))
        self.wait(2.5)
        self.play(diagram.animate.scale(0.6).move_to(np.array([0, 1.75, 0])), run_time=1)
        budget = VGroup(
            Text("standard FFN:  2 matrices of d × 4d  =  8d²", font=MATH, font_size=28),
            Text("SwiGLU FFN:    3 matrices of d × h    =  3dh", font=MATH, font_size=28),
            Text("3dh = 8d²   gives   h = 8d / 3 = (2/3) · 4d", font=MATH, font_size=28, color=ACCENT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to(np.array([0.0, -0.35, 0]))
        self.say("Why three matrices? Two make the gate and the value, one projects back.\nA plain FFN has two, so for the same parameters the hidden width must shrink.", wait=0)
        self.play(FadeIn(budget[0]))
        self.play(FadeIn(budget[1]))
        self.wait(2.5)
        self.play(FadeIn(budget[2], shift=0.1 * UP))
        self.wait(4)
        llama = Text("LLaMA-7B:  d = 4096,  (2/3) · 4d ≈ 10923,  rounded up to 11008",
                     font=MATH, font_size=26, color=GREY_B).next_to(budget, DOWN, buff=0.35)
        self.say("Shazeer cut the hidden width from 3072 to 2048 to keep the cost equal.\nLLaMA uses (2/3) · 4d, rounded up to a multiple of 256.", wait=0)
        self.play(FadeIn(llama))
        self.wait(4.5)
        users = Text("SwiGLU:  PaLM · LLaMA · Mistral · Qwen · DeepSeek", font=SERIF, font_size=30, color=ACCENT)
        users.next_to(llama, DOWN, buff=0.35)
        self.say("It trained to lower loss than ReLU or GELU feed-forward layers at equal cost.\nLLaMA, Mistral, Qwen and DeepSeek all use SwiGLU.", wait=0)
        self.play(FadeIn(users))
        self.wait(5)
        self.diagram, self.act_box, self.budget = diagram, act, VGroup(budget, llama, users)

    def geglu_section(self):
        self.mark("geglu")
        self.play(FadeOut(self.budget))
        self.heading("GeGLU   W₂ (GELU(W₁ x) ⊙ W₃ x)")
        self.play(self.diagram.animate.scale(1 / 0.6).move_to(np.array([-0.9, 0.6, 0])), run_time=1)
        new = box("GELU", 1.4, 0.9, CURVE).move_to(self.act_box)
        self.say("GeGLU is the same block with GELU as the gate instead of SiLU.\nIt comes from the same Shazeer (2020) paper.", wait=0)
        self.play(Transform(self.act_box, new), run_time=1.2)
        self.wait(3)
        users = Text("GeGLU:  Google's Gemma", font=SERIF, font_size=30, color=CURVE).move_to(np.array([0, -1.5, 0]))
        self.say("Google's Gemma models use GeGLU. Since GELU and SiLU are so alike,\nthe two variants scored almost the same in Shazeer's experiments.", wait=0)
        self.play(FadeIn(users))
        self.wait(5)
        self.clear_stage()

    def summary(self):
        self.mark("summary")
        self.heading("All together")
        ax = make_axes((-4, 3), (-1.2, 3), 7.0, 3.6, np.array([-2.6, 0.8, 0]), xticks=(-4, -2, 0, 2), yticks=(-1, 1, 2, 3))
        items = ((sigmoid, "sigmoid", GREY_B), (math.tanh, "tanh", TEAL), (relu, "ReLU", BAD),
                 (lambda v: leaky(v, 0.1), "Leaky ReLU", GOOD), (gelu, "GELU", CURVE), (silu, "SiLU", ACCENT))
        curves = VGroup(*[graph(ax, f, -4, 3).set_stroke(c, 4) for f, _, c in items])
        legend = VGroup(*[legend_row(n, c) for _, n, c in items]).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        gl = VGroup(legend_row("SwiGLU = SiLU gate", ACCENT), legend_row("GeGLU = GELU gate", CURVE))
        gl.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        side = VGroup(legend, gl).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to(np.array([4.2, 0.9, 0]))
        self.say("Every curve together. The trend runs from squashing functions\ntoward ones that let large values through and gate the small ones.", wait=0)
        self.play(Create(ax), run_time=0.8)
        self.play(LaggedStart(*[Create(c) for c in curves], lag_ratio=0.3), FadeIn(legend, lag_ratio=0.2), run_time=3)
        self.play(FadeIn(gl))
        self.wait(3)
        events = (("1986", "sigmoid\nbackprop"), ("1998", "tanh\nadvice"), ("2010", "ReLU"), ("2013", "Leaky\nReLU"),
                  ("2016", "GELU"), ("2017", "SiLU, GLU"), ("2020", "SwiGLU\nGeGLU"), ("2023", "LLaMA"))
        y0 = -1.65
        line = Line(np.array([-6.3, y0, 0]), np.array([6.3, y0, 0])).set_stroke(GREY_B, 2)
        marks = VGroup()
        for k, (yr, what) in enumerate(events):
            px = -5.8 + k * (11.6 / (len(events) - 1))
            d = Dot(np.array([px, y0, 0]), radius=0.07, color=ACCENT)
            t1 = Text(yr, font=MONO, font_size=20, color=ACCENT).next_to(d, UP, buff=0.12)
            t2 = Text(what, font=SERIF, font_size=18, color=GREY_A, line_spacing=0.8).next_to(d, DOWN, buff=0.12)
            marks.add(VGroup(d, t1, t2))
        self.say("One line of history: 1986 sigmoid, 2010 ReLU, 2016 GELU, 2017 gates,\n2020 SwiGLU and GeGLU, and since 2023 the open LLMs built on them.", wait=0)
        self.play(Create(line), LaggedStart(*[FadeIn(m, shift=0.1 * UP) for m in marks], lag_ratio=0.2), run_time=3)
        self.wait(6)
        self.play(*[FadeOut(m) for m in self.mobjects])
        self.cap, self.head = None, None
        end = Text("A modern LLM's nonlinearity is a learned gate.", font=SERIF, font_size=40)
        self.play(Write(end), run_time=1.6)
        self.wait(3)
        self.play(FadeOut(end))

    def heading(self, text):
        """Swap the section title at the top of the frame."""
        new = Text(text, font=MATH, font_size=36, color=WHITE)
        if new.width > 13:
            new.scale_to_fit_width(13)
        new.to_edge(UP, buff=0.35)
        anims = [FadeIn(new, shift=0.15 * DOWN)]
        if self.head is not None:
            anims.append(FadeOut(self.head))
        self.play(*anims, run_time=0.6)
        self.head = new

    def clear_stage(self):
        """Fade out everything but the caption and heading."""
        keep = {id(self.cap), id(self.head)}
        rest = [m for m in self.mobjects if id(m) not in keep]
        if rest:
            self.play(*[FadeOut(m) for m in rest], run_time=0.7)

    def mark(self, name):
        print(f"SECTION {name} t={self.renderer.time:.1f}")

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
        # No narration, so hold long enough to read (~20 characters a second);
        # captions that lead into an animation get half that, the animation covers the rest.
        self.wait(max(wait, len(text) / 20) if wait else len(text) / 45)


def make_axes(xr, yr, xl, yl, centre, xticks=(), yticks=()):
    """Axes with Text tick labels added as a child (no LaTeX on this machine)."""
    ax = Axes(x_range=[xr[0], xr[1], 1], y_range=[yr[0], yr[1], 1], x_length=xl, y_length=yl,
              axis_config={"include_numbers": False, "include_tip": False, "include_ticks": False,
                           "stroke_color": GREY_B, "stroke_width": 2})
    ax.move_to(centre)
    labels = VGroup()
    for v in xticks:
        if v == 0:
            continue
        p = ax.c2p(v, 0)
        labels.add(Line(p + 0.08 * DOWN, p + 0.08 * UP).set_stroke(GREY_B, 2))
        labels.add(Text(fmt(v), font=MONO, font_size=20, color=GREY_B).next_to(p, DOWN, buff=0.15))
    for v in yticks:
        p = ax.c2p(0, v)
        labels.add(Line(p + 0.08 * LEFT, p + 0.08 * RIGHT).set_stroke(GREY_B, 2))
        labels.add(Text(fmt(v), font=MONO, font_size=20, color=GREY_B).next_to(p, LEFT, buff=0.15))
    ax.add(labels)
    return ax


def graph(ax, f, a, b, n=240):
    """f on [a, b] as a polyline with a vertex exactly at 0, so kinks stay sharp."""
    k = max(2, round(n * (0 - a) / (b - a)))
    xs = np.concatenate([np.linspace(a, 0, k), np.linspace(0, b, n - k + 1)[1:]])
    return VMobject().set_points_as_corners([ax.c2p(x, f(x)) for x in xs])


def fmt(v):
    return str(int(v)) if float(v).is_integer() else str(v)


def node(label, colour):
    c = Circle(0.42).set_stroke(colour, 3).set_fill(config.background_color, 1)
    return VGroup(c, Text(label, font=MATH, font_size=30 if len(label) < 3 else 22).move_to(c))


def box(label, w, h, colour):
    r = RoundedRectangle(corner_radius=0.12, width=w, height=h).set_stroke(colour, 3).set_fill(config.background_color, 1)
    return VGroup(r, Text(label, font=MATH, font_size=30).move_to(r))


def arrow(a, b):
    """An arrow from the edge of a's frame to the edge of b's, along the line between centres."""
    pa, pb = a.get_center(), b.get_center()
    return Arrow(edge(a, pb - pa), edge(b, pa - pb), buff=0.1, stroke_width=4, color=GREY_B,
                 max_tip_length_to_length_ratio=0.15)


def edge(mob, direction):
    """Where a ray from mob's centre leaves its bounding box."""
    d = direction / np.linalg.norm(direction)
    w, h = mob[0].width / 2, mob[0].height / 2
    t = min(w / abs(d[0]) if d[0] else np.inf, h / abs(d[1]) if d[1] else np.inf)
    return mob.get_center() + t * d


def legend_row(text, colour):
    return VGroup(Line(ORIGIN, 0.5 * RIGHT).set_stroke(colour, 5),
                  Text(text, font=MATH, font_size=24)).arrange(RIGHT, buff=0.2)
