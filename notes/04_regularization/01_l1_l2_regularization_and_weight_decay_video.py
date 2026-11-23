"""Renders a Manim video on L1 and L2 regularization, and on why weight decay is not the learning rate.

Contents:
- Regularization: the scene, with one method per section of the video
- Chart: a plot frame whose ranges need not contain 0
- precompute: every number on screen, with asserts for the claims the captions make
- draw_flights, flight_strings, ridge, lasso_path, rmse: the synthetic drone dataset and the two regularized fits
- decay_run, adam_means: gradient descent with weight decay, and Adam with an L2 penalty against AdamW
- touch_l2, touch_l1, geo_loss, geo_loss_many, ellipse_arcs: the two-weight picture of a weight budget meeting the loss contours
- shrink_l2, shrink_l1: the one-weight rules each penalty applies
- index_of, bar, cached, M, num, wtxt, lam_txt, clock: lookup, drawing helpers and number formatting
- fonts, colours, the dataset's settings and the layout constants
"""
import os

import numpy as np
from manim import *

SERIF = "Cambria"
"""Description: Serif font for captions, titles and feature names; Cambria's lining figures keep "L1" from reading as "Lı"."""
MONO = "Consolas"
"""Description: Monospace font for weights, readouts and tick labels, so digits line up."""
MATH = "Cambria"
"""Description: Font for formulas, because Cambria has the Greek letters, ∑, √ and ← they need."""
RED = "#FC6255"
"""Description: Colour of plain least squares, the overfit model, and of Adam with an L2 penalty."""
BLUE = "#58C4DD"
"""Description: Colour of L2 regularization (ridge) and of AdamW."""
AMBER = "#FFD166"
"""Description: Colour of L1 regularization (lasso)."""
GREEN = "#83C167"
"""Description: Colour of the test error, measured on new flights."""
PURPLE = "#B48EDE"
"""Description: Colour of the fourth run in the learning-rate chart and of the weight-decay row."""
config.background_color = "#0b0d10"

SEED = 85
"""Description: The dataset's seed. Of the first 120 seeds tried, it is one of three where lasso keeps exactly the four real features."""
N_TRAIN = 14
"""Description: The number of logged training flights."""
N_TEST = 500
"""Description: The number of new flights used to measure test error."""
NOISE = 0.08
"""Description: The standard deviation of the drain's random part, in % per minute."""
NAMES = ("payload mass", "wind speed", "climb rate", "air temperature", "hour of day",
         "landing pad number", "day of week", "GPS satellites", "firmware build", "operator ID")
"""Description: The ten logged features, the four real ones first."""
COEF = np.array([0.30, 0.06, 0.25, -0.02, 0, 0, 0, 0, 0, 0])
"""Description: Each feature's true effect on drain in raw units (% per minute per kg, per m/s, per m/s, per °C); the last six have none."""
REAL = COEF != 0
"""Description: Which features truly affect the drain."""
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
"""Description: Day-of-week names for the sample flight."""
FINE = np.linspace(-4.0, 0.0, 801)
"""Description: The grid of log10 λ values for the ridge and lasso paths, from 0.0001 to 1."""
A_1D, LAM_1D = 0.3, 0.5
"""Description: In the one-weight example, the weight the data alone would choose and the penalty strength."""
GEO_A = np.array([[1.0, 0.5], [0.5, 0.7]])
"""Description: The curvature of the two-weight training loss, which tilts and stretches its contour ellipses."""
GEO_WS = np.array([1.5, 1.0])
"""Description: The least-squares weights in the two-weight picture, the centre of the ellipses."""
GEO_T = 1.0
"""Description: The weight budget in the two-weight picture: the disk's radius and the diamond's half-diagonal."""
TOY_TARGET = 2.0
"""Description: In the decay chart, the weight the data alone would choose: the loss is ½(w − 2)²."""
TOY_RUNS = ((0.1, 0.0, GREY_B), (0.3, 0.0, BLUE), (0.1, 0.25, AMBER), (0.3, 0.25, PURPLE))
"""Description: The (learning rate η, decay λ, colour) of each run in the decay chart."""
TOY_STEPS = 80
"""Description: How many steps each run in the decay chart takes, enough for the slowest run to settle within 0.001."""
DRONE_ETAS = (0.1, 0.3)
"""Description: The two learning rates used to check that decay on the drone data lands on the ridge weights."""
ADAM_ETA, ADAM_LAM, ADAM_STEPS, ADAM_RUNS = 0.01, 0.1, 2000, 1000
"""Description: The Adam demo's learning rate, decay strength, step count, and number of runs averaged."""
ADAM_SIGMA = np.array([10.0, 0.1])
"""Description: The typical gradient size of weight A and of weight B in the Adam demo; both gradients are pure noise."""

NAME_R = -3.35
"""Description: Screen x of the right edge of the feature names."""
X0 = -1.75
"""Description: Screen x of the weight chart's zero line."""
SCALE = 5.0
"""Description: Screen units per unit of weight in the bar chart."""
VAL_R = 0.85
"""Description: Screen x of the right edge of the weight numbers."""
ROW_Y = 2.3 - 0.43 * np.arange(len(NAMES))
"""Description: Screen y of each feature's row."""
BAR_H = 0.24
"""Description: Height of a weight bar."""
TEXT_CACHE = {}
"""Description: Text mobjects already built, keyed by (kind, text, size, colour), for cached()."""


class Regularization(Scene):
    """
    Description: The video's scene. Each section method draws one part of the video and ends on an empty frame.
    Fields:
    - cap: the caption on screen, or None.
    - head: the section heading on screen, or None.
    - d: the numbers from precompute().
    """

    def construct(self):
        """
        Description: Computes the numbers, then plays the sections in order.
        Inputs: none
        Outputs: none
        Constraints: when the REG_ONLY environment variable names sections, separated by commas, only those play.
        """
        self.cap = None
        self.head = None
        self.d = precompute()
        only = [s for s in os.environ.get("REG_ONLY", "").split(",") if s]
        for name in ("intro", "overfit", "ridge_l2", "lasso_l1", "why_zero", "geometry", "decay", "adam", "summary"):
            if not only or name in only:
                getattr(self, name)()

    def intro(self):
        """
        Description: Shows the title, then sets L1 and L2 loss beside L1 and L2 regularization.
        Inputs: none
        Outputs: none
        """
        self.mark("intro")
        title = Text("L1 and L2 regularization", font=SERIF, font_size=50)
        sub = Text("and why weight decay is not the learning rate", font=SERIF, font_size=30, color=GREY_B)
        sub.next_to(title, DOWN, buff=0.3)
        words = VGroup(title, sub).move_to(0.4 * UP)
        self.play(Write(title), FadeIn(sub, shift=0.2 * UP), run_time=1.6)
        self.wait(2.5)
        self.play(FadeOut(words))
        self.heading("Two measures, applied to two different things")
        lh = M("loss: how wrong the predictions are", 28, GREY_B)
        l2 = M("L2 loss    ∑<sub>i</sub> (y<sub>i</sub> − ŷ<sub>i</sub>)<sup>2</sup>", 34)
        l1 = M("L1 loss    ∑<sub>i</sub> |y<sub>i</sub> − ŷ<sub>i</sub>|", 34)
        left = VGroup(lh, l2, l1).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to(np.array([-3.55, 1.0, 0]))
        rh = M("regularization: how large the weights are", 28, GREY_B)
        r2 = M("L2 regularization    (λ/2) ∑<sub>j</sub> w<sub>j</sub><sup>2</sup>", 34, BLUE)
        r1 = M("L1 regularization    λ ∑<sub>j</sub> |w<sub>j</sub>|", 34, AMBER)
        right = VGroup(rh, r2, r1).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to(np.array([3.35, 1.0, 0]))
        obj = M("objective to minimize   =   loss   +   regularization", 34).move_to(1.35 * DOWN)
        self.say("L1 and L2 loss measure how wrong a model's predictions are. L1 and L2 regularization\n"
                 "apply the same two measures to the model's weights instead.", wait=0)
        self.play(FadeIn(left, lag_ratio=0.25), run_time=1.6)
        self.play(FadeIn(right, lag_ratio=0.25), run_time=1.6)
        self.wait(4)
        self.say("Here y<sub>i</sub> is the measured value for example i, and ŷ<sub>i</sub>, read y-hat, is the model's\n"
                 "prediction for it. w<sub>j</sub> is the weight the model puts on feature j.")
        self.say("λ, read lambda, is a number you choose. It sets how much the penalty counts against the loss,\n"
                 "and training minimizes the sum of the two.", wait=0)
        self.play(FadeIn(obj, shift=0.15 * UP))
        self.wait(5)
        self.say("The loss pulls the weights toward fitting the training data, and the penalty pulls them\n"
                 "toward 0. λ sets the balance between the two pulls.")
        self.end_section()

    def overfit(self):
        """
        Description: Shows one logged flight, then the overfit least-squares weights and their train and test errors.
        Inputs: none
        Outputs: none
        """
        self.mark("overfit")
        d = self.d
        self.heading("A practical case: predicting a drone's battery drain")
        names = self.name_column()
        raw = VGroup()
        for j, v in enumerate(d["flight1"]):
            t = Text(v, font=MONO, font_size=22, color=WHITE if REAL[j] else GREY_B)
            raw.add(t.move_to(np.array([X0 + 0.25, ROW_Y[j], 0]), aligned_edge=LEFT))
        y_d = ROW_Y[-1] - 0.55
        drain = VGroup(Text("battery drain", font=SERIF, font_size=22, color=GREEN)
                       .move_to(np.array([NAME_R, y_d, 0]), aligned_edge=RIGHT),
                       Text(f"{d['drain1']:.2f} % per minute", font=MONO, font_size=22, color=GREEN)
                       .move_to(np.array([X0 + 0.25, y_d, 0]), aligned_edge=LEFT))
        tag = Text(f"flight 1 of {N_TRAIN}", font=SERIF, font_size=24, color=GREY_B).move_to(np.array([X0 - 0.6, 2.95, 0]))
        self.say(f"A delivery company logged {N_TRAIN} flights. For each one it recorded ten features and the\n"
                 "battery drain in percent per minute. The data here are synthetic.", wait=0)
        self.play(FadeIn(tag), FadeIn(names, lag_ratio=0.08), FadeIn(raw, lag_ratio=0.08), FadeIn(drain), run_time=2)
        self.wait(5)
        self.say("Four features plausibly matter: payload, wind, climb rate and temperature. The other six have\n"
                 "no effect, because the simulation that made this data ignores them.")
        self.say("First each feature is rescaled to mean 0 and standard deviation 1. Then a weight of 0.1 means\n"
                 "the same size of effect for every feature, and one penalty treats them all alike.", wait=0)
        axes = self.weight_axes()
        bars = self.bars(d["w_ols"], RED)
        vals = self.values(d["w_ols"], RED)
        self.play(FadeOut(raw), FadeOut(drain), FadeOut(tag), run_time=0.8)
        self.play(FadeIn(axes[1:]), run_time=1)
        self.wait(4)
        self.say(f"Plain least squares picks the ten weights, plus an average, that best fit the {N_TRAIN} flights.\n"
                 "The white outlines mark the true effects.", wait=0)
        self.play(LaggedStart(*[GrowFromPoint(b, np.array([X0, ROW_Y[j], 0])) for j, b in enumerate(bars)], lag_ratio=0.1),
                  FadeIn(vals, lag_ratio=0.1), run_time=2)
        self.bring_to_front(axes[2])
        self.wait(5)
        panel = self.error_bars(d["ols_tr"], d["ols_te"])
        self.say(f"The typical miss on the {N_TRAIN} training flights is {d['ols_tr']:.3f} % per minute. On {N_TEST} new flights\n"
                 f"it is {d['ols_te']:.3f}, about {d['ols_ratio']:.0f} times larger. Typical miss here means root-mean-square error.", wait=0)
        self.play(FadeIn(panel[0]), LaggedStart(*[FadeIn(p, shift=0.2 * RIGHT) for p in panel[1:]], lag_ratio=0.5), run_time=2)
        self.wait(8)
        (n1, v1), (n2, v2) = d["ols_top_useless"]
        self.say(f"Eleven numbers fit to {N_TRAIN} flights leave almost no slack, so the fit bends to the noise. {n1.capitalize()}\n"
                 f"gets a weight of {num(v1, '.3f')} and {n2} {num(v2, '.3f')}, though neither affects the battery. That is overfitting.")
        self.end_section()

    def ridge_l2(self):
        """
        Description: Sweeps λ for L2 regularization, showing the weights shrinking and the U-shaped test error.
        Inputs: none
        Outputs: none
        """
        self.mark("ridge_l2")
        d = self.d
        self.heading("L2 regularization (ridge): penalize the squared weights")
        self.sweep(d["ridge_w"], d["ridge_tr"], d["ridge_te"], BLUE,
                   "J(w) = ½ · (1/n) ∑<sub>i</sub> (y<sub>i</sub> − ŷ<sub>i</sub>)<sup>2</sup>  +  "
                   "(λ/2) ∑<sub>j</sub> w<sub>j</sub><sup>2</sup>", d["k_ridge"], zeros=False,
                   captions=(
                       "L2 regularization adds (λ/2) times the sum of the squared weights. The ½ only rescales λ:\n"
                       f"it makes the penalty's gradient exactly λw. Here n = {N_TRAIN}, the number of flights.",
                       "At λ = 0.0001 the penalty barely counts, so the fit is almost plain least squares.\n"
                       "Now raise λ and watch the weights.",
                       "As λ grows the weights shrink toward 0 as a group. Single weights can drift or even cross 0,\n"
                       "but their total size falls the whole way, and none of them settles at exactly 0.",
                       f"Training error only rises with λ, because the penalty stops the weights from fitting the {N_TRAIN} flights.\n"
                       "Test error first falls, then rises: too little penalty overfits, and too much underfits.",
                       f"The lowest test error, {d['ridge_te'][d['k_ridge']]:.3f}, comes at λ = {lam_txt(10 ** FINE[d['k_ridge']])}. "
                       f"All ten weights are still nonzero there;\nthe six useless ones range from {d['ridge_useless_min']:.3f} "
                       f"to {d['ridge_useless_max']:.3f} in size.",
                       "In practice you would choose λ with held-out validation flights, not the test flights.\n"
                       "Arthur Hoerl and Robert Kennard published this method as ridge regression in 1970.",
                   ))
        self.end_section()

    def lasso_l1(self):
        """
        Description: Sweeps λ for L1 regularization, showing useless weights reaching exactly 0 and staying there.
        Inputs: none
        Outputs: none
        """
        self.mark("lasso_l1")
        d = self.d
        self.heading("L1 regularization (lasso): penalize the absolute weights")
        k = d["k_lasso"]
        self.sweep(d["lasso_w"], d["lasso_tr"], d["lasso_te"], AMBER,
                   "J(w) = ½ · (1/n) ∑<sub>i</sub> (y<sub>i</sub> − ŷ<sub>i</sub>)<sup>2</sup>  +  λ ∑<sub>j</sub> |w<sub>j</sub>|",
                   k, zeros=True,
                   captions=(
                       "L1 regularization adds λ times the sum of the absolute weights instead. λ values for L1 and L2\n"
                       "sit on different scales, so they cannot be compared with each other.",
                       "Start again at λ = 0.0001, almost plain least squares, and raise λ.",
                       "As λ grows, the six useless weights reach exactly 0 one at a time, well before the real ones.\n"
                       "A weight of exactly 0 means the model ignores that feature.",
                       f"Past λ = {d['lam_max']:.2f} every weight is 0, and the model predicts the average drain for every flight.\n"
                       f"On the way, {d['lasso_back'][0]} left 0 briefly, between λ = {d['lasso_back'][1]:.3f} and "
                       f"{d['lasso_back'][2]:.3f}, with a weight under {d['lasso_back'][3]:.3f}.",
                       f"At the best λ, {lam_txt(10 ** FINE[k])}, the six useless weights are exactly 0 and the four real ones remain.\n"
                       f"Test error is {d['lasso_te'][k]:.3f}, lower than ridge's {d['ridge_te'][d['k_ridge']]:.3f} on this data.",
                       f"This dataset was chosen to show the effect cleanly. With only {N_TRAIN} flights, lasso often keeps a\n"
                       "useless feature or drops a real one. Robert Tibshirani introduced the lasso in 1996.",
                   ))
        self.end_section()

    def why_zero(self):
        """
        Description: Explains exact zeros with one weight: the V-shaped penalty's corner, then the two shrink rules.
        Inputs: none
        Outputs: none
        """
        self.mark("why_zero")
        d = self.d
        self.heading("Why L1 gives exact zeros and L2 does not")
        a, lam = A_1D, LAM_1D
        panels = VGroup()
        sums = []
        for cx, kind, col in ((-3.45, "l2", BLUE), (3.45, "l1", AMBER)):
            ch = Chart((-0.5, 1.0), (0, 0.5), 5.0, 2.7, np.array([cx, 0.25, 0]), xticks=(-0.5, 0, 0.5, 1),
                       yticks=(0, 0.25, 0.5), tick_size=18)
            pen = (lambda w: lam / 2 * w * w) if kind == "l2" else (lambda w: lam * abs(w))
            fit = DashedVMobject(ch.curve(lambda w: 0.5 * (w - a) ** 2, -0.5, 1.0), num_dashes=40).set_stroke(GREY_B, 2.5)
            pc = DashedVMobject(ch.curve(pen, -0.5, 1.0), num_dashes=40).set_stroke(col, 2.5)
            tot = ch.curve(lambda w, p=pen: 0.5 * (w - a) ** 2 + p(w), -0.5, 1.0, n=601).set_stroke(col, 5)
            zl = DashedLine(ch.c2p(0, 0), ch.c2p(0, 0.5), dash_length=0.06).set_stroke(GREY_D, 1.5)
            label = M("½(w − a)<sup>2</sup>  +  (λ/2) w<sup>2</sup>" if kind == "l2" else "½(w − a)<sup>2</sup>  +  λ |w|", 28, col)
            label.next_to(ch.group, UP, buff=0.3)
            wmin = d["l2_at_a"] if kind == "l2" else d["l1_at_a"]
            pt = ch.c2p(wmin, 0.5 * (wmin - a) ** 2 + pen(wmin))
            dot = Dot(pt, radius=0.09, color=WHITE)
            txt = M(f"lowest point: w = {wmin:.1f}" if wmin else "lowest point: w = 0 exactly", 24, WHITE)
            txt.next_to(ch.c2p(wmin, 0), DOWN, buff=0.55)
            key = VGroup(M("dashed grey: fit term", 20, GREY_B), M("dashed colour: penalty", 20, col),
                         M("solid: their sum", 20, col)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
            key.next_to(ch.c2p(0.22, 0.5), DOWN, buff=0.08)
            panels.add(VGroup(ch.group, zl, label, fit, pc, key))
            sums.append((tot, dot, txt))
        self.say("Take a single weight w. Fitting the data alone would set it to a = 0.3, so the fit term is\n"
                 "½(w − a)<sup>2</sup>. Add a penalty with λ = 0.5 and look for the lowest point of the sum.", wait=0)
        self.play(FadeIn(panels, lag_ratio=0.1), run_time=2)
        self.wait(6)
        tot, dot, txt = sums[0]
        self.say("L2's penalty is a smooth bowl, flat at 0. Near 0 its pull fades away, so the lowest point of\n"
                 f"the sum lands at w = {d['l2_at_a']:.1f}: smaller than 0.3, but not 0.", wait=0)
        self.play(Create(tot), run_time=1.5)
        self.play(FadeIn(dot, scale=1.5), FadeIn(txt))
        self.wait(7)
        tot, dot, txt = sums[1]
        self.say("L1's penalty is a V with a sharp corner at 0, and its slope is λ = 0.5 on either side. The data\n"
                 "pulls with a slope of only 0.3, so the lowest point of the sum sits exactly at the corner, w = 0.", wait=0)
        self.play(Create(tot), run_time=1.5)
        self.play(FadeIn(dot, scale=1.5), FadeIn(txt))
        self.wait(8)
        self.clear_stage()

        ch = Chart((-1.2, 1.2), (-1.2, 1.2), 3.9, 3.9, np.array([-2.75, 0.25, 0]), xticks=(-1, 0, 1), yticks=(-1, 0, 1), tick_size=18)
        through = VGroup(Line(ch.c2p(-1.2, 0), ch.c2p(1.2, 0)), Line(ch.c2p(0, -1.2), ch.c2p(0, 1.2))).set_stroke(GREY_D, 1.5)
        xl = M("a: the weight the data alone would choose", 22, GREY_B).next_to(ch.group, DOWN, buff=0.12)
        yl = M("w: the weight after the penalty", 22, GREY_B).rotate(PI / 2).next_to(ch.group, LEFT, buff=0.12)
        band = Polygon(ch.c2p(-lam, -1.2), ch.c2p(lam, -1.2), ch.c2p(lam, 1.2), ch.c2p(-lam, 1.2))
        band.set_stroke(width=0).set_fill(AMBER, 0.1)
        band_t = M("L1 gives exactly 0 here", 20, AMBER).next_to(ch.c2p(0, 1.2), DOWN, buff=0.1)
        ident = DashedVMobject(ch.curve(lambda v: v, -1.2, 1.2), num_dashes=30).set_stroke(GREY_B, 2)
        l2c = ch.curve(lambda v: shrink_l2(v, lam), -1.2, 1.2, n=481).set_stroke(BLUE, 5)
        l1c = ch.curve(lambda v: shrink_l1(v, lam), -1.2, 1.2, n=481).set_stroke(AMBER, 5)
        rules = VGroup(M("no penalty:   w = a", 28, GREY_B), M("L2:   w = a / (1 + λ)", 28, BLUE),
                       M("L1:   w = sign(a) · max(|a| − λ, 0)", 28, AMBER), M("λ = 0.5", 28, GREY_B))
        rules.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(np.array([3.4, 1.25, 0]))
        at = ValueTracker(-1.2)
        probe = always_redraw(lambda: VGroup(
            DashedLine(ch.c2p(at.get_value(), -1.2), ch.c2p(at.get_value(), 1.2), dash_length=0.05).set_stroke(WHITE, 1.5),
            Dot(ch.c2p(at.get_value(), shrink_l2(at.get_value(), lam)), radius=0.08, color=BLUE),
            Dot(ch.c2p(at.get_value(), shrink_l1(at.get_value(), lam)), radius=0.08, color=AMBER)))
        read = always_redraw(lambda: VGroup(
            cached("math", f"a = {num(at.get_value(), '.2f')}", 28, WHITE),
            cached("math", f"L2 gives  w = {num(shrink_l2(at.get_value(), lam), '.2f')}", 28, BLUE),
            cached("math", f"L1 gives  w = {num(shrink_l1(at.get_value(), lam), '.2f')}", 28, AMBER))
            .arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to(np.array([3.4, -0.95, 0]), aligned_edge=LEFT)
            .shift((rules.get_left()[0] - 3.4) * RIGHT))
        self.say("Doing this for every value of a gives each penalty's rule. L2 divides: the weight keeps\n"
                 "1/(1 + λ) of its size, here two thirds, however small it is.", wait=0)
        self.play(FadeIn(ch.group), FadeIn(through), FadeIn(xl), FadeIn(yl), FadeIn(ident), run_time=1)
        self.play(Create(l2c), FadeIn(rules[0]), FadeIn(rules[1]), FadeIn(rules[3]), run_time=1.5)
        self.wait(5)
        self.say("L1 subtracts: it takes λ off the size of every weight, and any weight the data puts within λ of 0\n"
                 f"lands exactly on 0. At a = 0.9, L2 gives {d['l2_at_09']:.1f} and L1 gives {d['l1_at_09']:.1f}.", wait=0)
        self.play(FadeIn(band), FadeIn(band_t), Create(l1c), FadeIn(rules[2]), run_time=1.5)
        self.add(probe, read)
        self.play(at.animate.set_value(1.2), run_time=5, rate_func=linear)
        self.play(at.animate.set_value(0.9), run_time=1)
        self.wait(5)
        self.say("The lasso fit earlier applied this rule to one weight at a time, with a computed from the other\n"
                 "weights' current values, cycling until nothing changed. That method is called coordinate descent.")
        self.end_section()

    def geometry(self):
        """
        Description: Shows the two-weight picture: loss ellipses growing until they touch a disk and a diamond.
        Inputs: none
        Outputs: none
        """
        self.mark("geometry")
        d = self.d
        self.heading("The same idea with two weights")
        s = 1.2
        frames, grow, finals = VGroup(), [], []
        for cx, kind, col in ((-3.55, "l2", BLUE), (3.55, "l1", AMBER)):
            ch = Chart((-1.3, 2.3), (-1.3, 2.0), 3.6 * s, 3.3 * s, np.array([cx, -0.2, 0]))
            axes = VGroup(Line(ch.c2p(-1.3, 0), ch.c2p(2.3, 0)), Line(ch.c2p(0, -1.3), ch.c2p(0, 2.0))).set_stroke(GREY_B, 2)
            names = VGroup(M("w<sub>1</sub>", 24, GREY_B).next_to(ch.c2p(2.3, 0), DOWN, buff=0.1),
                           M("w<sub>2</sub>", 24, GREY_B).next_to(ch.c2p(0, 2.0), LEFT, buff=0.1))
            if kind == "l2":
                region = Circle(radius=GEO_T * s).move_to(ch.c2p(0, 0))
                title = M("L2 budget:   w<sub>1</sub><sup>2</sup> + w<sub>2</sub><sup>2</sup> ≤ t<sup>2</sup>", 28, col)
                touch, level = d["geo_l2"], d["geo_c2"]
            else:
                region = Polygon(ch.c2p(GEO_T, 0), ch.c2p(0, GEO_T), ch.c2p(-GEO_T, 0), ch.c2p(0, -GEO_T))
                title = M("L1 budget:   |w<sub>1</sub>| + |w<sub>2</sub>| ≤ t", 28, col)
                touch, level = d["geo_l1"], d["geo_c1"]
            region.set_stroke(col, 3).set_fill(col, 0.18)
            title.next_to(ch.c2p(0.5, 2.0), UP, buff=0.25)
            ws = Dot(ch.c2p(*GEO_WS), radius=0.07, color=WHITE)
            ws_t = M("least squares", 20, GREY_B).next_to(ws, UR, buff=0.05)
            rings = VGroup(*[ellipse_arcs(ch, c).set_stroke(GREY_B, 1.5, opacity=0.6) for c in (0.04, 0.15)])
            tr = ValueTracker(0.01)
            moving = always_redraw(lambda ch=ch, tr=tr: ellipse_arcs(ch, tr.get_value()).set_stroke(WHITE, 3))
            tdot = Dot(ch.c2p(*touch), radius=0.09, color=col)
            if kind == "l2":
                tl = M(f"({touch[0]:.2f}, {touch[1]:.2f})", 24, col).next_to(tdot, UL, buff=0.05)
            else:
                tl = M(f"({touch[0]:.0f}, {touch[1]:.0f}):  w<sub>2</sub> = 0 exactly", 24, col).next_to(tdot, DOWN, buff=0.15)
            tl.add_background_rectangle(color=config.background_color, opacity=0.85, buff=0.06)
            frames.add(VGroup(axes, names, region, title, rings, ws, ws_t))
            grow.append((tr, moving, level))
            finals.append(VGroup(tdot, tl))
        self.say("Adding λ times a penalty gives the same answer as capping the penalty at some budget t, and\n"
                 "a larger λ means a smaller budget. With two weights, the budget is a region you can draw.", wait=0)
        self.play(FadeIn(frames, lag_ratio=0.05), run_time=2)
        self.wait(6)
        self.say("The ellipses are contours of equal training loss around the least-squares weights. The best\n"
                 "weights within budget sit where the smallest ellipse first touches the allowed region.")
        tr, moving, level = grow[0]
        self.add(moving)
        self.say("For L2 the region is a disk. It has no corners, so the touching point here has both weights\n"
                 f"nonzero: w<sub>1</sub> = {d['geo_l2'][0]:.2f} and w<sub>2</sub> = {d['geo_l2'][1]:.2f}.", wait=0)
        self.play(tr.animate.set_value(level), run_time=3, rate_func=smooth)
        self.play(FadeIn(finals[0], scale=1.3))
        self.wait(6)
        tr, moving, level = grow[1]
        self.add(moving)
        self.say("For L1 the region is a diamond whose corners lie on the axes. Here the ellipse touches the corner\n"
                 "at (1, 0), so w<sub>2</sub> is exactly 0. With many weights the diamond has many corners and edges,\n"
                 "so this happens often.", wait=0)
        self.play(tr.animate.set_value(level), run_time=3, rate_func=smooth)
        self.play(FadeIn(finals[1], scale=1.3))
        self.wait(9)
        self.end_section()

    def decay(self):
        """
        Description: Shows the weight-decay update, then four runs that separate the learning rate's job from λ's.
        Inputs: none
        Outputs: none
        """
        self.mark("decay")
        d = self.d
        self.heading("Weight decay and the learning rate are different knobs")
        f1 = M("w  ←  w  −  η ∇L(w)  −  η λ w", 34)
        f2 = M("=  (1 − ηλ) w  −  η ∇L(w)", 34)
        f2.next_to(f1, DOWN, buff=0.25).align_to(f1, LEFT).shift(0.75 * RIGHT)
        defs = M("η: learning rate, the step size      λ: decay strength      ∇L(w): gradient of the training loss", 22, GREY_B)
        top = VGroup(f1, f2).move_to(np.array([0, 2.3, 0]))
        defs.next_to(top, DOWN, buff=0.3)
        self.say("Weight decay changes the update step. Each step moves the weight down the gradient by an amount\n"
                 "set by the learning rate η, read eta, and also shrinks it a little, by an amount set by λ.", wait=0)
        self.play(FadeIn(f1, shift=0.1 * DOWN), FadeIn(defs), run_time=1.2)
        self.wait(7)
        self.say("Rearranged, each step first multiplies the weight by (1 − ηλ), then takes the usual gradient step.\n"
                 f"With η = 0.1 and λ = 0.25 that factor is {d['toy_factor']:.3f}.", wait=0)
        self.play(FadeIn(f2, shift=0.1 * DOWN))
        self.wait(7)
        ch = Chart((0, TOY_STEPS), (0, 2.2), 7.4, 2.45, np.array([-1.0, -0.75, 0]), xticks=(0, 20, 40, 60, 80),
                   yticks=(0, 1, 2), tick_size=18)
        xl = M("step", 22, GREY_B).next_to(ch.c2p(TOY_STEPS / 2, 0), DOWN, buff=0.45)
        lines = VGroup()
        for v in (TOY_TARGET, TOY_TARGET / (1 + 0.25)):
            lines.add(DashedLine(ch.c2p(0, v), ch.c2p(TOY_STEPS, v), dash_length=0.07).set_stroke(GREY_D, 1.5))
        ends = VGroup(M(f"λ = 0: ends at {TOY_TARGET:.1f}", 24, GREY_B).next_to(ch.c2p(TOY_STEPS, TOY_TARGET), RIGHT, buff=0.2),
                      M(f"λ = 0.25: ends at {TOY_TARGET / 1.25:.1f}", 24, AMBER).next_to(ch.c2p(TOY_STEPS, TOY_TARGET / 1.25), RIGHT, buff=0.2))
        runs, keys = VGroup(), VGroup()
        for (eta, lam, col), path in zip(TOY_RUNS, d["toy_paths"]):
            runs.add(VMobject().set_points_as_corners([ch.c2p(k, v) for k, v in enumerate(path)]).set_stroke(col, 4))
            keys.add(M(f"η = {eta},  λ = {lam}", 20, col))
        keys.arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to(ch.c2p(0.72 * TOY_STEPS, 0.8))
        self.say("Take one weight whose loss is ½(w − 2)<sup>2</sup>, so the data alone would put it at 2. Each line is\n"
                 "gradient descent from w = 0 with a different η and λ.", wait=0)
        self.play(FadeOut(defs), FadeIn(ch.group), FadeIn(xl), FadeIn(lines), run_time=1)
        self.play(LaggedStart(*[Create(r) for r in runs], lag_ratio=0.4), FadeIn(keys, lag_ratio=0.4), run_time=4)
        self.play(FadeIn(ends))
        self.wait(5)
        self.say("Raising η from 0.1 to 0.3 makes the weight arrive sooner, at the same place, 2.0. Raising λ\n"
                 "from 0 to 0.25 moves where it ends up, to 1.6.")
        self.say("The resting point solves ∇L(w) + λw = 0, which has no η in it. It is also the lowest point of\n"
                 "L(w) + (λ/2)w<sup>2</sup>, so under plain gradient descent, weight decay is L2 regularization.")
        self.say(f"On the {N_TRAIN} drone flights, gradient descent with decay λ = {lam_txt(10 ** FINE[d['k_ridge']])} ends on the ridge weights from\n"
                 "before, to within 10<sup>−9</sup> for every weight, whether η is 0.1 or 0.3.")
        self.end_section()

    def adam(self):
        """
        Description: Shows why an L2 penalty under Adam differs from AdamW's decay, with two noise-driven weights.
        Inputs: none
        Outputs: none
        """
        self.mark("adam")
        d = self.d
        self.heading("Under Adam, an L2 penalty and weight decay differ")
        a1 = M("Adam + L2:    g = ∇L(w) + λw,    then    w ← w − η · m̂ / √v̂", 28, RED)
        a2 = M("AdamW:    w ← w − ηλw,    then    w ← w − η · m̂ / √v̂   with   g = ∇L(w)", 28, BLUE)
        a3 = M("m̂: running average of the gradient g        v̂: running average of g<sup>2</sup>", 22, GREY_B)
        top = VGroup(a1, a2, a3).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(np.array([0, 2.25, 0]))
        self.say("Adam, the usual optimizer for large networks, divides each weight's step by a running estimate\n"
                 "of that weight's gradient size. That changes what an L2 penalty does.", wait=0)
        self.play(FadeIn(a1, shift=0.1 * DOWN), FadeIn(a3), run_time=1.2)
        self.wait(5)
        self.say("With an L2 penalty, λw joins the gradient before Adam divides, so the penalty is divided too.\n"
                 "AdamW shrinks the weight directly, outside Adam's division. That is how PyTorch's AdamW does it.", wait=0)
        self.play(FadeIn(a2, shift=0.1 * DOWN))
        self.wait(7)
        ch = Chart((0, ADAM_STEPS), (0, 1.05), 7.0, 2.4, np.array([-1.3, -0.85, 0]), xticks=(0, 500, 1000, 1500, 2000),
                   yticks=(0, 0.5, 1), tick_size=18)
        xl = M("step", 22, GREY_B).next_to(ch.c2p(ADAM_STEPS / 2, 0), DOWN, buff=0.45)
        yl = M("average weight", 22, GREY_B).next_to(ch.c2p(0, 1.05), UP, buff=0.12)
        curves, labels = VGroup(), VGroup()
        k = np.arange(ADAM_STEPS + 1)
        spec = (("adam_l2", 0, RED, "Adam + L2, weight A", 0), ("adamw", 0, BLUE, "AdamW, weights A and B", 0.12),
                ("adamw", 1, BLUE, None, 0), ("adam_l2", 1, RED, "Adam + L2, weight B", -0.14))
        for mode, j, col, text, dy in spec:
            mean = d["adam"][mode][:, j]
            curve = VMobject().set_points_as_corners([ch.c2p(x, y) for x, y in zip(k[::10], mean[::10])])
            curves.add(curve.set_stroke(col, 3.5) if j == 0 else DashedVMobject(curve, num_dashes=60).set_stroke(col, 3.5))
            if text:
                labels.add(M(text, 22, col).next_to(ch.c2p(ADAM_STEPS, mean[-1]), RIGHT, buff=0.5).shift(dy * UP))
        self.say("Take two weights the data has no use for, so the penalty should shrink both. Weight A gets large,\n"
                 "noisy gradients of typical size 10. Weight B gets small ones, of typical size 0.1.", wait=0)
        self.play(FadeIn(ch.group), FadeIn(xl), FadeIn(yl), run_time=1)
        self.wait(5)
        self.say(f"Each line averages {ADAM_RUNS:,} runs with η = {ADAM_ETA} and λ = {ADAM_LAM}. Under AdamW both weights shrink by\n"
                 f"the same factor, {1 - ADAM_ETA * ADAM_LAM:.3f} per step, to about {d['adamw_end']:.2f} after {ADAM_STEPS:,} steps.", wait=0)
        self.play(Create(curves[1]), Create(curves[2]), FadeIn(labels[1]), run_time=3)
        self.wait(6)
        self.say(f"Under Adam with an L2 penalty, B drops to about 0 while A keeps {d['adam_l2_a']:.2f} of its size. A's large\n"
                 "gradients dilute the penalty's pull, so the weights whose gradients are largest are regularized least.", wait=0)
        self.play(Create(curves[0]), Create(curves[3]), FadeIn(labels[0]), FadeIn(labels[2]), run_time=3)
        self.wait(8)
        self.say("Ilya Loshchilov and Frank Hutter pointed this out in 2017 and proposed AdamW, published at ICLR 2019.\n"
                 "AdamW is now the standard way to train transformers with weight decay.")
        self.say("In PyTorch, Adam(weight_decay=λ) adds λw to the gradient, the L2 version. AdamW(weight_decay=λ)\n"
                 "decays the weights directly, and its default λ is 0.01.")
        self.end_section()

    def summary(self):
        """
        Description: Shows the closing table of the six terms and what each one does.
        Inputs: none
        Outputs: none
        """
        self.mark("summary")
        self.heading("Summary")
        rows = (("term", "penalizes or sets", "what it does", GREY_B),
                ("L2 loss", "squared errors, (y − ŷ)<sup>2</sup>", "big misses count most", WHITE),
                ("L1 loss", "absolute errors, |y − ŷ|", "less swayed by outliers", WHITE),
                ("L2 regularization", "squared weights, (λ/2) ∑ w<sup>2</sup>", "shrinks every weight, none to exactly 0", BLUE),
                ("L1 regularization", "absolute weights, λ ∑ |w|", "sets weak weights to exactly 0", AMBER),
                ("weight decay", "shrinks w by (1 − ηλ) each step", "equals L2 under gradient descent, not Adam", PURPLE),
                ("learning rate η", "the step size", "sets the speed, not where weights settle", GREY_A))
        table = VGroup()
        for i, (a, b, c, col) in enumerate(rows):
            y = 2.55 - 0.66 * i
            size = 24 if i else 22
            cells = VGroup()
            for text, x, wmax in ((a, -6.6, 2.8), (b, -3.7, 4.0), (c, 0.55, 6.3)):
                cell = M(text, size, col)
                if cell.width > wmax:
                    cell.scale_to_fit_width(wmax)
                cells.add(cell.move_to(np.array([x, y, 0]), aligned_edge=LEFT))
            table.add(cells)
        rule = Line(np.array([-6.6, 2.22, 0]), np.array([6.85, 2.22, 0])).set_stroke(GREY_D, 1.5)
        self.say("Loss measures the errors and regularization measures the weights. Weight decay applies the L2\n"
                 "pull step by step, and the learning rate only sets the step size.", wait=0)
        self.play(FadeIn(table[0]), Create(rule), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(r, shift=0.1 * UP) for r in table[1:]], lag_ratio=0.35), run_time=3)
        self.wait(7)
        self.say("In probability terms, the L2 penalty assumes before training that each weight follows a bell curve\n"
                 "centred at 0. The L1 penalty assumes a sharper peak at 0, the Laplace distribution.")
        self.wait(2)
        self.end_section()

    def sweep(self, path_w, path_tr, path_te, colour, formula, k_best, zeros, captions):
        """
        Description: Plays a λ sweep: weight bars and numbers on the left, train and test error against λ on the right.
        Inputs:
        - path_w: the weights at each λ in FINE, shape (len(FINE), 10).
        - path_tr, path_te: the training and test errors at each λ in FINE.
        - colour: the method's colour.
        - formula: the objective, as Pango markup.
        - k_best: the index in FINE of the λ with the lowest test error.
        - zeros: whether to show the count of weights that are exactly 0.
        - captions: six captions, in order: formula, start, sweep, curves, best λ, closing note.
        Outputs: none
        """
        axes = self.weight_axes()
        ch, chart_parts = self.error_chart()
        top = M(formula, 26, colour)
        if top.width > 5.5:
            top.scale_to_fit_width(5.5)
        top.move_to(np.array([4.0, 2.5, 0]))
        lt = ValueTracker(FINE[0])
        pick = lambda: path_w[index_of(lt.get_value())]
        bars = always_redraw(lambda: self.bars(pick(), colour))
        vals = always_redraw(lambda: self.values(pick(), colour))
        tr_curve = always_redraw(lambda: ch.curve(lambda x: np.interp(x, FINE, path_tr), FINE[0], lt.get_value()).set_stroke(GREY_B, 3))
        te_curve = always_redraw(lambda: ch.curve(lambda x: np.interp(x, FINE, path_te), FINE[0], lt.get_value()).set_stroke(GREEN, 4))

        def readout():
            """
            Description: Builds the live readout of λ and both errors, plus the count of exact zeros when asked.
            Inputs: none; it reads the λ tracker.
            Outputs: the readout group.
            """
            k = index_of(lt.get_value())
            row = VGroup(cached("math", f"λ = {lam_txt(10 ** FINE[k])}", 22, colour),
                         cached("math", f"training {path_tr[k]:.3f}", 22, GREY_B),
                         cached("math", f"test {path_te[k]:.3f}", 22, GREEN)).arrange(RIGHT, buff=0.3)
            row.move_to(np.array([1.9, 1.85, 0]), aligned_edge=LEFT)
            if zeros:
                z = int((path_w[k] == 0).sum())
                row.add(cached("math", f"weights exactly 0:  {z} of {len(NAMES)}", 22, colour if z else GREY_B)
                        .move_to(np.array([1.9, 1.38, 0]), aligned_edge=LEFT))
            return row

        read = always_redraw(readout)
        self.add(bars)
        self.play(FadeIn(axes), FadeIn(vals), FadeIn(chart_parts), FadeIn(top), run_time=1)
        self.add(tr_curve, te_curve, read)
        self.say(captions[0])
        self.say(captions[1])
        self.say(captions[2], wait=0)
        self.play(lt.animate.set_value(FINE[-1]), run_time=9, rate_func=linear)
        self.wait(3)
        self.say(captions[3])
        best = ch.c2p(FINE[k_best], path_te[k_best])
        self.play(lt.animate.set_value(FINE[k_best]), run_time=2.5)
        tag = M("lowest test error", 20, GREEN).next_to(best, UP, buff=0.3)
        mark = VGroup(Dot(best, radius=0.08, color=GREEN),
                      tag.add_background_rectangle(color=config.background_color, opacity=0.85, buff=0.05))
        self.say(captions[4], wait=0)
        self.play(FadeIn(mark, scale=1.3))
        self.wait(9)
        self.say(captions[5])

    def name_column(self):
        """
        Description: Builds the ten feature names, real ones in white and useless ones in grey.
        Inputs: none
        Outputs: the names, right-aligned at NAME_R.
        """
        g = VGroup()
        for j, nm in enumerate(NAMES):
            g.add(Text(nm, font=SERIF, font_size=22, color=WHITE if REAL[j] else GREY_B)
                  .move_to(np.array([NAME_R, ROW_Y[j], 0]), aligned_edge=RIGHT))
        return g

    def weight_axes(self):
        """
        Description: Builds the bar chart's frame: names, zero line, grid lines, axis label, and the true-effect outlines.
        Inputs: none
        Outputs: VGroup(names, frame, outlines with their legend).
        """
        top, bot = ROW_Y[0] + 0.3, ROW_Y[-1] - 0.3
        frame = VGroup(Line(np.array([X0, top, 0]), np.array([X0, bot, 0])).set_stroke(GREY_B, 2))
        for v in (-0.2, 0.2):
            x = X0 + v * SCALE
            frame.add(DashedLine(np.array([x, top, 0]), np.array([x, bot, 0]), dash_length=0.06).set_stroke(GREY_D, 1.5))
            frame.add(Text(num(v, ".1f"), font=MONO, font_size=18, color=GREY_B).move_to(np.array([x, bot - 0.2, 0])))
        frame.add(Text("0", font=MONO, font_size=18, color=GREY_B).move_to(np.array([X0, bot - 0.2, 0])))
        frame.add(Text("weight, per standard deviation of the feature", font=SERIF, font_size=20, color=GREY_B)
                  .move_to(np.array([X0 - 0.5, bot - 0.5, 0])))
        outlines = VGroup()
        for j in range(len(NAMES)):
            if REAL[j]:
                outlines.add(bar(self.d["w_true"][j], ROW_Y[j]).set_fill(opacity=0).set_stroke(WHITE, 2, opacity=0.85))
        legend = VGroup(Rectangle(width=0.4, height=BAR_H).set_stroke(WHITE, 2, opacity=0.85),
                        Text("true effect", font=SERIF, font_size=20, color=GREY_B)).arrange(RIGHT, buff=0.15)
        legend.move_to(np.array([X0 + 0.2, top + 0.32, 0]))
        return VGroup(self.name_column(), frame, VGroup(outlines, legend))

    def bars(self, w, colour):
        """
        Description: Builds one horizontal bar per weight.
        Inputs:
        - w: the ten weights.
        - colour: the bar colour.
        Outputs: the bars; a weight of exactly 0 gets an invisible bar.
        """
        return VGroup(*[bar(w[j], ROW_Y[j]).set_stroke(width=0).set_fill(colour, 0.85 if w[j] else 0) for j in range(len(w))])

    def values(self, w, colour):
        """
        Description: Builds the column of weight numbers beside the bars.
        Inputs:
        - w: the ten weights.
        - colour: the colour of nonzero weights; exact zeros are grey.
        Outputs: the numbers, right-aligned at VAL_R.
        """
        return VGroup(*[cached("mono", wtxt(w[j]), 20, colour if w[j] else GREY_B)
                        .move_to(np.array([VAL_R, ROW_Y[j], 0]), aligned_edge=RIGHT) for j in range(len(w))])

    def error_bars(self, tr, te):
        """
        Description: Builds the right-hand panel comparing training and test error as two bars.
        Inputs:
        - tr, te: the training and test root-mean-square errors.
        Outputs: VGroup(heading, training row, test row).
        """
        k = 18.0
        g = VGroup(Text("typical miss, % per minute", font=SERIF, font_size=22, color=GREY_B)
                   .move_to(np.array([2.0, 1.95, 0]), aligned_edge=LEFT))
        for i, (lab, v, col) in enumerate(((f"on the {N_TRAIN} training flights", tr, GREY_B), (f"on {N_TEST} new flights", te, GREEN))):
            y = 1.0 - 1.35 * i
            t = Text(lab, font=SERIF, font_size=22, color=col).move_to(np.array([2.0, y + 0.38, 0]), aligned_edge=LEFT)
            r = Rectangle(width=max(v * k, 0.03), height=0.32).set_stroke(width=0).set_fill(col, 0.85)
            r.move_to(np.array([2.0, y - 0.08, 0]), aligned_edge=LEFT)
            g.add(VGroup(t, r, Text(f"{v:.3f}", font=MONO, font_size=22, color=col).next_to(r, RIGHT, buff=0.15)))
        return g

    def error_chart(self):
        """
        Description: Builds the right-hand chart of error against λ on a log scale.
        Inputs: none
        Outputs: (chart, VGroup of its frame, axis label and legend).
        """
        ch = Chart((-4, 0), (0, 0.3), 4.6, 2.4, np.array([4.3, -0.55, 0]), xticks=(-4, -3, -2, -1, 0),
                   yticks=(0, 0.1, 0.2, 0.3), tick_size=16, xlabels={-4: "0.0001", -3: "0.001", -2: "0.01", -1: "0.1", 0: "1"})
        xl = M("λ, log scale", 22, GREY_B).next_to(ch.c2p(-2, 0), DOWN, buff=0.45)
        key = VGroup(M(f"test error, {N_TEST} new flights", 20, GREEN), M(f"training error, {N_TRAIN} flights", 20, GREY_B))
        key.arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to(ch.c2p(-3.75, 0.29), aligned_edge=UL)
        unit = M("% per minute", 18, GREY_B).next_to(ch.c2p(-4, 0.3), UP, buff=0.12)
        return ch, VGroup(ch.group, xl, key, unit)

    def heading(self, text):
        """
        Description: Swaps the section title at the top of the frame.
        Inputs:
        - text: the new title, as Pango markup.
        Outputs: none
        """
        new = M(text, 36)
        if new.width > 13:
            new.scale_to_fit_width(13)
        new.to_edge(UP, buff=0.35)
        anims = [FadeIn(new, shift=0.15 * DOWN)]
        if self.head is not None:
            anims.append(FadeOut(self.head))
        self.play(*anims, run_time=0.6)
        self.head = new

    def end_section(self):
        """
        Description: Fades out everything on screen, caption and heading included.
        Inputs: none
        Outputs: none
        Constraints: every section ends with this call, so sections rendered on their own join without a seam.
        """
        for m in self.mobjects:
            m.clear_updaters()
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
        self.cap, self.head = None, None

    def clear_stage(self):
        """
        Description: Fades out everything on screen except the caption and heading.
        Inputs: none
        Outputs: none
        """
        keep = {id(self.cap), id(self.head)}
        rest = [m for m in self.mobjects if id(m) not in keep]
        for m in rest:
            m.clear_updaters()
        if rest:
            self.play(*[FadeOut(m) for m in rest], run_time=0.7)

    def mark(self, name):
        """
        Description: Prints the time a section starts, for finding it in the rendered video.
        Inputs:
        - name: the section's name.
        Outputs: none
        """
        print(f"SECTION {name} t={self.renderer.time:.1f}")

    def say(self, text, wait=1.5):
        """
        Description: Swaps the caption at the bottom of the frame and holds it long enough to read.
        Inputs:
        - text: the caption, as Pango markup.
        - wait: the shortest hold in seconds; 0 means an animation follows at once.
        Outputs: none
        """
        new = MarkupText(text, font=SERIF, font_size=26)
        if new.width > 13.4:
            new.scale_to_fit_width(13.4)
        new.to_edge(DOWN, buff=0.4)
        anims = [FadeIn(new, shift=0.15 * UP)]
        if self.cap is not None:
            anims.append(FadeOut(self.cap, shift=0.15 * UP))
        self.play(*anims, run_time=0.6)
        self.cap = new
        # No narration, so hold long enough to read (~20 characters a second);
        # captions that lead into an animation get half that, the animation covers the rest.
        plain = len(text.replace("<sub>", "").replace("</sub>", "").replace("<sup>", "").replace("</sup>", ""))
        self.wait(max(wait, plain / 20) if wait else plain / 45)


class Chart:
    """
    Description: A plot frame drawn along its bottom and left edges, so its ranges need not contain 0.
    Fields:
    - xr, yr: the data ranges, as (low, high).
    - xl, yl: the frame's width and height, in screen units.
    - origin: the screen point of (xr[0], yr[0]).
    - group: the axes, ticks and tick labels to draw.
    """

    def __init__(self, xr, yr, xl, yl, centre, xticks=(), yticks=(), tick_size=20, xlabels=None):
        """
        Description: Builds the frame with its ticks and tick labels.
        Inputs:
        - xr, yr: the data ranges, as (low, high).
        - xl, yl: the frame's width and height, in screen units.
        - centre: the frame's centre on screen.
        - xticks, yticks: the values to tick on each axis.
        - tick_size: the tick labels' font size.
        - xlabels: optional text for each x tick, keyed by its value.
        Outputs: none
        """
        self.xr, self.yr, self.xl, self.yl = xr, yr, xl, yl
        self.origin = centre - np.array([xl / 2, yl / 2, 0])
        g = VGroup(Line(self.c2p(xr[0], yr[0]), self.c2p(xr[1], yr[0])).set_stroke(GREY_B, 2),
                   Line(self.c2p(xr[0], yr[0]), self.c2p(xr[0], yr[1])).set_stroke(GREY_B, 2))
        for v in xticks:
            p = self.c2p(v, yr[0])
            g.add(Line(p + 0.07 * DOWN, p + 0.07 * UP).set_stroke(GREY_B, 2))
            text = xlabels[v] if xlabels else num(v, "g")
            g.add(Text(text, font=MONO, font_size=tick_size, color=GREY_B).next_to(p, DOWN, buff=0.14))
        for v in yticks:
            p = self.c2p(xr[0], v)
            g.add(Line(p + 0.07 * LEFT, p + 0.07 * RIGHT).set_stroke(GREY_B, 2))
            g.add(Text(num(v, "g"), font=MONO, font_size=tick_size, color=GREY_B).next_to(p, LEFT, buff=0.14))
        self.group = g

    def curve(self, f, a, b, n=240):
        """
        Description: Draws f on [a, b] as a polyline, dropping the points outside the y-range.
        Inputs:
        - f: the function to draw.
        - a, b: the x interval.
        - n: the number of sample points.
        Outputs: the polyline, or an empty one when [a, b] is too short to draw.
        Constraints: each curve drawn here leaves the y-range at most once at each end, so dropping points
        never joins two separate pieces.
        """
        if b - a < 1e-6:
            return VMobject()
        pts = []
        for v in np.linspace(a, b, n):
            y = f(v)
            if self.yr[0] <= y <= self.yr[1]:
                pts.append(self.c2p(v, y))
        return VMobject().set_points_as_corners(pts) if len(pts) > 1 else VMobject()

    def c2p(self, x, y):
        """
        Description: Maps a data point to the screen.
        Inputs:
        - x, y: the data coordinates.
        Outputs: the screen point.
        """
        return self.origin + np.array([(x - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.xl,
                                       (y - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.yl, 0])


def precompute():
    """
    Description: Computes every number the video shows and checks the claims its captions make.
    Inputs: none
    Outputs: a dict of arrays and numbers, keyed by name.
    """
    d = {}
    rng = np.random.default_rng(SEED)
    Xtr, ytr = draw_flights(rng, N_TRAIN)
    Xte, yte = draw_flights(rng, N_TEST)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Z, Zt = (Xtr - mu) / sd, (Xte - mu) / sd
    b = ytr.mean()
    yc = ytr - b
    n = N_TRAIN
    d["w_true"] = COEF * sd
    d["flight1"] = flight_strings(Xtr[0])
    d["drain1"] = ytr[0]

    w_ols = np.linalg.lstsq(Z, yc, rcond=None)[0]
    d["w_ols"] = w_ols
    d["ols_tr"], d["ols_te"] = rmse(b + Z @ w_ols, ytr), rmse(b + Zt @ w_ols, yte)
    d["ols_ratio"] = d["ols_te"] / d["ols_tr"]
    assert d["ols_tr"] < 0.3 * NOISE and d["ols_te"] > 2 * NOISE, "least squares must visibly overfit"
    useless = np.flatnonzero(~REAL)
    top2 = useless[np.argsort(-np.abs(w_ols[useless]))[:2]]
    d["ols_top_useless"] = [(NAMES[j], w_ols[j]) for j in top2]
    assert np.abs(w_ols[top2]).min() > 0.04, "the named useless weights must be visibly nonzero"

    lams = 10 ** FINE
    d["ridge_w"] = np.array([ridge(Z, yc, lam) for lam in lams])
    d["ridge_tr"] = np.array([rmse(b + Z @ w, ytr) for w in d["ridge_w"]])
    d["ridge_te"] = np.array([rmse(b + Zt @ w, yte) for w in d["ridge_w"]])
    size = (d["ridge_w"] ** 2).sum(1)
    assert np.all(np.diff(size) < 0), "ridge's total squared weight must fall as λ grows"
    assert np.all(np.diff(d["ridge_tr"]) > 0), "ridge's training error must only rise with λ"
    kr = int(np.argmin(d["ridge_te"]))
    d["k_ridge"] = kr
    assert 0 < kr < len(FINE) - 1 and d["ridge_te"][kr] < d["ols_te"], "ridge's test error must dip below least squares"
    assert d["ridge_te"][0] > d["ridge_te"][kr] < d["ridge_te"][-1], "ridge's test error must be U-shaped"
    wr = d["ridge_w"][kr]
    assert all(wtxt(v) not in ("+0.000", "−0.000") for v in wr), "every ridge weight must print as nonzero at the best λ"
    d["ridge_useless_min"], d["ridge_useless_max"] = np.abs(wr[~REAL]).min(), np.abs(wr[~REAL]).max()

    d["lasso_w"] = lasso_path(Z, yc, lams)
    d["lasso_tr"] = np.array([rmse(b + Z @ w, ytr) for w in d["lasso_w"]])
    d["lasso_te"] = np.array([rmse(b + Zt @ w, yte) for w in d["lasso_w"]])
    for lam, w in zip(lams, d["lasso_w"]):
        c = Z.T @ (yc - Z @ w) / n
        nz = w != 0
        assert np.all(np.abs(c[~nz]) <= lam + 1e-8), "lasso optimality: a zero weight's pull must not exceed λ"
        assert np.allclose(c[nz], lam * np.sign(w[nz]), atol=1e-8), "lasso optimality: a nonzero weight's pull must equal λ"
    kl = int(np.argmin(d["lasso_te"]))
    d["k_lasso"] = kl
    assert np.array_equal(d["lasso_w"][kl] != 0, REAL), "at its best λ, lasso must keep exactly the four real features"
    assert np.all(d["lasso_w"][kl][~REAL] == 0.0), "the useless lasso weights must be exactly zero"
    assert d["lasso_te"][kl] < d["ridge_te"][kr], "lasso's best test error must beat ridge's on this data"
    d["lam_max"] = np.abs(Z.T @ yc / n).max()
    assert np.all(d["lasso_w"][lams >= d["lam_max"]] == 0) and np.any(d["lasso_w"][lams < d["lam_max"]] != 0)
    first, back = [], []
    for j in range(len(NAMES)):
        z = d["lasso_w"][:, j] == 0
        k0 = int(np.argmax(z))
        first.append(k0)
        if not z[k0:].all():
            off = k0 + np.flatnonzero(~z[k0:])
            back.append((j, lams[off[0]], lams[off[-1] + 1], np.abs(d["lasso_w"][off, j]).max()))
    first = np.array(first)
    assert len(set(first[~REAL])) == int((~REAL).sum()), "the useless weights must reach 0 one at a time"
    assert first[~REAL].max() < first[REAL].min(), "all useless weights must reach 0 before any real one"
    assert len(back) == 1 and not REAL[back[0][0]], "exactly one useless weight may leave 0 again, as the caption says"
    assert back[0][1] > lams[kl] and back[0][3] < 0.01, "it must leave 0 only after the best λ, and stay too small to see"
    d["lasso_back"] = (NAMES[back[0][0]], back[0][1], back[0][2], np.ceil(back[0][3] * 1000) / 1000)
    assert np.all(np.diff(d["lasso_tr"]) >= -1e-12), "lasso's training error must not fall as λ grows"

    a = A_1D
    d["l2_at_a"], d["l1_at_a"] = shrink_l2(a, LAM_1D), shrink_l1(a, LAM_1D)
    d["l2_at_09"], d["l1_at_09"] = shrink_l2(0.9, LAM_1D), shrink_l1(0.9, LAM_1D)
    assert abs(d["l2_at_a"] - 0.2) < 1e-12 and d["l1_at_a"] == 0.0
    assert abs(d["l2_at_09"] - 0.6) < 1e-12 and abs(d["l1_at_09"] - 0.4) < 1e-12
    grid = np.linspace(-0.5, 1.0, 300001)
    for rule, pen in ((d["l2_at_a"], LAM_1D / 2 * grid ** 2), (d["l1_at_a"], LAM_1D * np.abs(grid))):
        assert abs(grid[np.argmin(0.5 * (grid - a) ** 2 + pen)] - rule) < 1e-4, "a shrink rule must give the sum's lowest point"

    d["geo_l2"], d["geo_l1"] = touch_l2(), touch_l1()
    d["geo_c2"], d["geo_c1"] = geo_loss(d["geo_l2"]), geo_loss(d["geo_l1"])
    assert np.all(d["geo_l2"] > 0.2), "the disk's touching point must have both weights clearly nonzero"
    assert d["geo_l1"][1] == 0.0 and d["geo_l1"][0] == GEO_T, "the diamond must be touched at its corner (t, 0)"
    probe = np.random.default_rng(0).uniform(-1, 1, (200000, 2))
    disk, diamond = probe[(probe ** 2).sum(1) <= GEO_T ** 2], probe[np.abs(probe).sum(1) <= GEO_T]
    assert geo_loss_many(disk).min() >= d["geo_c2"] - 1e-9 and geo_loss_many(diamond).min() >= d["geo_c1"] - 1e-9

    d["toy_paths"] = [decay_run(np.array([0.0]), lambda w: w - TOY_TARGET, eta, lam, TOY_STEPS)[:, 0] for eta, lam, _ in TOY_RUNS]
    for (eta, lam, _), path in zip(TOY_RUNS, d["toy_paths"]):
        assert abs(path[-1] - TOY_TARGET / (1 + lam)) < 1e-3, "each run must end at 2 / (1 + λ)"
    d["toy_factor"] = 1 - 0.1 * 0.25
    lam = lams[kr]
    hess_max = np.linalg.eigvalsh(Z.T @ Z / n).max()
    for eta in DRONE_ETAS:
        assert eta < 2 / (hess_max + lam), "gradient descent must be stable at this learning rate"
        w_gd = decay_run(np.zeros(len(NAMES)), lambda w: -Z.T @ (yc - Z @ w) / n, eta, lam, 40000)[-1]
        assert np.abs(w_gd - d["ridge_w"][kr]).max() < 1e-9, "decay on the drone data must land on the ridge weights"

    d["adam"] = adam_means()
    ref = (1 - ADAM_ETA * ADAM_LAM) ** np.arange(ADAM_STEPS + 1)
    for j in (0, 1):
        assert np.abs(d["adam"]["adamw"][:, j] - ref).max() < 0.05, "AdamW's average must follow (1 − ηλ)^t"
    d["adamw_end"] = ref[-1]
    d["adam_l2_a"] = d["adam"]["adam_l2"][-1, 0]
    assert d["adam_l2_a"] > 0.7 and abs(d["adam"]["adam_l2"][-1, 1]) < 0.05, "Adam + L2 must spare A and collapse B"
    print("PRECOMPUTE", f"ols {d['ols_tr']:.4f}/{d['ols_te']:.4f}", f"ridge λ {lams[kr]:.4g} te {d['ridge_te'][kr]:.4f}",
          f"lasso λ {lams[kl]:.4g} te {d['lasso_te'][kl]:.4f}", f"λmax {d['lam_max']:.4f}",
          f"adamw {d['adamw_end']:.3f} adam_l2 A {d['adam_l2_a']:.3f}")
    return d


def draw_flights(rng, n):
    """
    Description: Draws n synthetic flights: ten raw features and the battery drain.
    Inputs:
    - rng: the seeded generator.
    - n: the number of flights.
    Outputs: (X, y): the raw features, shape (n, 10), and the drain in % per minute.
    Constraints: wind rises with the hour of day, so one useless feature is correlated with a real one.
    """
    payload = rng.uniform(0.2, 2.5, n)
    hour = rng.uniform(6, 20, n)
    wind = np.clip(2 + 0.45 * (hour - 6) + rng.normal(0, 2.2, n), 0, None)
    climb = rng.uniform(0.5, 4.0, n)
    temp = rng.normal(15, 7, n)
    pad = rng.integers(1, 13, n).astype(float)
    dow = rng.integers(0, 7, n).astype(float)
    gps = rng.integers(7, 20, n).astype(float)
    fw = rng.integers(100, 140, n).astype(float)
    op = rng.integers(1, 10, n).astype(float)
    X = np.column_stack([payload, wind, climb, temp, hour, pad, dow, gps, fw, op])
    y = 0.9 + X @ COEF + rng.normal(0, NOISE, n)
    return X, y


def flight_strings(x):
    """
    Description: Formats one flight's raw features for display.
    Inputs:
    - x: the ten raw feature values.
    Outputs: ten strings with units.
    """
    return [f"{x[0]:.2f} kg", f"{x[1]:.1f} m/s", f"{x[2]:.1f} m/s", f"{num(x[3], '.0f')} °C", clock(x[4]),
            f"pad {int(x[5])}", DAYS[int(x[6])], f"{int(x[7])} in view", f"build {int(x[8])}", f"#{int(x[9])}"]


def ridge(Z, yc, lam):
    """
    Description: Solves L2-regularized least squares in closed form.
    Inputs:
    - Z: the standardized features, shape (n, p).
    - yc: the centred targets.
    - lam: λ, for the objective ½ · mean squared error + (λ/2) ∑ w².
    Outputs: the weights.
    """
    n, p = Z.shape
    return np.linalg.solve(Z.T @ Z / n + lam * np.eye(p), Z.T @ yc / n)


def lasso_path(Z, yc, lams):
    """
    Description: Solves L1-regularized least squares at every λ by coordinate descent, warm-starting from the larger λ.
    Inputs:
    - Z: the standardized features, each with mean 0 and mean square 1.
    - yc: the centred targets.
    - lams: the λ values, increasing, for the objective ½ · mean squared error + λ ∑ |w|.
    Outputs: the weights at each λ, shape (len(lams), p).
    Constraints: each column's mean square must be 1, which makes each one-weight update a plain soft threshold.
    """
    n, p = Z.shape
    out = np.zeros((len(lams), p))
    w = np.zeros(p)
    for k in range(len(lams) - 1, -1, -1):
        r = yc - Z @ w
        for _ in range(200000):
            delta = 0.0
            for j in range(p):
                new = shrink_l1(Z[:, j] @ (r + Z[:, j] * w[j]) / n, lams[k])
                if new != w[j]:
                    r -= Z[:, j] * (new - w[j])
                    delta = max(delta, abs(new - w[j]))
                    w[j] = new
            if delta < 1e-14:
                break
        out[k] = w
    return out


def rmse(pred, y):
    """
    Description: Computes the root-mean-square error, the typical size of a miss.
    Inputs:
    - pred, y: the predictions and the measured values.
    Outputs: the error.
    """
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def decay_run(w0, grad, eta, lam, steps):
    """
    Description: Runs gradient descent with weight decay, w ← w − η ∇L(w) − ηλw.
    Inputs:
    - w0: the starting weights.
    - grad: the loss gradient, as a function of the weights.
    - eta, lam: the learning rate η and the decay strength λ.
    - steps: the number of steps.
    Outputs: the weights after each step, starting with w0, shape (steps + 1, len(w0)).
    """
    out = [w0.copy()]
    w = w0.copy()
    for _ in range(steps):
        w = w - eta * grad(w) - eta * lam * w
        out.append(w.copy())
    return np.array(out)


def adam_means():
    """
    Description: Runs Adam with an L2 penalty and AdamW on two weights whose loss gradient is pure noise.
    Inputs: none
    Outputs: {"adam_l2": means, "adamw": means}, each the average weight over ADAM_RUNS runs, shape (ADAM_STEPS + 1, 2).
    Constraints: both methods see the same noise, so the gap between them comes from the update rule alone.
    """
    b1, b2, eps = 0.9, 0.999, 1e-8
    out = {}
    for mode in ("adam_l2", "adamw"):
        noise = np.random.default_rng(11)
        w = np.ones((ADAM_RUNS, 2))
        m, v = np.zeros_like(w), np.zeros_like(w)
        means = [w.mean(0)]
        for t in range(1, ADAM_STEPS + 1):
            g = noise.normal(0, 1, w.shape) * ADAM_SIGMA
            if mode == "adam_l2":
                g = g + ADAM_LAM * w
            else:
                w = w - ADAM_ETA * ADAM_LAM * w
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            w = w - ADAM_ETA * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
            means.append(w.mean(0))
        out[mode] = np.array(means)
    return out


def touch_l2():
    """
    Description: Finds the best weights in the two-weight picture under the L2 budget w₁² + w₂² ≤ t².
    Inputs: none
    Outputs: the touching point on the circle.
    """
    lo, hi = 0.0, 1e6
    for _ in range(200):
        mu = (lo + hi) / 2
        w = np.linalg.solve(GEO_A + mu * np.eye(2), GEO_A @ GEO_WS)
        lo, hi = (mu, hi) if np.linalg.norm(w) > GEO_T else (lo, mu)
    return np.linalg.solve(GEO_A + hi * np.eye(2), GEO_A @ GEO_WS)


def touch_l1():
    """
    Description: Finds the best weights in the two-weight picture under the L1 budget |w₁| + |w₂| ≤ t.
    Inputs: none
    Outputs: the touching point on the diamond, checked edge by edge.
    Constraints: the least-squares point lies outside the diamond, so the answer is on its boundary.
    """
    corners = [np.array(c, dtype=float) for c in ((GEO_T, 0), (0, GEO_T), (-GEO_T, 0), (0, -GEO_T))]
    best = None
    for k in range(4):
        p, q = corners[k], corners[(k + 1) % 4]
        e = q - p
        s = float(np.clip(-(e @ GEO_A @ (p - GEO_WS)) / (e @ GEO_A @ e), 0, 1))
        cand = p if s == 0 else (q if s == 1 else p + s * e)
        if best is None or geo_loss(cand) < geo_loss(best):
            best = cand
    return best


def geo_loss(w):
    """
    Description: The two-weight picture's training loss, ½ (w − w*)ᵀ A (w − w*).
    Inputs:
    - w: the two weights.
    Outputs: the loss.
    """
    e = w - GEO_WS
    return float(0.5 * e @ GEO_A @ e)


def geo_loss_many(W):
    """
    Description: The two-weight loss for many points at once.
    Inputs:
    - W: the points, shape (m, 2).
    Outputs: the losses, shape (m,).
    """
    E = W - GEO_WS
    return 0.5 * np.einsum("ij,jk,ik->i", E, GEO_A, E)


def ellipse_arcs(ch, c):
    """
    Description: Draws the contour where the two-weight loss equals c, keeping only the parts inside the chart.
    Inputs:
    - ch: the chart.
    - c: the loss level.
    Outputs: a VGroup of polylines, one per visible arc.
    """
    vals, vecs = np.linalg.eigh(GEO_A)
    th = np.linspace(0, 2 * np.pi, 400)
    pts = GEO_WS + (vecs @ (np.sqrt(2 * c / vals)[:, None] * np.array([np.cos(th), np.sin(th)]))).T
    inside = (pts[:, 0] >= ch.xr[0]) & (pts[:, 0] <= ch.xr[1]) & (pts[:, 1] >= ch.yr[0]) & (pts[:, 1] <= ch.yr[1])
    if inside.all():
        return VGroup(VMobject().set_points_as_corners([ch.c2p(*p) for p in pts]))
    start = int(np.argmin(inside))
    order = np.roll(np.arange(len(pts)), -start)
    arcs, run = VGroup(), []
    for i in order:
        if inside[i]:
            run.append(ch.c2p(*pts[i]))
        elif run:
            if len(run) > 1:
                arcs.add(VMobject().set_points_as_corners(run))
            run = []
    if len(run) > 1:
        arcs.add(VMobject().set_points_as_corners(run))
    return arcs


def shrink_l2(a, lam):
    """
    Description: L2's one-weight rule: the lowest point of ½(w − a)² + (λ/2)w².
    Inputs:
    - a: the weight the data alone would choose.
    - lam: λ.
    Outputs: a / (1 + λ).
    """
    return a / (1 + lam)


def shrink_l1(a, lam):
    """
    Description: L1's one-weight rule, the soft threshold: the lowest point of ½(w − a)² + λ|w|.
    Inputs:
    - a: the weight the data alone would choose.
    - lam: λ.
    Outputs: sign(a) · max(|a| − λ, 0), which is exactly 0.0 when |a| ≤ λ, never −0.0.
    """
    return float(np.sign(a) * (abs(a) - lam)) if abs(a) > lam else 0.0


def index_of(lv):
    """
    Description: Finds the grid point in FINE nearest a log10 λ value.
    Inputs:
    - lv: the log10 λ value.
    Outputs: the index into FINE.
    """
    step = FINE[1] - FINE[0]
    return int(np.clip(round((lv - FINE[0]) / step), 0, len(FINE) - 1))


def bar(w, y):
    """
    Description: Builds a horizontal bar from the zero line to a weight's value.
    Inputs:
    - w: the weight.
    - y: the row's screen y.
    Outputs: the bar as a polygon; a zero weight gives a sliver of width 0.0001.
    """
    x1 = X0 + w * SCALE
    lo, hi = min(X0, x1), max(X0, x1)
    hi = max(hi, lo + 1e-4)
    return Polygon(np.array([lo, y - BAR_H / 2, 0]), np.array([hi, y - BAR_H / 2, 0]),
                   np.array([hi, y + BAR_H / 2, 0]), np.array([lo, y + BAR_H / 2, 0]))


def cached(kind, text, size, colour):
    """
    Description: Returns a copy of a text mobject, building each distinct one only once.
    Inputs:
    - kind: "math" for a Cambria formula, or "mono" for a Consolas number.
    - text: the text, as Pango markup for "math".
    - size: the font size.
    - colour: the colour.
    Outputs: a fresh copy, free to move.
    Constraints: the live readouts rebuild every frame, and building text is slow, so unchanged strings must come from here.
    """
    key = (kind, text, size, str(colour))
    if key not in TEXT_CACHE:
        TEXT_CACHE[key] = M(text, size, colour) if kind == "math" else Text(text, font=MONO, font_size=size, color=colour)
    return TEXT_CACHE[key].copy()


def M(text, size=30, color=WHITE):
    """
    Description: Builds a formula from Pango markup in Cambria, so <sub> and <sup> give real subscripts.
    Inputs:
    - text: the markup.
    - size: the font size.
    - color: the default colour.
    Outputs: the text mobject.
    """
    return MarkupText(text, font=MATH, font_size=size, color=color)


def num(v, spec):
    """
    Description: Formats a number for display with a true minus sign.
    Inputs:
    - v: the number.
    - spec: a format spec, such as ".2f".
    Outputs: the formatted string.
    """
    return format(v, spec).replace("-", "−")


def wtxt(w):
    """
    Description: Formats a weight for the bar chart's number column.
    Inputs:
    - w: the weight.
    Outputs: "0" for a weight that is exactly zero, otherwise a signed value with three decimals.
    """
    return "0" if w == 0 else num(w, "+.3f")


def lam_txt(v):
    """
    Description: Formats a λ value with three significant digits.
    Inputs:
    - v: the λ value.
    Outputs: the string, such as "0.0708".
    """
    return f"{v:.3g}"


def clock(h):
    """
    Description: Formats an hour of day as a 24-hour clock time.
    Inputs:
    - h: the hour, as a fraction of a day's hours, such as 14.4.
    Outputs: the time, such as "14:24".
    """
    total = int(round(h * 60))
    return f"{total // 60:02d}:{total % 60:02d}"
