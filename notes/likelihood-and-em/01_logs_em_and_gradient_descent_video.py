"""Renders a Manim video on how the log, EM's E-step and M-step, and gradient descent each fit a Gaussian mixture.

Contents:
- LogsEMGradient: the scene, with one method per section of the video
- Chart: a plot frame whose ranges need not contain 0
- precompute: every number on screen, with an assert for each claim a caption makes
- em_step, ascend, mean_ll_grad, mean_ll, unpack: fitting the mixture by EM and by gradient ascent
- make_bound, first_within, loglik, mix_pdf, comp: EM's lower bound, the convergence test and the mixture density
- component_curves, density_curve, number_line, blend, box, arrow, M: drawing helpers
- sx, sy, num, fmt: screen coordinates and number formatting
- fonts, colours, the seeded dataset and the layout constants
"""
import math
import os

import numpy as np
from manim import *

SERIF = "Georgia"
"""Description: Serif font for captions and titles."""
MONO = "Consolas"
"""Description: Monospace font for code and tick labels."""
MATH = "Cambria"
"""Description: Font for formulas, because Cambria has the Greek letters, ∑, ∏, ℓ and arrows they need."""
BLUE_K = "#58C4DD"
"""Description: Colour of the left bell curve and of the points it owns."""
AMBER = "#FFD166"
"""Description: Colour of the right bell curve and of the points it owns."""
GOOD = "#83C167"
"""Description: Colour of EM's curve and of its lower bound."""
BAD = "#FC6255"
"""Description: Colour of warnings and of the unstable step size."""
PURPLE = "#B48EDE"
"""Description: Colour of the η = 0.3 curve in the race."""
GREY_DOT = "#7d8894"
"""Description: Colour of a data point whose bell curve is unknown."""
config.background_color = "#0b0d10"

RNG = np.random.default_rng(20261122)
"""Description: The seeded generator that draws the dataset."""
X = np.round(np.concatenate([RNG.normal(-1.6, 0.75, 72), RNG.normal(1.9, 1.0, 48)]), 3)
"""Description: The 120 data points: 72 drawn from N(-1.6, 0.75^2), then 48 from N(1.9, 1.0^2)."""
LABEL = np.array([0] * 72 + [1] * 48)
"""Description: Which bell curve drew each point, 0 for the left one and 1 for the right one."""
N = len(X)
"""Description: The number of data points."""
INIT = (np.array([0.5, 0.5]), np.array([-0.5, 0.5]), np.array([1.6, 1.6]))
"""Description: The starting guess for both EM and gradient descent, as (pi, mu, sigma) arrays over the two curves."""
ETAS = (3.0, 1.0, 0.3, 0.1)
"""Description: The gradient-descent step sizes in the race, applied to the mean log-likelihood."""
ETA_COLORS = {3.0: BAD, 1.0: BLUE_K, 0.3: PURPLE, 0.1: GREY_B}
"""Description: The colour of each step size's curve in the race."""
RACE_STEPS = 1000
"""Description: How many iterations each method runs in the race."""
BATCH = 30
"""Description: The minibatch size in the minibatch example."""
CLOSE = 0.01
"""Description: How close to EM's converged log-likelihood counts as reaching the top."""

XMIN, XMAX, BIN = -4.0, 5.0, 0.25
"""Description: The number line's data range and the dot histogram's bin width."""
Y0 = -1.9
"""Description: The screen height of the number line."""
DOT_H = 0.22
"""Description: The vertical spacing of stacked dots, in screen units."""
DOT_R = 0.095
"""Description: The radius of a data point's dot, in screen units."""
CURVE_SCALE = N * BIN * DOT_H
"""Description: Screen height per unit of density, which makes a density curve match the dot stacks."""


class LogsEMGradient(Scene):
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
        Constraints: when the EMV_ONLY environment variable names sections, separated by commas, only those play.
        """
        self.cap = None
        self.head = None
        self.d = precompute()
        only = [s for s in os.environ.get("EMV_ONLY", "").split(",") if s]
        for name in ("intro", "likelihood", "log_facts", "hidden_labels", "em_steps",
                     "why_it_climbs", "modern", "race", "then_and_now", "summary"):
            if not only or name in only:
                getattr(self, name)()

    def intro(self):
        """
        Description: Drops the data in as dots and shows the title.
        Inputs: none
        Outputs: none
        """
        self.mark("intro")
        dots, line = self.histogram()
        self.play(Create(line), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(dt, shift=0.3 * DOWN) for dt in dots], lag_ratio=0.012), run_time=2)
        title = Text("Logs, EM and gradient descent", font=SERIF, font_size=50)
        sub = Text("how a model is fit to data, in 1977 and today", font=SERIF, font_size=30, color=GREY_B)
        sub.next_to(title, DOWN, buff=0.3)
        words = VGroup(title, sub).move_to(1.9 * UP)
        self.play(Write(title), FadeIn(sub, shift=0.2 * UP), run_time=1.6)
        self.wait(2.5)
        self.play(FadeOut(VGroup(words, dots, line)))

    def likelihood(self):
        """
        Description: Shows the likelihood as a product of 120 curve heights, and why that product underflows.
        Inputs: none
        Outputs: none
        """
        self.mark("likelihood")
        d = self.d
        best = d["best_params"]
        self.heading("Likelihood: how well a model explains the data")
        dots, line = self.histogram()
        mix = density_curve(lambda v: mix_pdf(best, v)).set_stroke(WHITE, 4)
        self.play(FadeIn(line), FadeIn(dots))
        self.say(f"Here are {N} points. A model of them is a density curve, which gives every point\n"
                 "a height: its probability density under the model.", wait=0)
        self.play(Create(mix), run_time=1.5)
        self.wait(3)
        sticks, labels = VGroup(), VGroup()
        for i in d["sample_idx"]:
            h = float(mix_pdf(best, X[i]))
            top = np.array([sx(X[i]), sy(h), 0])
            sticks.add(Line(np.array([sx(X[i]), Y0, 0]), top).set_stroke(AMBER, 5))
            lab = Text(f"{h:.2g}", font=MONO, font_size=22, color=AMBER).next_to(top, UP, buff=0.12)
            labels.add(lab.add_background_rectangle(color=config.background_color, opacity=0.85, buff=0.05))
        self.play(LaggedStart(*[Create(s) for s in sticks], lag_ratio=0.2),
                  LaggedStart(*[FadeIn(lb) for lb in labels], lag_ratio=0.2), run_time=1.6)
        hs = [f"{float(mix_pdf(best, X[i])):.2g}" for i in d["sample_idx"][:2]]
        prod1 = M("likelihood = p(x<sub>1</sub>) · p(x<sub>2</sub>) · p(x<sub>3</sub>) · … · "
                  f"p(x<sub>{N}</sub>)", 32)
        prod2 = M(f"= {hs[0]} × {hs[1]} × … = {d['prod_text']}", 32, AMBER)
        prod = VGroup(prod1, prod2).arrange(DOWN, buff=0.3).move_to(2.15 * UP)
        assert max(mix_pdf(best, X)) < 1
        self.say(f"The likelihood multiplies all {N} heights. Every height here is below 1, so the product\n"
                 f"is tiny: {d['prod_text']}.", wait=0)
        self.play(FadeIn(prod1, shift=0.1 * DOWN))
        self.play(FadeIn(prod2, shift=0.1 * DOWN))
        self.wait(4)
        zero = M("float32 stores this as 0", 30, BAD).move_to(np.array([4.2, 0.95, 0]))
        self.say("In float32, and in the 16-bit formats large models train in, it rounds to exactly 0.\n"
                 f"1,000 such points would give about 10<sup>{num(d['exp_1000'], 'd')}</sup>, "
                 "below even float64's smallest, 5 × 10<sup>−324</sup>.", wait=0)
        self.play(FadeIn(zero, scale=1.2))
        self.wait(7)
        self.end_section()

    def log_facts(self):
        """
        Description: Shows three things the log does: a product becomes a sum, the peak stays put, and exp cancels.
        Inputs: none
        Outputs: none
        """
        self.mark("log_facts")
        d = self.d
        self.heading("What the log does for training")
        t1 = M("1.  A product becomes a sum", 34, AMBER)
        e1 = M("log(a · b · c) = log a + log b + log c", 32)
        e2 = M("log ∏<sub>i</sub> p(x<sub>i</sub>) = ∑<sub>i</sub> log p(x<sub>i</sub>)", 32)
        e3 = M(f"log({d['prod_text']}) = {num(d['ll_best'], '.1f')}", 32, BLUE_K)
        g = VGroup(t1, e1, e2, e3).arrange(DOWN, buff=0.38).move_to(1.25 * UP)
        self.say(f"First, the log turns the product into a sum. The log of {d['prod_text']} is\n"
                 f"{num(d['ll_best'], '.1f')}, an ordinary number that no computer rounds to zero.", wait=0)
        self.play(FadeIn(t1))
        for e in (e1, e2, e3):
            self.play(FadeIn(e, shift=0.1 * UP), run_time=0.8)
        self.wait(4)
        b1 = M(f"average log p over all {N} points:   {num(d['mean_all'], '.3f')}", 28)
        b2 = M(f"average log p over a random batch of {BATCH}:   {num(d['mean_batch'], '.3f')}", 28)
        bb = VGroup(b1, b2).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(g, DOWN, buff=0.5)
        self.say("A sum over examples can be estimated from a sample. The average over a random minibatch\n"
                 "estimates the average over all the data, which is what lets SGD train on batches.", wait=0)
        self.play(FadeIn(b1, shift=0.1 * UP))
        self.play(FadeIn(b2, shift=0.1 * UP))
        self.wait(7)
        self.clear_stage()

        t2 = M("2.  The peak stays where it was", 34, AMBER).move_to(2.45 * UP)
        lmax = 0.7 ** 7 * 0.3 ** 3
        left = Chart((0, 1), (0, 1.12), 5.0, 2.6, np.array([-3.4, -0.3, 0]), xticks=(0, 0.5, 1))
        right = Chart((0, 1), (-20, 0), 5.0, 2.6, np.array([3.4, -0.3, 0]), xticks=(0, 0.5, 1))
        lc = left.curve(lambda t: t ** 7 * (1 - t) ** 3 / lmax, 0, 1).set_stroke(BLUE_K, 4)
        rc = right.curve(lambda t: 7 * math.log(t) + 3 * math.log(1 - t), 0.01, 0.995).set_stroke(AMBER, 4)
        ll_lab = M("θ<sup>7</sup> (1 − θ)<sup>3</sup>", 28, BLUE_K).next_to(left.group, UP, buff=0.25)
        rl_lab = M("7 log θ + 3 log(1 − θ)", 28, AMBER).next_to(right.group, UP, buff=0.25)
        marks = VGroup()
        for ch, top in ((left, 1.12), (right, 0)):
            marks.add(DashedLine(ch.c2p(0.7, ch.yr[0]), ch.c2p(0.7, top), dash_length=0.08).set_stroke(GREY_B, 2))
            marks.add(M("θ = 0.7", 24, GREY_B).next_to(ch.c2p(0.7, ch.yr[0]), DOWN, buff=0.45))
        self.say("Second, the log never changes which value is bigger, so the peak doesn't move.\n"
                 "Seven heads in ten coin flips: the likelihood and its log both peak at θ = 0.7.", wait=0)
        self.play(FadeIn(t2), FadeIn(left.group), FadeIn(right.group), run_time=1)
        self.play(Create(lc), FadeIn(ll_lab), Create(rc), FadeIn(rl_lab), run_time=1.8)
        self.play(Create(marks), run_time=1)
        self.wait(7)
        self.clear_stage()

        t3 = M("3.  The log undoes exp", 34, AMBER)
        g1 = M("N(x | μ, σ<sup>2</sup>) = exp(−(x − μ)<sup>2</sup> / 2σ<sup>2</sup>) / (σ√(2π))", 30)
        g2 = M("−log N(x | μ, σ<sup>2</sup>) = (x − μ)<sup>2</sup> / 2σ<sup>2</sup> + log σ + log √(2π)", 30)
        g3 = M("best μ for x<sub>1</sub>, …, x<sub>n</sub>:  minimize ∑<sub>i</sub> (x<sub>i</sub> − μ)<sup>2</sup>, "
               "so  μ = (1/n) ∑<sub>i</sub> x<sub>i</sub>", 30, BLUE_K)
        g4 = M("−log softmax(z)<sub>y</sub> = −z<sub>y</sub> + log ∑<sub>j</sub> exp(z<sub>j</sub>)"
               "      (cross-entropy)", 30)
        gg = VGroup(t3, g1, g2, g3, g4).arrange(DOWN, buff=0.42).move_to(0.6 * UP)
        self.say("Third, the log undoes exp. The negative log of a Gaussian is a squared error plus terms\n"
                 "without μ, so the best mean for one bell curve is just the average of its points.", wait=0)
        self.play(FadeIn(t3))
        for e in (g1, g2, g3):
            self.play(FadeIn(e, shift=0.1 * UP), run_time=0.9)
        self.wait(6)
        self.say("The same cancellation turns softmax's negative log-likelihood into the cross-entropy loss:\n"
                 "the exp disappears and the correct class's logit z<sub>y</sub> is left.", wait=0)
        self.play(FadeIn(g4, shift=0.1 * UP))
        self.wait(6)
        self.end_section()

    def hidden_labels(self):
        """
        Description: Fits each bell curve from known labels, then hides the labels to show the log of a sum.
        Inputs: none
        Outputs: none
        """
        self.mark("hidden_labels")
        d = self.d
        known = d["known_params"]
        self.heading("A Gaussian mixture: two bell curves, labels hidden")
        dots, line = self.histogram([BLUE_K if lb == 0 else AMBER for lb in LABEL])
        c1, c2 = component_curves(known)
        t1 = M(f"μ<sub>1</sub> = mean of the blue points = {num(known[1][0], '.2f')}", 28, BLUE_K)
        t2 = M(f"μ<sub>2</sub> = mean of the amber points = {num(known[1][1], '.2f')}", 28, AMBER)
        tt = VGroup(t1, t2).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to(np.array([-2.6, 2.3, 0]))
        self.play(FadeIn(line), FadeIn(dots))
        self.say("Suppose we knew which of two bell curves made each point. Fact 3 finishes the job:\n"
                 "each curve's mean is the average of its own points, and its width is their spread.", wait=0)
        self.play(Create(c1), Create(c2), FadeIn(tt, lag_ratio=0.3), run_time=2)
        self.wait(6)
        f1 = M("p(x) = π<sub>1</sub> N(x | μ<sub>1</sub>, σ<sub>1</sub><sup>2</sup>) + "
               "π<sub>2</sub> N(x | μ<sub>2</sub>, σ<sub>2</sub><sup>2</sup>)", 32)
        f2 = M(f"ℓ = ∑<sub>i</sub> <span foreground='{BAD}'>log(</span> π<sub>1</sub> N<sub>1</sub>(x<sub>i</sub>) + "
               f"π<sub>2</sub> N<sub>2</sub>(x<sub>i</sub>) <span foreground='{BAD}'>)</span>", 32)
        ff = VGroup(f1, f2).arrange(DOWN, buff=0.35).move_to(2.15 * UP)
        self.say("In a mixture the labels are hidden. Each point's density is a weighted sum of both curves,\n"
                 "and the weights π<sub>1</sub> and π<sub>2</sub>, which add to 1, say how common each curve is.", wait=0)
        self.play(*[dt.animate.set_color(GREY_DOT) for dt in dots], FadeOut(c1), FadeOut(c2), FadeOut(tt), run_time=1.5)
        self.play(FadeIn(f1, shift=0.1 * DOWN))
        self.wait(5)
        self.play(FadeIn(f2, shift=0.1 * DOWN))
        self.say("Now the log sits outside a sum, so it can't split the terms. Setting the gradient to zero\n"
                 "makes each μ<sub>k</sub> a weighted mean whose weights depend on μ<sub>k</sub> itself: no closed form.", wait=8)
        self.end_section()

    def em_steps(self):
        """
        Description: Runs EM's E-step and M-step on the mixture until the log-likelihood is within CLOSE of where it settles.
        Inputs: none
        Outputs: none
        """
        self.mark("em_steps")
        d = self.d
        params, resp, lls = d["em_params"], d["em_resp"], d["em_ll"]
        T = d["em_to_top"]
        self.heading("EM: Dempster, Laird and Rubin (1977)")
        dots, line = self.histogram()
        c1, c2 = component_curves(params[0])
        chart = Chart((0, T), (-250, -224), 3.4, 1.7, np.array([4.75, 1.85, 0]), xticks=(0, 10, 20),
                      yticks=(-250, -230), tick_size=18)
        chart_lab = M("ℓ by iteration", 22, GREY_B).next_to(chart.group, UP, buff=0.12)
        hist = VGroup(Dot(chart.c2p(0, lls[0]), radius=0.045, color=GOOD))
        readout = self.readout(0, lls[0])
        self.play(FadeIn(line), FadeIn(dots))
        self.say("Expectation-maximization (EM) turns that circular dependence into an algorithm.\n"
                 "Start from a guess: two wide bell curves near the middle.", wait=0)
        self.play(Create(c1), Create(c2), FadeIn(chart.group), FadeIn(chart_lab), FadeIn(hist), FadeIn(readout),
                  run_time=1.6)
        self.wait(4.5)

        i = d["zoom_idx"]
        a, b = float(comp(params[0], 0, X[i])), float(comp(params[0], 1, X[i]))
        xa, xb = sx(X[i]) - 0.07, sx(X[i]) + 0.07
        seg_a = Line(np.array([xa, Y0, 0]), np.array([xa, sy(a), 0])).set_stroke(BLUE_K, 6)
        seg_b = Line(np.array([xb, Y0, 0]), np.array([xb, sy(b), 0])).set_stroke(AMBER, 6)
        ring = Circle(radius=0.17).set_stroke(WHITE, 3).move_to(dots[i])
        zoom = VGroup(
            M(f"x = {X[i]:.2f}", 26),
            M(f"π<sub>1</sub>N<sub>1</sub>(x) = {a:.3f}", 26, BLUE_K),
            M(f"π<sub>2</sub>N<sub>2</sub>(x) = {b:.3f}", 26, AMBER),
            M(f"amber share = {b:.3f} / {a + b:.3f} = {b / (a + b):.2f}", 26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to(np.array([-3.6, 1.9, 0]))
        self.say("E-step: for each point, ask what share of its height each curve supplies.\n"
                 f"At x = {X[i]:.2f} the amber curve supplies {b:.3f} of {a + b:.3f}, a share of {b / (a + b):.2f}.", wait=0)
        self.play(Create(ring), Create(seg_a), Create(seg_b), run_time=1)
        self.play(FadeIn(zoom, lag_ratio=0.3), run_time=1.6)
        self.play(dots[i].animate.set_color(blend(resp[0][1][i])))
        self.wait(5)
        e_form = M("E-step:   r<sub>ik</sub> = π<sub>k</sub> N<sub>k</sub>(x<sub>i</sub>) / "
                   "∑<sub>j</sub> π<sub>j</sub> N<sub>j</sub>(x<sub>i</sub>)", 28)
        e_form.move_to(np.array([-6.6, 2.55, 0]), aligned_edge=LEFT)
        self.say("Do that for every point. These shares are the responsibilities r<sub>ik</sub>, soft labels:\n"
                 "each point is part blue and part amber.", wait=0)
        self.play(FadeOut(zoom), FadeOut(seg_a), FadeOut(seg_b), FadeOut(ring), FadeIn(e_form, shift=0.1 * DOWN))
        self.play(*[dt.animate.set_color(blend(resp[0][1][j])) for j, dt in enumerate(dots)], run_time=1.5)
        self.wait(4)
        m_form = VGroup(
            M("M-step:   N<sub>k</sub> = ∑<sub>i</sub> r<sub>ik</sub>     π<sub>k</sub> = N<sub>k</sub> / N", 28),
            M("μ<sub>k</sub> = ∑<sub>i</sub> r<sub>ik</sub> x<sub>i</sub> / N<sub>k</sub>     "
              "σ<sub>k</sub><sup>2</sup> = ∑<sub>i</sub> r<sub>ik</sub> (x<sub>i</sub> − μ<sub>k</sub>)<sup>2</sup> / N<sub>k</sub>", 28),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        m_form[1].shift(1.45 * RIGHT)
        m_form.next_to(e_form, DOWN, buff=0.3, aligned_edge=LEFT)
        self.say("M-step: freeze the soft labels and refit each curve as if they were true labels.\n"
                 "These are fact 3's formulas with weights, so each one has a closed form.", wait=0)
        self.play(FadeIn(m_form, shift=0.1 * DOWN))
        n1, n2 = component_curves(params[1])
        p1 = Dot(chart.c2p(1, lls[1]), radius=0.045, color=GOOD)
        self.play(Transform(c1, n1), Transform(c2, n2), Transform(readout, self.readout(1, lls[1])),
                  FadeIn(p1), Create(Line(hist[-1].get_center(), p1.get_center()).set_stroke(GOOD, 2)), run_time=1.5)
        hist.add(p1)
        self.wait(4)
        self.say(f"ℓ rose from {num(lls[0], '.1f')} to {num(lls[1], '.1f')}. Repeat: new soft labels, then new curves.", wait=0)
        for t in range(2, T + 1):
            n1, n2 = component_curves(params[t])
            pt = Dot(chart.c2p(t, lls[t]), radius=0.045, color=GOOD)
            self.play(*[dt.animate.set_color(blend(resp[t - 1][1][j])) for j, dt in enumerate(dots)],
                      Transform(c1, n1), Transform(c2, n2), Transform(readout, self.readout(t, lls[t])),
                      FadeIn(pt), Create(Line(hist[-1].get_center(), pt.get_center()).set_stroke(GOOD, 2)),
                      run_time=0.55 if t < 8 else 0.35)
            hist.add(pt)
        self.say(f"After {T} rounds ℓ = {num(lls[T], '.2f')}, within {CLOSE} of where EM settles, and it rose at\n"
                 "every round. For a Gaussian mixture, EM's log-likelihood never goes down.", wait=8)
        self.end_section()

    def why_it_climbs(self):
        """
        Description: Shows Jensen's inequality, then EM's lower bounds climbing the log-likelihood as a function of mu_2.
        Inputs: none
        Outputs: none
        """
        self.mark("why_it_climbs")
        d = self.d
        self.heading("Why ℓ never goes down: the log bends downward")
        ch = Chart((0, 4), (-2.2, 1.6), 5.6, 3.6, np.array([-3.4, 0.2, 0]), xticks=(1, 2, 3, 4), yticks=(-2, -1, 0, 1))
        logc = ch.curve(math.log, 0.1, 4).set_stroke(BLUE_K, 4)
        pa, pb = 0.4, 3.6
        chord = Line(ch.c2p(pa, math.log(pa)), ch.c2p(pb, math.log(pb))).set_stroke(GREY_B, 3)
        mid = (pa + pb) / 2
        avg_logs = (math.log(pa) + math.log(pb)) / 2
        top = Dot(ch.c2p(mid, math.log(mid)), radius=0.08, color=BLUE_K)
        low = Dot(ch.c2p(mid, avg_logs), radius=0.08, color=GREY_B)
        gap = DashedLine(low.get_center(), top.get_center(), dash_length=0.06).set_stroke(WHITE, 2)
        # The curve and the chord both rise to the right, so the space above the curve near the y-axis
        # and below-right of the low dot is empty.
        lab_top = M(f"log of the average = {math.log(mid):.2f}", 24, BLUE_K).next_to(top, UP, buff=0.5)
        lab_top.align_to(ch.c2p(0.15, 0), LEFT)
        lab_low = M(f"average of the logs = {avg_logs:.2f}", 24, GREY_B).next_to(low, DOWN, buff=0.2).align_to(low, LEFT)
        jensen = VGroup(
            M("log ∑<sub>k</sub> q<sub>k</sub> a<sub>k</sub>  ≥  ∑<sub>k</sub> q<sub>k</sub> log a<sub>k</sub>", 32),
            M("for weights q<sub>k</sub> ≥ 0 that add to 1", 24, GREY_B),
        ).arrange(DOWN, buff=0.22).move_to(np.array([3.6, 1.3, 0]))
        self.say("One more property of the log: it bends downward. So the log of an average is at least\n"
                 "the average of the logs. This is Jensen's inequality, published in 1906.", wait=0)
        self.play(FadeIn(ch.group), Create(logc), run_time=1.2)
        self.play(Create(chord), FadeIn(top), FadeIn(low), run_time=1)
        self.play(Create(gap), FadeIn(lab_top), FadeIn(lab_low), FadeIn(jensen, shift=0.1 * UP), run_time=1.2)
        self.wait(6)
        apply = VGroup(
            M("log ∑<sub>k</sub> π<sub>k</sub>N<sub>k</sub>(x)  =  log ∑<sub>k</sub> q<sub>k</sub> · "
              "π<sub>k</sub>N<sub>k</sub>(x) / q<sub>k</sub>", 28),
            M("≥  ∑<sub>k</sub> q<sub>k</sub> log( π<sub>k</sub>N<sub>k</sub>(x) / q<sub>k</sub> )", 28, GOOD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(np.array([3.4, -0.9, 0]))
        self.say("Apply it to each point's term in ℓ with any weights q<sub>k</sub>. The green side has the log inside\n"
                 "the sum, so it is easy to maximize. With q = the responsibilities, the two sides are equal.", wait=0)
        self.play(FadeIn(apply[0], shift=0.1 * UP))
        self.play(FadeIn(apply[1], shift=0.1 * UP))
        self.wait(8)
        self.clear_stage()

        steps, bounds, ll_m2 = d["stair_steps"], d["stair_bounds"], d["stair_ll"]
        ch = Chart((-1.5, 3.0), (-400, -215), 9.4, 4.4, np.array([-1.1, 0.15, 0]),
                   xticks=(-1, 0, 1, 2, 3), yticks=(-400, -350, -300, -250))
        xlab = M("μ<sub>2</sub>", 26, GREY_B).next_to(ch.c2p(3.0, -400), RIGHT, buff=0.25)
        ylab = M("ℓ", 26, GREY_B).next_to(ch.c2p(-1.5, -215), UP, buff=0.12)
        llc = ch.curve(ll_m2, -1.5, 3.0, n=200).set_stroke(WHITE, 4)
        note = M("other parameters held at their best values", 22, GREY_B).next_to(ch.group, UP, buff=0.12)
        dot = Dot(ch.c2p(steps[0], ll_m2(steps[0])), radius=0.09, color=AMBER)
        read = self.stair_readout(0, steps[0], ll_m2(steps[0]))
        self.say("Watch one parameter, μ<sub>2</sub>, with the others fixed. White is ℓ. The E-step builds the\n"
                 "green bound, which lies below ℓ everywhere and touches it at the current guess.", wait=0)
        self.play(FadeIn(ch.group), FadeIn(xlab), FadeIn(ylab), FadeIn(note), Create(llc), run_time=1.4)
        self.play(FadeIn(dot), FadeIn(read))
        for k, bound in enumerate(bounds):
            bc = ch.curve(bound, -1.5, 3.0, n=200).set_stroke(GOOD, 4)
            new = steps[k + 1]
            on_bound = Dot(ch.c2p(new, bound(new)), radius=0.07, color=GOOD)
            rise = DashedLine(on_bound.get_center(), ch.c2p(new, ll_m2(new)), dash_length=0.06).set_stroke(AMBER, 3)
            slow = k == 0
            self.play(Create(bc), run_time=1.4 if slow else 0.8)
            if slow:
                self.wait(3)
                self.say("The M-step jumps to the bound's peak. ℓ there is at least the bound, and the bound's\n"
                         "peak is at least the old ℓ, where the two touched. So ℓ cannot fall.", wait=0)
            self.play(dot.animate.move_to(on_bound), FadeIn(on_bound), run_time=1.2 if slow else 0.7)
            self.play(Create(rise), dot.animate.move_to(ch.c2p(new, ll_m2(new))),
                      Transform(read, self.stair_readout(k + 1, new, ll_m2(new))), run_time=1.0 if slow else 0.6)
            if slow:
                self.wait(5)
            self.play(FadeOut(bc), FadeOut(on_bound), run_time=0.4)
        lls = ", ".join(num(ll_m2(s), ".2f") for s in steps)
        self.say(f"Each round climbs: {lls}.\nEM is a staircase of lower bounds, and the log's downward bend is what builds it.", wait=6)
        self.end_section()

    def modern(self):
        """
        Description: Shows the autograd version, the responsibilities inside the gradient, and EM's step for a mean as a gradient step.
        Inputs: none
        Outputs: none
        """
        self.mark("modern")
        d = self.d
        self.heading("Today: gradient steps on the loss")
        code = VGroup(*[Text(s, font=MONO, font_size=22, t2c={"# NLL": AMBER, "# autograd": GREY_B}) for s in (
            "for step in range(1000):",
            "log_pi = torch.log_softmax(w, dim=0)",
            "log_px = Normal(mu, log_sd.exp()).log_prob(x[:, None]) + log_pi",
            "loss = -torch.logsumexp(log_px, dim=1).mean()   # NLL",
            "opt.zero_grad(); loss.backward(); opt.step()   # autograd",
        )]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        for line in code[1:]:
            line.shift(0.55 * RIGHT)  # the loop body's indent
        panel = SurroundingRectangle(code, buff=0.3, corner_radius=0.1).set_stroke(GREY_B, 1.5).set_fill("#12161b", 1)
        VGroup(panel, code).move_to(1.6 * UP)
        self.say("Today you would hand the same mixture to autograd. The loss is the negative log-likelihood\n"
                 "averaged over the points, and an optimizer nudges every parameter downhill.", wait=0)
        self.play(FadeIn(panel), FadeIn(code, lag_ratio=0.15), run_time=2)
        self.wait(6)
        self.say("logsumexp computes log ∑ exp(a<sub>k</sub>) as m + log ∑ exp(a<sub>k</sub> − m), with m the largest a<sub>k</sub>,\n"
                 "so nothing underflows. EM code computes its responsibilities the same way.", wait=8)
        grad = M("∂loss/∂μ<sub>k</sub> = −(1/N) ∑<sub>i</sub> "
                 f"<span foreground='{AMBER}'>r<sub>ik</sub></span> (x<sub>i</sub> − μ<sub>k</sub>) / σ<sub>k</sub><sup>2</sup>", 32)
        grad.move_to(0.55 * DOWN)
        self.say("Differentiate the loss with respect to a mean and the responsibilities appear inside the\n"
                 "gradient. Backprop computes the E-step's soft labels without being asked.", wait=0)
        self.play(FadeIn(grad, shift=0.1 * UP))
        self.wait(7)
        steps = VGroup(
            M("gradient descent:   μ<sub>k</sub> ← μ<sub>k</sub> − <span foreground='#FC6255'>η</span> · ∂loss/∂μ<sub>k</sub>", 30),
            M("EM's M-step:   μ<sub>k</sub> ← μ<sub>k</sub> − "
              f"<span foreground='{GOOD}'>(N σ<sub>k</sub><sup>2</sup> / N<sub>k</sub>)</span> · ∂loss/∂μ<sub>k</sub>", 30),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(grad, DOWN, buff=0.4)
        s1, s2 = d["em_mean_steps"]
        self.say("For the means, EM's M-step is exactly a gradient step, with its own step size for each curve\n"
                 f"(Xu and Jordan, 1996). Near the answer it is about {s1:.1f} for the left curve and {s2:.1f} for the right.", wait=0)
        self.play(FadeIn(steps[0], shift=0.1 * UP))
        self.play(FadeIn(steps[1], shift=0.1 * UP))
        self.wait(9)
        self.end_section()

    def race(self):
        """
        Description: Races EM against gradient descent at several step sizes, then lists why networks train with gradients.
        Inputs: none
        Outputs: none
        """
        self.mark("race")
        d = self.d
        self.heading("The same mixture, two ways to climb")
        ch = Chart((0, 3), (-270, -222), 8.6, 4.2, np.array([-1.6, 0.05, 0]), xticks=(), yticks=(-270, -250, -230))
        ticks = VGroup()
        for e in range(4):
            p = ch.c2p(e, -270)
            ticks.add(Line(p + 0.08 * DOWN, p + 0.08 * UP).set_stroke(GREY_B, 2))
            ticks.add(Text(str(10 ** e), font=MONO, font_size=20, color=GREY_B).next_to(p, DOWN, buff=0.15))
        xlab = M("iterations, log scale", 22, GREY_B).next_to(ch.c2p(3, -270), RIGHT, buff=0.3)
        ylab = M("ℓ", 26, GREY_B).next_to(ch.c2p(0, -222), UP, buff=0.12)
        top = d["to_top"]
        lines, legend, results = VGroup(), VGroup(), []
        series = [("EM", "EM", d["em_ll"], GOOD)] + [(e, f"η = {fmt(e)}", d["gd_ll"][e], ETA_COLORS[e]) for e in ETAS]
        for key, name, lls, col in series:
            pts = [ch.c2p(math.log10(t), min(max(lls[t], -270), -222)) for t in range(1, RACE_STEPS + 1)]
            line = VMobject().set_points_as_corners(pts)
            if key == "EM":
                line.set_stroke(col, 5).set_z_index(2)  # stays on top of the gradient-descent curves
            elif key == 3.0:
                line.set_stroke(col, 1.5, opacity=0.5)  # a thousand jagged steps; keep them from hiding the rest
            else:
                line.set_stroke(col, 3)
            lines.add(line)
            legend.add(VGroup(Line(ORIGIN, 0.45 * RIGHT).set_stroke(col, 5), M(name, 24)).arrange(RIGHT, buff=0.18))
            results.append(M(f"{name}:  {top[key]}" if top[key] else f"{name}:  never settles", 23))
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to(np.array([3.1, 0.9, 0]), aligned_edge=LEFT)
        res_head = M(f"iterations until ℓ is within\n{CLOSE} of the top", 22, GREY_B)
        res_head.next_to(legend, UP, buff=0.3, aligned_edge=LEFT)
        self.say("Race them on this dataset from the same start. η is gradient descent's one step size,\n"
                 "shared by all five parameters. The iteration axis is on a log scale.", wait=0)
        self.play(FadeIn(ch.group), FadeIn(ticks), FadeIn(xlab), FadeIn(ylab), run_time=1)
        self.play(Create(lines[0]), FadeIn(legend[0]), run_time=1.5)
        for k in range(1, len(series)):
            self.play(Create(lines[k]), FadeIn(legend[k]), run_time=1.5)
        self.wait(2)
        self.say(f"EM gets within {CLOSE} of the top in {top['EM']} iterations, with nothing to tune. Gradient descent\n"
                 f"takes {top[1.0]} steps at η = 1, {top[0.3]} at η = 0.3 and {top[0.1]} at η = 0.1. At η = 3 it never settles.", wait=0)
        self.play(FadeIn(res_head),
                  *[Transform(row[1], new.move_to(row[1], aligned_edge=LEFT)) for row, new in zip(legend, results)],
                  run_time=1.5)
        self.wait(8)
        self.say(f"One η serves every parameter, so the stiffest direction sets the limit: here η must stay below\n"
                 f"about {d['eta_limit']:.1f}, or the left curve's mean and width bounce. EM picks a step for each curve.", wait=9)
        self.clear_stage()

        self.heading("So why do neural networks train with gradients?")
        reasons = VGroup(*[Text(s, font=SERIF, font_size=27) for s in (
            "1.  No closed-form M-step. A deep network's best weights have no weighted-mean\n"
            "     formula, but autograd gives a gradient for any differentiable model.",
            "2.  Minibatches. Each EM step reads the whole dataset; a gradient step reads a\n"
            "     few hundred examples, because ℓ is a sum.",
            "3.  Often nothing is hidden. A classifier's labels are in the data, so its\n"
            "     log-likelihood is just cross-entropy and there is no E-step.",
            "4.  When the hidden part is rich, like an image's latent code, the exact\n"
            "     E-step is out of reach.",
        )]).arrange(DOWN, aligned_edge=LEFT, buff=0.38).move_to(0.35 * UP)
        if reasons.width > 13:
            reasons.scale_to_fit_width(13)
        self.say("EM wins this small race.\nNeural networks still train with gradients, for reasons of scale and generality.", wait=0)
        for r in reasons:
            self.play(FadeIn(r, shift=0.1 * RIGHT), run_time=0.8)
            self.wait(4.5)
        self.wait(3)
        self.end_section()

    def then_and_now(self):
        """
        Description: Compares EM with gradient training, then shows EM's bound surviving as the VAE's ELBO.
        Inputs: none
        Outputs: none
        """
        self.mark("then_and_now")
        self.heading("EM's ideas in today's models")
        data = (
            ("one step", "closed-form refit", "small step along the gradient"),
            ("step size", "built in", "η, by hand or adapted (Adam)"),
            ("data per step", "the whole dataset", "a minibatch"),
            ("works for", "closed-form E- and M-steps", "any differentiable model"),
            ("guarantee", "ℓ never decreases", "none for one minibatch step"),
        )
        cols = (-6.6, -3.75, 1.75)
        y0 = 1.95
        table = VGroup(VGroup(Text("·", font=SERIF, font_size=24).set_opacity(0),
                              Text("EM (1977)", font=SERIF, font_size=26, color=GOOD),
                              Text("gradient training (today)", font=SERIF, font_size=26, color=BLUE_K)))
        for k, (what, em, gd) in enumerate(data):
            table.add(VGroup(Text(what, font=SERIF, font_size=24, color=GREY_B),
                             M(em, 25), M(gd, 25)))
        for k, row in enumerate(table):
            for c, cell in enumerate(row):
                cell.move_to(np.array([cols[c], y0 - 0.62 * k, 0]), aligned_edge=LEFT)
        rule = Line(np.array([-6.5, y0 - 0.32, 0]), np.array([6.8, y0 - 0.32, 0])).set_stroke(GREY_B, 1.5)
        self.say("Side by side: EM gets a step with nothing to tune that never goes downhill, and pays for it\n"
                 "by needing closed-form steps and the full dataset.", wait=0)
        self.play(FadeIn(table[0]), Create(rule))
        self.play(LaggedStart(*[FadeIn(r, shift=0.1 * RIGHT) for r in table[1:]], lag_ratio=0.3), run_time=3)
        self.wait(7)
        self.clear_stage()

        x_box = box("x", 1.0, 0.9, GREY_B)
        enc = box("encoder", 2.2, 0.9, GOOD)
        z_box = box("z", 1.0, 0.9, AMBER)
        dec = box("decoder", 2.2, 0.9, BLUE_K)
        xh = box("x̂", 1.0, 0.9, GREY_B)
        chain = VGroup(x_box, enc, z_box, dec, xh).arrange(RIGHT, buff=0.75).move_to(1.6 * UP)
        arrows = VGroup(*[arrow(a, b) for a, b in zip(chain, chain[1:])])
        q_lab = M("predicts q(z | x)", 24, GOOD).next_to(enc, UP, buff=0.18)
        p_lab = M("models p(x | z)", 24, BLUE_K).next_to(dec, UP, buff=0.18)
        elbo = VGroup(
            M("∑<sub>k</sub> q<sub>k</sub> log( π<sub>k</sub>N<sub>k</sub>(x) / q<sub>k</sub> )   becomes", 28, GREY_B),
            M("ELBO = 𝔼<sub>q(z | x)</sub>[ log p(x | z) ] − KL( q(z | x) ‖ p(z) )  ≤  log p(x)", 30, GOOD),
        ).arrange(DOWN, buff=0.3).move_to(0.6 * DOWN)
        self.say("EM's bound is still here. A variational autoencoder (Kingma and Welling, 2013) maximizes\n"
                 "the same lower bound, now called the ELBO, by gradient steps on minibatches.", wait=0)
        self.play(LaggedStart(*[FadeIn(b) for b in chain], lag_ratio=0.15), run_time=1.5)
        self.play(*[GrowArrow(a) for a in arrows], FadeIn(q_lab), FadeIn(p_lab))
        self.play(FadeIn(elbo[0], shift=0.1 * UP))
        self.play(FadeIn(elbo[1], shift=0.1 * UP))
        self.wait(6)
        self.say("Its encoder predicts the soft labels q(z | x) in one pass, standing in for the E-step, and\n"
                 "the decoder is the model. Neal and Hinton (1998) showed both EM steps raise this same bound.", wait=10)
        self.end_section()

    def summary(self):
        """
        Description: Lists the video's four conclusions.
        Inputs: none
        Outputs: none
        """
        self.mark("summary")
        self.heading("Summary")
        lines = VGroup(*[Text(s, font=SERIF, font_size=26) for s in (
            "The log turns the likelihood's product into a sum: no underflow, and minibatches.",
            "Minimizing the negative log-likelihood gives cross-entropy and squared-error losses.",
            "EM fits a Gaussian mixture by alternating soft labels (E) and closed-form refits (M),\n"
            "and the log's downward bend guarantees ℓ never falls.",
            "Gradient training climbs the same ℓ with small steps that work for any\n"
            "differentiable model, and VAEs keep EM's lower bound as the ELBO.",
        )])
        for ln in lines:
            if ln.width > 13:
                ln.scale_to_fit_width(13)
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to(0.3 * UP)
        self.play(LaggedStart(*[FadeIn(ln, shift=0.1 * RIGHT) for ln in lines], lag_ratio=0.5), run_time=4)
        self.wait(12)
        self.end_section()

    def histogram(self, colors=None):
        """
        Description: Draws the data as stacked dots over a number line.
        Inputs:
        - colors: one colour per point, or None for grey.
        Outputs: (dots, number line), where dots[i] is the dot for X[i].
        """
        dots = VGroup(*[Dot(p, radius=DOT_R, color=GREY_DOT) for p in self.d["dot_pos"]])
        if colors is not None:
            for dt, c in zip(dots, colors):
                dt.set_color(c)
        return dots, number_line()

    def readout(self, t, ll):
        """
        Description: Builds the EM iteration counter shown under the mini chart.
        Inputs:
        - t: the iteration number.
        - ll: the log-likelihood after that iteration.
        Outputs: the text mobject.
        """
        return M(f"iteration {t}     ℓ = {num(ll, '.1f')}", 26).move_to(np.array([4.75, 0.25, 0]))

    def stair_readout(self, k, m2, ll):
        """
        Description: Builds the staircase's round counter.
        Inputs:
        - k: the round number.
        - m2: the current value of mu_2.
        - ll: the log-likelihood at that value.
        Outputs: the three-line text group.
        """
        return VGroup(M(f"round {k}", 26, GREY_B), M(f"μ<sub>2</sub> = {num(m2, '.3f')}", 26, AMBER),
                      M(f"ℓ = {num(ll, '.2f')}", 26)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to(np.array([5.4, 1.2, 0]))

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

    def __init__(self, xr, yr, xl, yl, centre, xticks=(), yticks=(), tick_size=20):
        """
        Description: Builds the frame with its ticks and tick labels.
        Inputs:
        - xr, yr: the data ranges, as (low, high).
        - xl, yl: the frame's width and height, in screen units.
        - centre: the frame's centre on screen.
        - xticks, yticks: the values to tick on each axis.
        - tick_size: the tick labels' font size.
        Outputs: none
        """
        self.xr, self.yr, self.xl, self.yl = xr, yr, xl, yl
        self.origin = centre - np.array([xl / 2, yl / 2, 0])
        g = VGroup(Line(self.c2p(xr[0], yr[0]), self.c2p(xr[1], yr[0])).set_stroke(GREY_B, 2),
                   Line(self.c2p(xr[0], yr[0]), self.c2p(xr[0], yr[1])).set_stroke(GREY_B, 2))
        for v in xticks:
            p = self.c2p(v, yr[0])
            g.add(Line(p + 0.07 * DOWN, p + 0.07 * UP).set_stroke(GREY_B, 2))
            g.add(Text(num(v, "g"), font=MONO, font_size=tick_size, color=GREY_B).next_to(p, DOWN, buff=0.14))
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
        Outputs: the polyline.
        Constraints: each curve drawn here leaves the y-range at most once at each end, so dropping points
        never joins two separate pieces.
        """
        pts = []
        for v in np.linspace(a, b, n):
            y = f(v)
            if self.yr[0] <= y <= self.yr[1]:
                pts.append(self.c2p(v, y))
        return VMobject().set_points_as_corners(pts)

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
    Description: Computes every number the video shows and asserts each claim a caption makes.
    Inputs: none
    Outputs: a dict holding the EM and gradient-descent trajectories, the staircase's bounds, the dot
    positions and the numbers the captions quote.
    Constraints: raises AssertionError when a claim fails, for example if EM's log-likelihood ever decreases.
    """
    d = {}
    params, resp = [INIT], []
    for _ in range(RACE_STEPS):
        new, r = em_step(params[-1])
        params.append(new)
        resp.append(r)
    lls = np.array([loglik(p) for p in params])
    assert np.all(np.diff(lls) >= -1e-9), "EM's log-likelihood went down"
    best = lls[-1]
    d.update(em_params=params, em_resp=resp, em_ll=lls, best_params=params[-1], ll_best=best)
    d["em_to_top"] = first_within(lls, best)

    dens = mix_pdf(params[-1], X)
    log10_sum = float(np.log10(dens).sum())
    exponent = math.floor(log10_sum)
    d["prod_text"] = f"{10 ** (log10_sum - exponent):.1f} × 10<sup>{num(exponent, 'd')}</sup>"
    assert np.prod(dens.astype(np.float32)) == 0.0, "the float32 product should underflow"
    assert np.prod(dens) > 0, "the float64 product should not underflow"
    d["exp_1000"] = round(1000 * float(np.log10(dens).mean()))
    batch = np.random.default_rng(3).choice(N, BATCH, replace=False)
    d["mean_all"], d["mean_batch"] = float(np.log(dens).mean()), float(np.log(dens[batch]).mean())
    d["sample_idx"] = [int(np.argmin(abs(X - v))) for v in (-2.9, -1.6, 0.2, 1.75, 3.6)]

    known = (np.array([72, 48]) / N, np.array([X[LABEL == k].mean() for k in (0, 1)]),
             np.array([X[LABEL == k].std() for k in (0, 1)]))
    d["known_params"] = known
    d["zoom_idx"] = int(np.argmin(abs(X - 2.5)))

    # EM's M-step for each mean is a gradient step on the mean NLL with step N sigma_k^2 / N_k (Xu and Jordan, 1996).
    pi0, mu0, sd0 = INIT
    r0 = resp[0]
    nk = r0.sum(1)
    dloss_dmu = -(r0 * (X - mu0[:, None])).sum(1) / sd0 ** 2 / N
    assert np.allclose(mu0 - N * sd0 ** 2 / nk * dloss_dmu, params[1][1]), "M-step is not that gradient step"
    pi_b, mu_b, sd_b = params[-1]
    d["em_mean_steps"] = sd_b ** 2 / pi_b  # N sigma^2 / N_k with N_k = pi_k N at the answer

    th0 = np.array([0.0, mu0[0], mu0[1], math.log(sd0[0]), math.log(sd0[1])])
    num_g = np.array([(mean_ll(th0 + 1e-6 * e) - mean_ll(th0 - 1e-6 * e)) / 2e-6 for e in np.eye(5)])
    assert np.allclose(num_g, mean_ll_grad(th0), atol=1e-7), "analytic gradient is wrong"
    d["gd_ll"] = {e: ascend(th0, e) for e in ETAS}
    d["to_top"] = {"EM": d["em_to_top"], **{e: first_within(d["gd_ll"][e], best) for e in ETAS}}
    assert d["to_top"][3.0] is None and np.std(d["gd_ll"][3.0][-100:]) > 0.5, "eta = 3 should keep bouncing"
    assert all(d["to_top"][e] for e in (1.0, 0.3, 0.1)), "the smaller steps should all arrive"

    th_b = np.array([math.log(pi_b[0] / pi_b[1]), mu_b[0], mu_b[1], math.log(sd_b[0]), math.log(sd_b[1])])
    H = np.array([(mean_ll_grad(th_b + 1e-5 * e) - mean_ll_grad(th_b - 1e-5 * e)) / 2e-5 for e in np.eye(5)])
    w, v = np.linalg.eigh((H + H.T) / 2)
    d["eta_limit"] = 2 / abs(w[0])
    assert set(np.argsort(-abs(v[:, 0]))[:2]) == {1, 3}, "stiffest direction should be the left curve's mean and width"
    assert 1.0 < d["eta_limit"] < 3.0

    def ll_m2(m):
        """
        Description: Computes the log-likelihood as a function of mu_2, with the other parameters at EM's answer.
        Inputs:
        - m: the value of mu_2.
        Outputs: the log-likelihood.
        """
        return loglik((pi_b, np.array([mu_b[0], m]), sd_b))

    steps, bounds = [-1.0], []
    grid = np.linspace(-1.5, 3.0, 181)
    for _ in range(4):
        m = steps[-1]
        here = (pi_b, np.array([mu_b[0], m]), sd_b)
        a, b = comp(here, 0, X), comp(here, 1, X)
        q = np.stack([a / (a + b), b / (a + b)])
        bound = make_bound(q, pi_b, mu_b[0], sd_b)
        assert abs(bound(m) - ll_m2(m)) < 1e-6, "the bound should touch ell at the current guess"
        assert all(bound(g) <= ll_m2(g) + 1e-7 for g in grid), "the bound should lie below ell"
        bounds.append(bound)
        steps.append(float((q[1] * X).sum() / q[1].sum()))
    assert all(ll_m2(b) > ll_m2(a) for a, b in zip(steps, steps[1:]))
    d.update(stair_steps=steps, stair_bounds=bounds, stair_ll=ll_m2)

    stacks, counts = np.zeros(N, int), {}
    bins = np.floor((X - XMIN) / BIN).astype(int)
    for i in np.argsort(X, kind="stable"):
        stacks[i] = counts.get(bins[i], 0)
        counts[bins[i]] = stacks[i] + 1
    d["dot_pos"] = [np.array([sx(XMIN + (bins[i] + 0.5) * BIN), Y0 + (stacks[i] + 0.5) * DOT_H + 0.02, 0])
                    for i in range(N)]
    for k, v in d.items():
        if k not in ("em_params", "em_resp", "em_ll", "gd_ll", "dot_pos", "stair_bounds", "stair_ll", "best_params"):
            print(f"  {k}: {v}".encode("ascii", "replace").decode())  # Windows consoles may not be UTF-8
    return d


def em_step(params):
    """
    Description: Runs one EM iteration: the E-step's responsibilities, then the M-step's closed-form refit.
    Inputs:
    - params: the current (pi, mu, sigma).
    Outputs: (new params, r), where r is the 2 x N array of responsibilities from the E-step.
    """
    a, b = comp(params, 0, X), comp(params, 1, X)
    r = np.stack([a / (a + b), b / (a + b)])
    nk = r.sum(1)
    mu = (r * X).sum(1) / nk
    sd = np.sqrt((r * (X - mu[:, None]) ** 2).sum(1) / nk)
    return (nk / N, mu, sd), r


def ascend(th, eta):
    """
    Description: Runs plain gradient ascent on the mean log-likelihood, which is gradient descent on the loss.
    Inputs:
    - th: the starting parameters, as (logit pi_1, mu_1, mu_2, log sigma_1, log sigma_2).
    - eta: the step size.
    Outputs: the summed log-likelihood before the first step and after each of RACE_STEPS steps.
    """
    lls = [mean_ll(th) * N]
    for _ in range(RACE_STEPS):
        th = th + eta * mean_ll_grad(th)
        lls.append(mean_ll(th) * N)
    return np.array(lls)


def mean_ll_grad(th):
    """
    Description: Computes the gradient of the mean log-likelihood.
    Inputs:
    - th: the parameters, as (logit pi_1, mu_1, mu_2, log sigma_1, log sigma_2).
    Outputs: the gradient, in the same order.
    """
    params = unpack(th)
    a, b = comp(params, 0, X), comp(params, 1, X)
    r = np.stack([a / (a + b), b / (a + b)])
    mu, sd = params[1], params[2]
    z = (X - mu[:, None]) / sd[:, None]
    g = np.concatenate([[(r[0] - params[0][0]).sum()], (r * z / sd[:, None]).sum(1), (r * (z ** 2 - 1)).sum(1)])
    return g / N


def mean_ll(th):
    """
    Description: Computes the mean log-likelihood, which is the negative of the loss.
    Inputs:
    - th: the parameters, as (logit pi_1, mu_1, mu_2, log sigma_1, log sigma_2).
    Outputs: the mean log-likelihood.
    """
    return loglik(unpack(th)) / N


def unpack(th):
    """
    Description: Converts unconstrained parameters to the mixture's (pi, mu, sigma).
    Inputs:
    - th: the parameters, as (logit pi_1, mu_1, mu_2, log sigma_1, log sigma_2).
    Outputs: (pi, mu, sigma) arrays over the two curves.
    """
    p1 = 1 / (1 + math.exp(-th[0]))
    return (np.array([p1, 1 - p1]), np.array(th[1:3]), np.exp(th[3:5]))


def make_bound(q, pi, mu1, sd):
    """
    Description: Builds EM's lower bound on the log-likelihood as a function of mu_2, with the soft labels frozen.
    Inputs:
    - q: the 2 x N responsibilities to freeze.
    - pi, sd: the weights and widths, held fixed.
    - mu1: the left curve's mean, held fixed.
    Outputs: a function from mu_2 to the bound's value.
    """
    def bound(m):
        """
        Description: Evaluates the lower bound at one value of mu_2.
        Inputs:
        - m: the value of mu_2.
        Outputs: the bound's value.
        """
        params = (pi, np.array([mu1, m]), sd)
        total = 0.0
        for k in range(2):
            ok = q[k] > 0
            total += (q[k][ok] * (np.log(comp(params, k, X[ok])) - np.log(q[k][ok]))).sum()
        return float(total)
    return bound


def first_within(lls, best):
    """
    Description: Finds the first iteration whose log-likelihood is within CLOSE of best.
    Inputs:
    - lls: the log-likelihood after each iteration, starting from iteration 0.
    - best: the value to approach.
    Outputs: that iteration's index, or None when no iteration gets that close.
    """
    hits = np.nonzero(best - np.asarray(lls) < CLOSE)[0]
    return int(hits[0]) if len(hits) else None


def loglik(params):
    """
    Description: Computes the mixture's log-likelihood of the data.
    Inputs:
    - params: (pi, mu, sigma).
    Outputs: the log-likelihood summed over all N points.
    """
    return float(np.log(mix_pdf(params, X)).sum())


def mix_pdf(params, v):
    """
    Description: Evaluates the mixture density.
    Inputs:
    - params: (pi, mu, sigma).
    - v: a value or an array of values.
    Outputs: the density at v.
    """
    return comp(params, 0, v) + comp(params, 1, v)


def comp(params, k, v):
    """
    Description: Evaluates one weighted bell curve, pi_k N(v | mu_k, sigma_k^2).
    Inputs:
    - params: (pi, mu, sigma).
    - k: which curve, 0 or 1.
    - v: a value or an array of values.
    Outputs: the weighted density at v.
    """
    pi, mu, sd = params
    return pi[k] * np.exp(-0.5 * ((v - mu[k]) / sd[k]) ** 2) / (sd[k] * math.sqrt(2 * math.pi))


def component_curves(params):
    """
    Description: Draws both weighted bell curves over the number line.
    Inputs:
    - params: (pi, mu, sigma).
    Outputs: (left curve, right curve), each in its own colour.
    """
    return (density_curve(lambda v: comp(params, 0, v)).set_stroke(BLUE_K, 4),
            density_curve(lambda v: comp(params, 1, v)).set_stroke(AMBER, 4))


def density_curve(f, n=220):
    """
    Description: Draws a density over the number line at the dot histogram's scale.
    Inputs:
    - f: the density.
    - n: the number of sample points.
    Outputs: the polyline.
    """
    return VMobject().set_points_as_corners([np.array([sx(v), sy(float(f(v))), 0]) for v in np.linspace(XMIN, XMAX, n)])


def number_line():
    """
    Description: Draws the number line under the dot histogram, with a tick at each integer.
    Inputs: none
    Outputs: the line, its ticks and their labels.
    """
    g = VGroup(Line(np.array([sx(XMIN), Y0, 0]), np.array([sx(XMAX), Y0, 0])).set_stroke(GREY_B, 2))
    for v in range(-3, 5):
        p = np.array([sx(v), Y0, 0])
        g.add(Line(p + 0.07 * DOWN, p + 0.07 * UP).set_stroke(GREY_B, 2))
        g.add(Text(num(v, "d"), font=MONO, font_size=18, color=GREY_B).next_to(p, DOWN, buff=0.1))
    return g


def blend(r):
    """
    Description: Colours a point by how much each bell curve owns it.
    Inputs:
    - r: the right curve's responsibility for the point.
    Outputs: blue at 0, amber at 1, and a mix of the two in between.
    """
    return interpolate_color(ManimColor(BLUE_K), ManimColor(AMBER), float(r))


def box(label, w, h, colour):
    """
    Description: Draws a labelled rounded box for the VAE diagram.
    Inputs:
    - label: the text inside.
    - w, h: the box's width and height.
    - colour: the border colour.
    Outputs: the box and its label.
    """
    r = RoundedRectangle(corner_radius=0.12, width=w, height=h).set_stroke(colour, 3).set_fill(config.background_color, 1)
    return VGroup(r, M(label, 28).move_to(r))


def arrow(a, b):
    """
    Description: Draws an arrow from the right side of one box to the left side of the next.
    Inputs:
    - a, b: the two boxes.
    Outputs: the arrow.
    """
    return Arrow(a.get_right(), b.get_left(), buff=0.08, stroke_width=4, color=GREY_B,
                 max_tip_length_to_length_ratio=0.25)


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


def sx(v):
    """
    Description: Maps a data value to its screen x on the number line.
    Inputs:
    - v: the data value.
    Outputs: the screen x, which is (v - 0.5) * 1.3.
    """
    return (v - 0.5) * 1.3


def sy(f):
    """
    Description: Maps a density to its screen height above the number line.
    Inputs:
    - f: the density.
    Outputs: the screen y.
    """
    return Y0 + f * CURVE_SCALE


def num(v, spec):
    """
    Description: Formats a number for display with a true minus sign.
    Inputs:
    - v: the number.
    - spec: a format spec, such as ".2f".
    Outputs: the formatted string.
    """
    return format(v, spec).replace("-", "−")


def fmt(v):
    """
    Description: Formats a step size without a trailing ".0".
    Inputs:
    - v: the number.
    Outputs: the formatted string.
    """
    return str(int(v)) if float(v).is_integer() else str(v)
