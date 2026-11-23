"""Renders a Manim video on KL divergence and three jobs it does when training language models.

Contents:
- KLDivergence: the scene, with one method per section of the video
- Chart: a plot frame whose ranges need not contain 0
- precompute: every number on screen, with asserts for the claims the captions make
- entropy, cross_entropy, kl: the three information quantities on a discrete distribution
- train_path: gradient descent on a softmax model's scores toward sampled next words
- gauss, mixture, kl_fwd, kl_rev, fit_path: fitting one bell curve to two humps in each KL direction
- elbo, rlhf_policy: the ELBO gap for the EM mixture, and the KL-penalized RLHF optimum
- word_bars, bar_group, label_row, area_rects, rect_at, polyline, cached, M, num: drawing helpers and number formatting
- fonts, colours, the synthetic examples and the layout constants
"""
import os

import numpy as np
from manim import *

SERIF = "Cambria"
"""Description: Font for captions, titles and labels."""
MONO = "Consolas"
"""Description: Monospace font for tick labels and live readouts, so digits line up."""
MATH = "Cambria"
"""Description: Font for formulas, because Cambria has the Greek letters, ∑, ‖ and the arrows they need."""
P_COL = "#FFD166"
"""Description: Colour of the true distribution P and of the data's frequencies."""
Q_COL = "#58C4DD"
"""Description: Colour of the model Q, and of the tuned model π in the RLHF section."""
KL_COL = "#FC6255"
"""Description: Colour of KL divergence, and of the reverse-direction fit."""
GOOD = "#83C167"
"""Description: Colour of rewards, of the ELBO, and of rectangles that shrank."""
REF_COL = "#8b96a3"
"""Description: Colour of the reference model π_ref before RLHF tuning."""
config.background_color = "#0b0d10"

WORDS = ("box", "cup", "bolt", "wrench", "cable")
"""Description: The five candidate next words after the prompt "The robot arm picked up the"."""
P_TRUE = np.array([0.40, 0.25, 0.15, 0.12, 0.08])
"""Description: P, how often each word really follows the prompt in the synthetic data."""
Q_MODEL = np.array([0.30, 0.35, 0.02, 0.23, 0.10])
"""Description: Q, the model's prediction; it nearly misses "bolt", which makes the two KL directions differ."""
N_SAMPLES = 2000
"""Description: How many synthetic training sentences continue the prompt."""
SAMPLE_SEED = 7
"""Description: The seed for drawing the training sentences' next words from P."""
TRAIN_LR, TRAIN_STEPS = 1.0, 60
"""Description: The learning rate and step count of gradient descent on the model's scores."""
HUMP_W, HUMP_MU, HUMP_SD = np.array([0.6, 0.4]), np.array([-2.0, 2.0]), np.array([0.6, 0.6])
"""Description: The two-hump distribution in the directions section: weights, centres and widths."""
FIT_START = (-1.0, 1.0)
"""Description: The (centre, width) both fits start from in the directions section."""
FIT_LR, FIT_STEPS = 0.05, 4000
"""Description: The step size and step count of gradient descent on (centre, log width)."""
XS = np.linspace(-12.0, 12.0, 8001)
"""Description: The grid that the continuous KL integrals are summed over."""
EM_W, EM_MU, EM_SD = np.array([0.6, 0.4]), np.array([-1.6, 1.9]), np.array([0.75, 1.0])
"""Description: The true settings of the two-curve mixture from the EM video."""
X_POINT = 0.0
"""Description: The data point whose ELBO gap is shown."""
PI_REF = np.array([0.30, 0.30, 0.20, 0.15, 0.05])
"""Description: π_ref, the reference model's probability for each of five candidate responses."""
REWARD = np.array([1.0, 1.2, 0.8, 0.0, 3.0])
"""Description: R, the reward model's score for each response; y5 scores highest though π_ref rarely writes it."""
BETA_LOGS = np.linspace(1.0, -1.0, 401)
"""Description: The log10 β values of the RLHF sweep, from β = 10 down to β = 0.1."""
RL_SCALE = 2.5
"""Description: Screen height per unit of probability in the RLHF bars, so a bar at probability 1 clears the legend."""
RECT_LEFT, RECT_WIDTH, RECT_BASE, RECT_SCALE = -6.4, 8.4, -1.75, 0.8
"""Description: The area picture's left edge, total width for probability 1, baseline, and screen height per nat."""
PANEL_X = 2.5
"""Description: Screen x of the left edge of the formulas beside the area picture."""
TEXT_CACHE = {}
"""Description: Text mobjects already built, keyed by (kind, text, size, colour), for cached()."""


class KLDivergence(Scene):
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
        Constraints: when the KL_ONLY environment variable names sections, separated by commas, only those play.
        """
        self.cap = None
        self.head = None
        self.d = precompute()
        only = [s for s in os.environ.get("KL_ONLY", "").split(",") if s]
        for name in ("intro", "surprise", "kl_gap", "pretraining", "directions", "elbo_gap", "rlhf", "summary"):
            if not only or name in only:
                getattr(self, name)()

    def intro(self):
        """
        Description: Shows the title, then the next-word example with the true distribution P beside the model Q.
        Inputs: none
        Outputs: none
        """
        self.mark("intro")
        title = Text("KL divergence", font=SERIF, font_size=54)
        sub = Text("one formula behind the pretraining loss, the ELBO and RLHF", font=SERIF, font_size=30, color=GREY_B)
        sub.next_to(title, DOWN, buff=0.3)
        words = VGroup(title, sub).move_to(0.4 * UP)
        self.play(Write(title), FadeIn(sub, shift=0.2 * UP), run_time=1.6)
        self.wait(2.5)
        self.play(FadeOut(words))
        self.heading("How different are two distributions?")
        prompt = M("“The robot arm picked up the ___”", 32).move_to(np.array([-1.2, 2.55, 0]))
        chart = word_bars([P_TRUE, Q_MODEL], [P_COL, Q_COL], x0=-4.4, dx=1.6, base=-1.5, scale=6.0, width=0.55, values=True)
        key = VGroup(M("P: how often each word", 24, P_COL), M("really follows (synthetic)", 24, P_COL),
                     M("Q: what a model predicts", 24, Q_COL)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        key[2].shift(0.15 * DOWN)
        key.move_to(np.array([3.4, 1.5, 0]), aligned_edge=LEFT)
        self.say("A language model predicts the next word by giving every word a probability. Here P is how often\n"
                 "each word really follows this prompt, in synthetic data, and Q is what one model predicts.", wait=0)
        self.play(FadeIn(prompt, shift=0.1 * DOWN), FadeIn(chart[0]), run_time=1)
        self.play(FadeIn(chart[1], lag_ratio=0.1), FadeIn(key[:2]), run_time=1.5)
        self.play(FadeIn(chart[2], lag_ratio=0.1), FadeIn(key[2]), run_time=1.5)
        self.wait(5)
        self.say("Q is close to P on some words and far on others: it gives bolt 0.02 where P gives 0.15. KL divergence\n"
                 "turns all of those differences into one number. Building it starts with surprise.")
        self.end_section()

    def surprise(self):
        """
        Description: Shows surprise as −log p, then entropy as the area of rectangles of width P(y) and height −log P(y).
        Inputs: none
        Outputs: none
        """
        self.mark("surprise")
        d = self.d
        self.heading("Surprise, and its average: entropy")
        ch = Chart((0, 1), (0, 4.6), 5.4, 3.5, np.array([-3.0, 0.15, 0]), xticks=(0, 0.25, 0.5, 0.75, 1),
                   yticks=(0, 1, 2, 3, 4), tick_size=18)
        xl = M("probability p", 22, GREY_B).next_to(ch.c2p(0.5, 0), DOWN, buff=0.45)
        yl = M("surprise −log p, in nats", 22, GREY_B).move_to(ch.c2p(0, 4.6) + 0.3 * UP, aligned_edge=LEFT)
        curve = ch.curve(lambda p: -np.log(p), 0.0105, 1.0, n=400).set_stroke(WHITE, 4)
        dots = VGroup(*[Dot(ch.c2p(P_TRUE[j], -np.log(P_TRUE[j])), radius=0.07, color=P_COL) for j in range(len(WORDS))])
        rows = [("word", "P(y)", "−log P(y)", GREY_B)] + [(w, f"{P_TRUE[j]:.2f}", f"{-np.log(P_TRUE[j]):.2f}", P_COL)
                                                         for j, w in enumerate(WORDS)]
        table = VGroup(*[label_row((a, b, c), (1.9, 3.9, 5.3), 1.7 - 0.5 * i, 24, (col,) * 3, align="left")
                         for i, (a, b, c, col) in enumerate(rows)])
        self.say("The surprise of an outcome with probability p is −log p. A sure thing, p = 1, gives 0, and a one-in-a-\n"
                 "hundred event gives 4.6. The log is natural, so the unit is the nat; with log base 2 it would be the bit.", wait=0)
        self.play(FadeIn(ch.group), FadeIn(xl), FadeIn(yl), run_time=1)
        self.play(Create(curve), run_time=1.5)
        self.wait(6)
        self.say(f"Under P, box is the least surprising word, at {-np.log(P_TRUE[0]):.2f} nats, "
                 f"and cable the most, at {-np.log(P_TRUE[4]):.2f}.", wait=0)
        self.play(LaggedStart(*[FadeIn(dt, scale=1.5) for dt in dots], lag_ratio=0.2), FadeIn(table, lag_ratio=0.15), run_time=2)
        self.wait(4)
        self.clear_stage()

        rects, labels = area_rects(P_TRUE, -np.log(P_TRUE), P_COL, 0.35, 2)
        form = VGroup(M("H(P) = −∑<sub>y</sub> P(y) log P(y)", 28), M(f"= {d['H']:.3f} nats", 28, P_COL))
        form.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(np.array([PANEL_X + 0.4, 1.2, 0]), aligned_edge=LEFT)
        key = M("width P(y), height −log P(y)", 22, GREY_B).next_to(form, DOWN, buff=0.45, aligned_edge=LEFT)
        self.say("Entropy H(P) is the average surprise when the words come from P: each word's surprise, weighted by how\n"
                 "often it happens. Drawn as rectangles of width P(y) and height −log P(y), it is their total area.", wait=0)
        self.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in rects], lag_ratio=0.15), FadeIn(labels), FadeIn(key), run_time=2)
        self.play(FadeIn(form, shift=0.1 * UP))
        self.wait(6)
        self.say(f"Here H(P) = {d['H']:.3f} nats. It measures how unpredictable the next word really is, and no model\n"
                 "can be less surprised than this on average.")
        self.end_section()

    def kl_gap(self):
        """
        Description: Regrows the rectangles to the model's surprise, shows the grown and shrunk areas, and defines KL.
        Inputs: none
        Outputs: none
        """
        self.mark("kl_gap")
        d = self.d
        self.heading("Cross-entropy, and the gap that is KL")
        sP, sQ = -np.log(P_TRUE), -np.log(Q_MODEL)
        p_rects, labels = area_rects(P_TRUE, sP, P_COL, 0.35, 2)
        q_rects, _ = area_rects(P_TRUE, sQ, Q_COL, 0.0, 3)
        start = VGroup(*[r.copy().set_fill(opacity=0).set_stroke(Q_COL, 3) for r in p_rects])
        diff = VGroup()
        for j in range(len(WORDS)):
            lo, hi = sorted((sP[j], sQ[j]))
            col = KL_COL if sQ[j] > sP[j] else GOOD
            diff.add(rect_at(j, lo, hi).set_stroke(width=0).set_fill(col, 0.55))
        self.add(p_rects, labels)
        self.play(FadeIn(p_rects), FadeIn(labels), run_time=0.8)
        ce = VGroup(M("H(P, Q) = −∑<sub>y</sub> P(y) log Q(y)", 26, Q_COL), M(f"= {d['CE']:.3f} nats", 26, Q_COL))
        ce.arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to(np.array([PANEL_X, 2.35, 0]), aligned_edge=LEFT)
        self.say("Now the words still come from P, but you are surprised according to the model Q. The rectangles keep\n"
                 "their widths P(y) and take the heights −log Q(y). Their total area is the cross-entropy H(P, Q).", wait=0)
        self.add(start)
        self.play(Transform(start, q_rects), run_time=2)
        self.play(FadeIn(ce, shift=0.1 * DOWN))
        self.wait(6)
        tally = VGroup(M(f"grown:   +{d['grown']:.3f}", 26, KL_COL), M(f"shrunk:  −{d['shrunk']:.3f}", 26, GOOD),
                       M(f"net:   {d['KL']:.3f} nats", 26))
        tally.arrange(DOWN, aligned_edge=LEFT, buff=0.14).next_to(ce, DOWN, buff=0.4, aligned_edge=LEFT)
        self.say("Where Q gave a word less probability than P, its rectangle grew; bolt grew most, because Q gave it\n"
                 "only 0.02. Where Q gave more, it shrank. Grown area minus shrunk area is the KL divergence.", wait=0)
        self.play(FadeIn(diff, lag_ratio=0.15), run_time=1.5)
        self.play(FadeIn(tally, lag_ratio=0.3), run_time=1.5)
        self.wait(7)
        defn = VGroup(M("KL(P ‖ Q) = H(P, Q) − H(P)", 26, KL_COL), M("= ∑<sub>y</sub> P(y) log(P(y) / Q(y))", 26, KL_COL))
        defn.arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(tally, DOWN, buff=0.4, aligned_edge=LEFT)
        self.say(f"So KL(P ‖ Q) = H(P, Q) − H(P) = {d['CE']:.3f} − {d['H']:.3f} = {d['KL']:.3f} nats: the extra surprise you\n"
                 "pay for predicting with Q when the words really come from P.", wait=0)
        self.play(FadeIn(defn, shift=0.1 * UP))
        self.wait(7)
        swap = M(f"swapped:  KL(Q ‖ P) = {d['KL_rev']:.3f}", 26, GREY_A)
        swap.next_to(defn, DOWN, buff=0.3, aligned_edge=LEFT)
        self.say(f"KL is never negative, and it is 0 only when Q equals P. It is not symmetric, though: KL(Q ‖ P) = {d['KL_rev']:.3f},\n"
                 "because then the average is weighted by Q's probabilities instead of P's.", wait=0)
        self.play(FadeIn(swap, shift=0.1 * UP))
        self.wait(8)
        self.end_section()

    def pretraining(self):
        """
        Description: Trains the model on sampled next words and shows the loss falling to the data's entropy as KL goes to 0.
        Inputs: none
        Outputs: none
        """
        self.mark("pretraining")
        d = self.d
        self.heading("Job 1: the pretraining loss")
        frame = word_bars([d["P_hat"]], [P_COL], x0=-5.9, dx=1.15, base=-1.3, scale=5.0, width=0.42, values=False, offsets=(-0.22,))
        ch = Chart((0, TRAIN_STEPS), (1.40, 1.75), 5.2, 3.0, np.array([3.75, -0.25, 0]), xticks=(0, 20, 40, 60),
                   yticks=(1.4, 1.5, 1.6, 1.7), tick_size=18)
        xl = VGroup(M("gradient-descent step", 22, GREY_B).next_to(ch.c2p(TRAIN_STEPS / 2, 1.40), DOWN, buff=0.45),
                    M("loss, nats", 20, GREY_B).move_to(ch.c2p(0, 1.75) + 0.25 * UP, aligned_edge=LEFT))
        floor = DashedLine(ch.c2p(0, d["H_hat"]), ch.c2p(TRAIN_STEPS, d["H_hat"]), dash_length=0.08).set_stroke(GREY_B, 2)
        floor_t = M(f"floor: the data's entropy, {d['H_hat']:.3f}", 20, GREY_B).next_to(floor, DOWN, buff=0.1).align_to(floor, RIGHT)
        key = VGroup(M("data's frequencies", 22, P_COL), M("model Q", 22, Q_COL)).arrange(RIGHT, buff=0.5)
        key.move_to(np.array([-3.6, 2.45, 0]))
        st = ValueTracker(0)
        k_of = lambda: int(round(st.get_value()))
        q_bars = always_redraw(lambda: bar_group(d["train_Q"][k_of()], Q_COL, x0=-5.9, dx=1.15, base=-1.3, scale=5.0,
                                                 width=0.42, offset=0.22))
        loss_curve = always_redraw(lambda: polyline([ch.c2p(k, d["train_loss"][k]) for k in range(k_of() + 1)]).set_stroke(Q_COL, 4))
        read = always_redraw(lambda: VGroup(
            cached("math", f"step {k_of()}", 24, GREY_B), cached("math", f"loss {d['train_loss'][k_of()]:.3f}", 24, Q_COL),
            cached("math", f"KL(data ‖ Q) {d['train_kl'][k_of()]:.3f}", 24, KL_COL)).arrange(RIGHT, buff=0.4).move_to(np.array([3.75, 1.85, 0])))
        self.say(f"Training never sees P itself, only examples: here {N_SAMPLES:,} synthetic sentences that continue this prompt.\n"
                 "Their next words, counted, give the data's frequencies: the amber bars.", wait=0)
        self.play(FadeIn(frame), FadeIn(key[0]), run_time=1.2)
        self.wait(5)
        self.say("On each example the loss is the model's surprise at the actual next word, −log Q(y): the negative\n"
                 "log-likelihood from the log-likelihood video. Averaged over the examples, it is a cross-entropy.", wait=0)
        self.add(q_bars)
        self.play(FadeIn(key[1]), FadeIn(ch.group), FadeIn(xl), FadeIn(floor), FadeIn(floor_t), run_time=1)
        self.add(loss_curve, read)
        self.wait(6)
        self.say("So the loss equals the data's entropy plus KL(data ‖ Q). The entropy is fixed by the data, so\n"
                 "training can only shrink the KL.")
        self.say(f"Gradient descent on the model's scores moves Q onto the data's frequencies. The loss falls from "
                 f"{d['train_loss'][0]:.3f}\nto the floor, {d['H_hat']:.3f} nats, and the KL falls to 0.", wait=0)
        self.play(st.animate.set_value(TRAIN_STEPS), run_time=7, rate_func=linear)
        self.wait(5)
        self.say("A real language model's loss has the same two parts: entropy that no model can remove, and KL\n"
                 "that training keeps shrinking.")
        self.end_section()

    def directions(self):
        """
        Description: Fits one bell curve to two humps by each KL direction, from the same start, and compares the fits.
        Inputs: none
        Outputs: none
        """
        self.mark("directions")
        d = self.d
        self.heading("The order matters: covering versus hiding")
        ch = Chart((-5, 5), (0, 0.72), 8.0, 3.6, np.array([-1.7, -0.2, 0]), xticks=(-4, -2, 0, 2, 4), yticks=(0, 0.25, 0.5), tick_size=18)
        p_line = ch.curve(lambda x: mixture(x, HUMP_W, HUMP_MU, HUMP_SD), -5, 5, n=400)
        p_fill = p_line.copy().add_line_to(ch.c2p(5, 0)).add_line_to(ch.c2p(-5, 0)).add_line_to(ch.c2p(-5, mixture(-5, HUMP_W, HUMP_MU, HUMP_SD)))
        p_fill.set_stroke(width=0).set_fill(P_COL, 0.25)
        p_line.set_stroke(P_COL, 3)
        p_lab = M("P: two humps", 24, P_COL).next_to(ch.c2p(-2, 0.66), UP, buff=0.05)
        start = DashedVMobject(ch.curve(lambda x: gauss(x, *FIT_START), -5, 5, n=300), num_dashes=40).set_stroke(GREY_B, 2)
        start_t = M(f"start: μ = {num(FIT_START[0], '.1f')}, σ = {FIT_START[1]:.1f}", 22, GREY_B).move_to(np.array([4.9, 1.6, 0]))
        fwd_t, rev_t = ValueTracker(0), ValueTracker(0)
        fwd_path, rev_path = d["fwd_path"], d["rev_path"]
        at = lambda path, t: path[int(round(t * (len(path) - 1)))]
        fwd_curve = always_redraw(lambda: ch.curve(lambda x: gauss(x, *at(fwd_path, fwd_t.get_value())), -5, 5, n=300).set_stroke(Q_COL, 4))
        rev_curve = always_redraw(lambda: ch.curve(lambda x: gauss(x, *at(rev_path, rev_t.get_value())), -5, 5, n=300).set_stroke(KL_COL, 4))
        fwd_read = always_redraw(lambda: VGroup(
            cached("math", "minimize KL(P ‖ Q)", 24, Q_COL),
            cached("math", f"μ = {num(at(fwd_path, fwd_t.get_value())[0], '.2f')},  σ = {at(fwd_path, fwd_t.get_value())[1]:.2f}", 24, Q_COL))
            .arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to(np.array([3.3, 0.8, 0]), aligned_edge=LEFT))
        rev_read = always_redraw(lambda: VGroup(
            cached("math", "minimize KL(Q ‖ P)", 24, KL_COL),
            cached("math", f"μ = {num(at(rev_path, rev_t.get_value())[0], '.2f')},  σ = {at(rev_path, rev_t.get_value())[1]:.2f}", 24, KL_COL))
            .arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to(np.array([3.3, -0.4, 0]), aligned_edge=LEFT))
        self.say("Fit a single bell curve Q, with centre μ and width σ, to a distribution P with two humps. Gradient\n"
                 "descent can minimize either KL(P ‖ Q) or KL(Q ‖ P), and both runs start from the same curve.", wait=0)
        self.play(FadeIn(ch.group), FadeIn(p_fill), Create(p_line), FadeIn(p_lab), run_time=1.5)
        self.play(Create(start), FadeIn(start_t), run_time=1)
        self.wait(5)
        self.add(fwd_curve, fwd_read)
        self.say("KL(P ‖ Q) averages over P, so it charges Q wherever P has probability that Q lacks. Its best fit\n"
                 "spreads out to cover both humps, even though that puts probability in the empty valley.", wait=0)
        self.play(fwd_t.animate.set_value(1), run_time=4, rate_func=smooth)
        self.wait(6)
        self.add(rev_curve, rev_read)
        self.say("KL(Q ‖ P) averages over Q, so it charges Q for probability where P has little. Its best fit hides inside\n"
                 f"the bigger hump. A fit inside the smaller hump is a worse local optimum: {d['rev_other']:.2f} nats against {d['rev_main']:.2f}.", wait=0)
        self.play(rev_t.animate.set_value(1), run_time=4, rate_func=smooth)
        self.wait(7)
        self.say("Pretraining uses the covering direction: the model must give some probability to every word the data\n"
                 "uses. The next two jobs use the hiding direction.")
        self.end_section()

    def elbo_gap(self):
        """
        Description: Shows that the gap between log p(x) and the ELBO is a KL, closed by the E-step, and the VAE's form.
        Inputs: none
        Outputs: none
        """
        self.mark("elbo_gap")
        d = self.d
        self.heading("Job 2: the gap under the ELBO")
        top = M("log p(x)  =  ELBO(q)  +  KL(q ‖ p(z | x))", 32).move_to(2.6 * UP)
        left = Chart((-5, 6), (0, 0.36), 5.4, 2.8, np.array([-3.65, -0.15, 0]), xticks=(-4, -2, 0, 2, 4), yticks=(0, 0.2), tick_size=16)
        c1 = left.curve(lambda x: EM_W[0] * gauss(x, EM_MU[0], EM_SD[0]), -5, 6, n=300).set_stroke(Q_COL, 3)
        c2 = left.curve(lambda x: EM_W[1] * gauss(x, EM_MU[1], EM_SD[1]), -5, 6, n=300).set_stroke(P_COL, 3)
        xline = DashedLine(left.c2p(X_POINT, 0), left.c2p(X_POINT, 0.36), dash_length=0.06).set_stroke(WHITE, 2)
        xlab = M("x = 0", 22).next_to(left.c2p(X_POINT, 0.36), UP, buff=0.08)
        hits = VGroup(Dot(left.c2p(X_POINT, d["joint"][0]), radius=0.07, color=Q_COL), Dot(left.c2p(X_POINT, d["joint"][1]), radius=0.07, color=P_COL))
        post = VGroup(M(f"p(z = left | x) = {d['post'][0]:.2f}", 22, Q_COL), M(f"p(z = right | x) = {d['post'][1]:.2f}", 22, P_COL))
        post.arrange(DOWN, aligned_edge=LEFT, buff=0.08).next_to(left.c2p(1.6, 0.33), DOWN, buff=0.0, aligned_edge=LEFT)
        right = Chart((0, 1), (-3.7, -2.7), 5.0, 2.8, np.array([3.6, -0.15, 0]), xticks=(0, 0.5, 1), yticks=(-3.5, -3, -2.75),
                      tick_size=16)
        rxl = M("q: the guessed share for the left curve", 20, GREY_B).next_to(right.c2p(0.5, -3.7), DOWN, buff=0.42)
        lp = DashedLine(right.c2p(0, d["log_px"]), right.c2p(1, d["log_px"]), dash_length=0.08).set_stroke(WHITE, 2)
        lp_t = M(f"log p(x) = {num(d['log_px'], '.3f')}", 20).next_to(right.c2p(1, d["log_px"]), UP, buff=0.1).align_to(lp, RIGHT)
        e_curve = right.curve(lambda q: elbo(q), 0.005, 0.995, n=300).set_stroke(GOOD, 4)
        e_t = M("ELBO(q)", 20, GOOD).next_to(right.c2p(0.08, elbo(0.08)), RIGHT, buff=0.12)
        qt = ValueTracker(0.1)
        gap = always_redraw(lambda: VGroup(
            Line(right.c2p(qt.get_value(), elbo(qt.get_value())), right.c2p(qt.get_value(), d["log_px"])).set_stroke(KL_COL, 5),
            Dot(right.c2p(qt.get_value(), elbo(qt.get_value())), radius=0.07, color=GOOD)))
        read = always_redraw(lambda: VGroup(
            cached("math", f"q = {qt.get_value():.2f}", 24, GREY_A),
            cached("math", f"ELBO = {num(elbo(qt.get_value()), '.3f')}", 24, GOOD),
            cached("math", f"KL = {d['log_px'] - elbo(qt.get_value()):.3f}", 24, KL_COL)).arrange(RIGHT, buff=0.45).move_to(np.array([0.6, 1.9, 0])))
        self.say("Back to the two-curve mixture from the EM video, with its true settings. For the point x = 0, the hidden z\n"
                 f"says which curve made it. Its posterior p(z | x) gives the left curve {d['post'][0]:.2f} and the right {d['post'][1]:.2f}.", wait=0)
        self.play(FadeIn(left.group), Create(c1), Create(c2), run_time=1.5)
        self.play(Create(xline), FadeIn(xlab), FadeIn(hits), FadeIn(post), run_time=1.2)
        self.wait(6)
        self.say("Take any guess q for that share. The ELBO from the EM video is a lower bound on log p(x), and the gap\n"
                 "between them is exactly KL(q ‖ p(z | x)), the KL from the guess to the true posterior.", wait=0)
        self.play(FadeIn(top, shift=0.1 * DOWN), FadeIn(right.group), FadeIn(rxl), Create(lp), FadeIn(lp_t), run_time=1.2)
        self.play(Create(e_curve), FadeIn(e_t), run_time=1.2)
        self.add(gap, read)
        self.wait(4)
        self.play(qt.animate.set_value(0.9), run_time=4, rate_func=smooth)
        self.play(qt.animate.set_value(d["post"][0]), run_time=2.5, rate_func=smooth)
        self.say("The gap is 0 only when q equals the posterior, so that is where the bound touches. That is what EM's\n"
                 "E-step does, and why the staircase in the EM video touched the curve at every step.")
        vae = M("ELBO(q)  =  E<sub>q</sub>[log p(x | z)]  −  KL(q ‖ p(z))", 30, GOOD).move_to(2.6 * UP)
        self.play(FadeOut(top, shift=0.1 * UP), FadeIn(vae, shift=0.1 * UP))
        self.say("Rearranged, the same ELBO is a fit term minus a second KL, from q to the prior p(z), here the curve weights\n"
                 "0.6 and 0.4. That is the line in the VAE frame: the encoder outputs q, and this KL keeps q near the prior.")
        self.wait(2)
        self.end_section()

    def rlhf(self):
        """
        Description: Sweeps β in the KL-penalized RLHF objective and shows the tuned model collapsing onto one response.
        Inputs: none
        Outputs: none
        """
        self.mark("rlhf")
        d = self.d
        self.heading("Job 3: the leash in RLHF")
        prompt = M("“Why did the robot stop?”", 26, GREY_A).move_to(np.array([-6.4, 2.05, 0]), aligned_edge=LEFT)
        names = VGroup()
        for j in range(len(PI_REF)):
            x = -5.9 + 1.15 * j
            names.add(VGroup(M(f"y<sub>{j + 1}</sub>", 24).move_to(np.array([x, -1.75, 0])),
                             M(f"R = {REWARD[j]:.1f}", 20, GOOD).move_to(np.array([x, -2.15, 0]))))
        axis = Line(np.array([-6.5, -1.5, 0]), np.array([-0.8, -1.5, 0])).set_stroke(GREY_B, 2)
        ref_bars = bar_group(PI_REF, REF_COL, x0=-5.9, dx=1.15, base=-1.5, scale=RL_SCALE, width=0.42, offset=-0.22)
        key = VGroup(M("π<sub>ref</sub>: before tuning", 22, REF_COL), M("π: tuned", 22, Q_COL)).arrange(RIGHT, buff=0.4)
        key.move_to(np.array([-6.4, 1.45, 0]), aligned_edge=LEFT)
        obj = M("maximize   E<sub>π</sub>[R(y)]  −  β · KL(π ‖ π<sub>ref</sub>)", 30).move_to(np.array([0, 2.75, 0]))
        best = M("best π(y)  =  π<sub>ref</sub>(y) · e<sup>R(y)/β</sup> / Z", 30, Q_COL).move_to(np.array([3.4, 2.05, 0]))
        ch = Chart((0, 3.2), (0.9, 3.1), 4.6, 2.6, np.array([4.15, -0.55, 0]), xticks=(0, 1, 2, 3), yticks=(1, 2, 3), tick_size=16)
        cxl = M("KL(π ‖ π<sub>ref</sub>)", 20, KL_COL).next_to(ch.c2p(1.6, 0.9), DOWN, buff=0.42)
        cyl = M("expected reward", 20, GOOD).move_to(ch.c2p(0, 3.1) + 0.22 * UP, aligned_edge=LEFT)
        bt = ValueTracker(BETA_LOGS[0])
        k_of = lambda: int(np.clip(round((BETA_LOGS[0] - bt.get_value()) / (BETA_LOGS[0] - BETA_LOGS[-1]) * (len(BETA_LOGS) - 1)), 0, len(BETA_LOGS) - 1))
        pi_bars = always_redraw(lambda: bar_group(d["rl_pi"][k_of()], Q_COL, x0=-5.9, dx=1.15, base=-1.5, scale=RL_SCALE,
                                                  width=0.42, offset=0.22))
        trace = always_redraw(lambda: VGroup(
            polyline([ch.c2p(d["rl_kl"][k], d["rl_er"][k]) for k in range(k_of() + 1)]).set_stroke(GOOD, 3),
            Dot(ch.c2p(d["rl_kl"][k_of()], d["rl_er"][k_of()]), radius=0.07, color=WHITE)))
        read = always_redraw(lambda: VGroup(
            cached("math", f"β = {10 ** BETA_LOGS[k_of()]:.3g}", 24, WHITE),
            cached("math", f"expected reward {d['rl_er'][k_of()]:.3f}", 24, GOOD),
            cached("math", f"KL {d['rl_kl'][k_of()]:.3f}", 24, KL_COL)).arrange(RIGHT, buff=0.4).move_to(np.array([3.6, 1.5, 0])))
        self.say("After pretraining, RLHF tunes the model toward responses that a reward model scores highly. π<sub>ref</sub> is the\n"
                 "model before tuning, π the model being tuned, and R(y) the reward for response y. The numbers are synthetic.", wait=0)
        self.play(FadeIn(prompt), Create(axis), FadeIn(names), FadeIn(ref_bars, lag_ratio=0.1), FadeIn(key[0]), run_time=1.5)
        self.wait(6)
        self.say("The objective takes the expected reward and subtracts β times KL(π ‖ π<sub>ref</sub>). β plays the part λ played\n"
                 "in the regularization video: it sets how much each move away from π<sub>ref</sub> costs.", wait=0)
        self.play(FadeIn(obj, shift=0.1 * DOWN))
        self.wait(7)
        self.say("This objective has an exact best answer: reweight π<sub>ref</sub> by e<sup>R(y)/β</sup>, then divide by Z, the sum that\n"
                 "makes the probabilities add to 1. DPO (Rafailov and colleagues, 2023) is built on this same formula.", wait=0)
        self.play(FadeIn(best, shift=0.1 * DOWN))
        self.add(pi_bars, trace, read)
        self.play(FadeIn(key[1]), FadeIn(ch.group), FadeIn(cxl), FadeIn(cyl), run_time=1)
        self.wait(7)
        self.say(f"At β = 10 the tuned model barely moves: the KL is {d['rl_kl'][0]:.3f} nats. As β shrinks, π piles onto y<sub>5</sub>, which the\n"
                 "reward model scores 3.0 but π<sub>ref</sub> almost never writes.", wait=0)
        self.play(bt.animate.set_value(BETA_LOGS[-1]), run_time=8, rate_func=linear)
        self.wait(5)
        self.say(f"At β = 0.1 nearly all probability sits on y<sub>5</sub>, and the KL reaches {d['rl_kl'][-1]:.2f} nats, the most possible here.\n"
                 "Chasing one score that hard is how a policy over-optimizes an imperfect reward model.")
        self.say("InstructGPT (Ouyang and colleagues, 2022) added a per-token KL penalty from the supervised model to\n"
                 "prevent exactly this. It is the hiding direction: π can't put probability where π<sub>ref</sub> puts none.")
        self.end_section()

    def summary(self):
        """
        Description: Shows the closing table of the three jobs, each with its KL and direction.
        Inputs: none
        Outputs: none
        """
        self.mark("summary")
        self.heading("One formula, three jobs")
        rows = (("job", "the KL", "direction", "what it does", GREY_B),
                ("pretraining loss", "entropy + KL(data ‖ Q)", "covering", "covers every word the data uses", P_COL),
                ("EM and VAEs", "gap = KL(q ‖ p(z | x))", "hiding", "the bound touches at the posterior", GOOD),
                ("RLHF", "β · KL(π ‖ π<sub>ref</sub>)", "hiding", "π stays near what π<sub>ref</sub> writes", Q_COL))
        table = VGroup()
        for i, (a, b, c, e, col) in enumerate(rows):
            y = 1.9 - 0.85 * i
            size = 24 if i else 22
            cells = VGroup()
            for text, x, wmax in ((a, -6.6, 2.4), (b, -4.0, 4.0), (c, 0.3, 1.5), (e, 2.0, 4.9)):
                cell = M(text, size, col)
                if cell.width > wmax:
                    cell.scale_to_fit_width(wmax)
                cells.add(cell.move_to(np.array([x, y, 0]), aligned_edge=LEFT))
            table.add(cells)
        rule = Line(np.array([-6.6, 1.5, 0]), np.array([6.9, 1.5, 0])).set_stroke(GREY_D, 1.5)
        formula = M("KL(P ‖ Q)  =  ∑<sub>y</sub> P(y) log(P(y) / Q(y))", 34, KL_COL).move_to(2.75 * UP)
        self.say("One formula does all three jobs. The distribution on the left of the bars decides whose probabilities\n"
                 "weight the average, and that decides whether the fit spreads out or stays put.", wait=0)
        self.play(FadeIn(formula), FadeIn(table[0]), Create(rule), run_time=1)
        self.play(LaggedStart(*[FadeIn(r, shift=0.1 * UP) for r in table[1:]], lag_ratio=0.4), run_time=2.5)
        self.wait(8)
        self.say("Solomon Kullback and Richard Leibler introduced this quantity in 1951, as the mean information\n"
                 "per observation for telling two hypotheses apart. In their paper, “divergence” meant the two-way sum\n"
                 "KL(P ‖ Q) + KL(Q ‖ P), which Harold Jeffreys had used in 1948.")
        self.wait(2)
        self.end_section()

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
        Outputs: the polyline, or an empty one when fewer than two points are in range.
        Constraints: each curve drawn here leaves the y-range at most once at each end, so dropping points
        never joins two separate pieces.
        """
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
    d["H"], d["CE"] = entropy(P_TRUE), cross_entropy(P_TRUE, Q_MODEL)
    d["KL"], d["KL_rev"] = kl(P_TRUE, Q_MODEL), kl(Q_MODEL, P_TRUE)
    assert abs(d["CE"] - d["H"] - d["KL"]) < 1e-12, "cross-entropy must be entropy plus KL"
    shown = round(round(d["CE"], 3) - round(d["H"], 3), 3)
    assert f"{shown:.3f}" == f"{d['KL']:.3f}", "the shown subtraction must give the shown KL"
    sP, sQ = -np.log(P_TRUE), -np.log(Q_MODEL)
    d["grown"] = float((P_TRUE * np.maximum(sQ - sP, 0)).sum())
    d["shrunk"] = float((P_TRUE * np.maximum(sP - sQ, 0)).sum())
    assert f"{round(round(d['grown'], 3) - round(d['shrunk'], 3), 3):.3f}" == f"{d['KL']:.3f}", "grown minus shrunk must show as KL"
    assert int(np.argmax(sQ - sP)) == WORDS.index("bolt"), "bolt's rectangle must grow the most"
    assert d["KL"] - d["KL_rev"] > 0.05, "the two directions must differ visibly"
    assert int(np.argmin(sP)) == 0 and int(np.argmax(sP)) == len(WORDS) - 1, "box must be least surprising and cable most"

    rng = np.random.default_rng(SAMPLE_SEED)
    ys = rng.choice(len(WORDS), size=N_SAMPLES, p=P_TRUE)
    d["P_hat"] = np.bincount(ys, minlength=len(WORDS)) / N_SAMPLES
    d["H_hat"] = entropy(d["P_hat"])
    d["train_Q"] = train_path(np.log(Q_MODEL), d["P_hat"])
    d["train_loss"] = np.array([cross_entropy(d["P_hat"], q) for q in d["train_Q"]])
    d["train_kl"] = np.array([kl(d["P_hat"], q) for q in d["train_Q"]])
    assert np.all(np.diff(d["train_loss"]) < 0), "the training loss must fall at every step"
    assert d["train_kl"][-1] < 5e-4 and f"{d['train_kl'][-1]:.3f}" == "0.000", "the KL must reach 0 at the shown precision"
    assert np.allclose(d["train_loss"] - d["H_hat"], d["train_kl"]), "loss must equal the data's entropy plus KL"
    assert 1.40 < d["H_hat"] < d["train_loss"][0] < 1.75, "the loss must fit the chart"

    p_h = mixture(XS, HUMP_W, HUMP_MU, HUMP_SD)
    d["fwd_path"] = fit_path(kl_fwd, p_h, FIT_START)
    d["rev_path"] = fit_path(kl_rev, p_h, FIT_START)
    mm = float((HUMP_W * HUMP_MU).sum())
    ss = float(np.sqrt((HUMP_W * (HUMP_SD ** 2 + HUMP_MU ** 2)).sum() - mm ** 2))
    assert np.allclose(d["fwd_path"][-1], (mm, ss), atol=1e-3), "the forward fit must match P's mean and spread"
    rev_end = d["rev_path"][-1]
    assert abs(rev_end[0] - HUMP_MU[0]) < 0.05 and rev_end[1] < 0.7, "the reverse fit must sit inside the bigger hump"
    other = fit_path(kl_rev, p_h, (1.0, 1.0))[-1]
    assert abs(other[0] - HUMP_MU[1]) < 0.05, "from the right, the reverse fit must settle in the smaller hump"
    d["rev_main"], d["rev_other"] = kl_rev(p_h, *rev_end), kl_rev(p_h, *other)
    assert d["rev_main"] < d["rev_other"], "the bigger hump must be the better reverse fit"
    peaks = 1 / (np.concatenate([d["fwd_path"][:, 1], d["rev_path"][:, 1]]) * np.sqrt(2 * np.pi))
    assert max(mixture(np.linspace(-5, 5, 2001), HUMP_W, HUMP_MU, HUMP_SD)) < 0.72 and peaks.max() < 0.72, "every fit must stay on the chart"

    d["joint"] = EM_W * gauss(X_POINT, EM_MU, EM_SD)
    d["log_px"] = float(np.log(d["joint"].sum()))
    d["post"] = d["joint"] / d["joint"].sum()
    assert f"{d['post'][0]:.2f}" == "0.56" and f"{d['post'][1]:.2f}" == "0.44", "the shown posterior must add to 1"
    qs = np.linspace(0.005, 0.995, 397)
    gaps = np.array([d["log_px"] - elbo(q) for q in qs])
    kls = np.array([kl(np.array([q, 1 - q]), d["post"]) for q in qs])
    assert np.abs(gaps - kls).max() < 1e-12, "the ELBO gap must be exactly KL(q ‖ posterior)"
    assert abs(qs[np.argmin(gaps)] - d["post"][0]) < 0.003, "the bound must touch at the posterior"
    for q in (0.2, d["post"][0], 0.7):
        qq = np.array([q, 1 - q])
        vae = float((qq * np.log(gauss(X_POINT, EM_MU, EM_SD))).sum() - kl(qq, EM_W))
        assert abs(vae - elbo(q)) < 1e-12, "the VAE form must equal the ELBO"
    assert -3.7 < min(elbo(0.005), elbo(0.995)) and d["log_px"] < -2.7

    d["rl_pi"] = np.array([rlhf_policy(10 ** b) for b in BETA_LOGS])
    d["rl_er"] = d["rl_pi"] @ REWARD
    d["rl_kl"] = np.array([kl(p, PI_REF) for p in d["rl_pi"]])
    assert np.all(np.diff(d["rl_kl"]) > 0) and np.all(np.diff(d["rl_er"]) > 0), "smaller β must mean more KL and more reward"
    probe = np.random.default_rng(3).dirichlet(np.ones(len(PI_REF)), 20000)
    for b in (3.0, 1.0, 0.3):
        best = rlhf_policy(b)
        j_best = best @ REWARD - b * kl(best, PI_REF)
        j_probe = probe @ REWARD - b * (probe * np.log(probe / PI_REF)).sum(1)
        assert j_probe.max() <= j_best + 1e-12, "the closed form must beat every random policy"
        log_z = float(np.log((PI_REF * np.exp(REWARD / b)).sum()))
        kl_star = (probe * np.log(probe / best)).sum(1)
        assert np.abs(j_probe - (b * log_z - b * kl_star)).max() < 1e-9, "the objective must equal β log Z − β KL(π ‖ π*)"
    assert f"{d['rl_kl'][-1]:.2f}" == f"{-np.log(PI_REF[4]):.2f}" == "3.00", "at β = 0.1 the KL must show its maximum, 3.00"
    assert d["rl_pi"][-1][4] > 0.99 and d["rl_kl"][0] < 0.01
    print("PRECOMPUTE", f"H {d['H']:.3f} CE {d['CE']:.3f} KL {d['KL']:.3f} rev {d['KL_rev']:.3f}",
          f"train {d['train_loss'][0]:.3f}->{d['train_loss'][-1]:.3f} floor {d['H_hat']:.3f}",
          f"fwd {d['fwd_path'][-1].round(3)} rev {rev_end.round(3)} {d['rev_main']:.3f}/{d['rev_other']:.3f}",
          f"post {d['post'].round(3)} logp {d['log_px']:.3f}", f"rl {d['rl_kl'][0]:.3f}->{d['rl_kl'][-1]:.3f}")
    return d


def entropy(p):
    """
    Description: Entropy, the average surprise of outcomes drawn from p, in nats.
    Inputs:
    - p: a probability vector.
    Outputs: −∑ p log p.
    """
    return float(-(p * np.log(p)).sum())


def cross_entropy(p, q):
    """
    Description: Cross-entropy, the average surprise under q of outcomes drawn from p, in nats.
    Inputs:
    - p, q: probability vectors.
    Outputs: −∑ p log q.
    """
    return float(-(p * np.log(q)).sum())


def kl(p, q):
    """
    Description: The KL divergence from q to p: the extra surprise of predicting with q when outcomes come from p.
    Inputs:
    - p, q: probability vectors.
    Outputs: ∑ p log(p / q), in nats.
    """
    return float((p * np.log(p / q)).sum())


def train_path(z0, target):
    """
    Description: Runs gradient descent on a softmax model's scores to minimize the cross-entropy with the data.
    Inputs:
    - z0: the starting scores, so the starting model is softmax(z0).
    - target: the data's word frequencies.
    Outputs: the model's probabilities after each step, starting with step 0, shape (TRAIN_STEPS + 1, len(z0)).
    Constraints: the gradient of the cross-entropy with respect to the scores is softmax(z) − target.
    """
    z = z0.copy()
    out = []
    for _ in range(TRAIN_STEPS + 1):
        q = np.exp(z - z.max())
        q /= q.sum()
        out.append(q)
        z = z - TRAIN_LR * (q - target)
    return np.array(out)


def gauss(x, mu, sd):
    """
    Description: The bell-curve density with centre mu and width sd.
    Inputs:
    - x: the point or points.
    - mu, sd: the centre and width.
    Outputs: the density.
    """
    return np.exp(-(x - mu) ** 2 / (2 * sd * sd)) / (sd * np.sqrt(2 * np.pi))


def mixture(x, w, mu, sd):
    """
    Description: The density of a weighted sum of bell curves.
    Inputs:
    - x: the point or points.
    - w, mu, sd: the weights, centres and widths.
    Outputs: ∑ w_k · gauss(x, mu_k, sd_k).
    """
    x = np.asarray(x, dtype=float)
    return (w[:, None] * gauss(x.reshape(1, -1), mu[:, None], sd[:, None])).sum(0).reshape(x.shape)


def kl_fwd(p, mu, sd):
    """
    Description: KL(P ‖ Q) for P on the grid XS and Q a bell curve, by a Riemann sum.
    Inputs:
    - p: P's density on XS.
    - mu, sd: Q's centre and width.
    Outputs: the divergence in nats.
    """
    q = gauss(XS, mu, sd)
    return float((p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum() * (XS[1] - XS[0]))


def kl_rev(p, mu, sd):
    """
    Description: KL(Q ‖ P) for Q a bell curve and P on the grid XS, by a Riemann sum.
    Inputs:
    - p: P's density on XS.
    - mu, sd: Q's centre and width.
    Outputs: the divergence in nats.
    """
    q = gauss(XS, mu, sd)
    return float((q * (np.log(q + 1e-300) - np.log(p + 1e-300))).sum() * (XS[1] - XS[0]))


def fit_path(f, p, start):
    """
    Description: Runs gradient descent on (centre, log width) of a bell curve to minimize a KL with central differences.
    Inputs:
    - f: kl_fwd or kl_rev.
    - p: P's density on XS.
    - start: the starting (centre, width).
    Outputs: the (centre, width) after each step, starting with start, shape (FIT_STEPS + 1, 2).
    Constraints: the width is optimized through its log, so it stays positive.
    """
    m, ls = start[0], np.log(start[1])
    out = [(m, np.exp(ls))]
    e = 1e-5
    for _ in range(FIT_STEPS):
        gm = (f(p, m + e, np.exp(ls)) - f(p, m - e, np.exp(ls))) / (2 * e)
        gl = (f(p, m, np.exp(ls + e)) - f(p, m, np.exp(ls - e))) / (2 * e)
        m, ls = m - FIT_LR * gm, ls - FIT_LR * gl
        out.append((m, np.exp(ls)))
    return np.array(out)


def elbo(q):
    """
    Description: The ELBO of the EM mixture at X_POINT for the guess q of the left curve's share.
    Inputs:
    - q: the guessed share for the left curve, strictly between 0 and 1.
    Outputs: ∑ q_z log(p(z) p(x | z) / q_z).
    """
    qq = np.array([q, 1 - q])
    return float((qq * np.log(EM_W * gauss(X_POINT, EM_MU, EM_SD) / qq)).sum())


def rlhf_policy(beta):
    """
    Description: The best policy for expected reward minus β · KL(π ‖ π_ref): π_ref reweighted by e^(R/β).
    Inputs:
    - beta: the KL penalty's strength.
    Outputs: the policy's probabilities over the five responses.
    """
    a = np.log(PI_REF) + REWARD / beta
    w = np.exp(a - a.max())
    return w / w.sum()


def word_bars(series, colours, x0, dx, base, scale, width, values, offsets=(-0.3, 0.3)):
    """
    Description: Builds a bar chart over the five words, one bar per series per word.
    Inputs:
    - series: the probability vectors to draw.
    - colours: one colour per series.
    - x0, dx: the first word's screen x and the spacing.
    - base, scale: the screen y of probability 0 and the screen height per unit of probability.
    - width: each bar's width.
    - values: whether to print each value above its bar.
    - offsets: each series' horizontal offset from its word.
    Outputs: VGroup(axis and word labels, then one group of bars per series).
    """
    xs = [x0 + dx * j for j in range(len(WORDS))]
    frame = VGroup(Line(np.array([xs[0] - 0.65, base, 0]), np.array([xs[-1] + 0.65, base, 0])).set_stroke(GREY_B, 2),
                   label_row(WORDS, xs, base - 0.3, 24, (WHITE,) * len(WORDS)))
    out = VGroup(frame)
    for vals, col, off in zip(series, colours, offsets):
        out.add(bar_group(vals, col, x0, dx, base, scale, width, off, values))
    return out


def bar_group(vals, colour, x0, dx, base, scale, width, offset, values=False):
    """
    Description: Builds one series of bars, one per position.
    Inputs:
    - vals: the probabilities.
    - colour: the bars' colour.
    - x0, dx, base, scale, width: as in word_bars.
    - offset: the series' horizontal offset from each position.
    - values: whether to print each value above its bar.
    Outputs: the bars, each followed by its value when values is true.
    """
    g = VGroup()
    for j, v in enumerate(vals):
        h = max(v * scale, 1e-3)
        bar = Rectangle(width=width, height=h).set_stroke(width=0).set_fill(colour, 0.85)
        bar.move_to(np.array([x0 + dx * j + offset, base + h / 2, 0]))
        g.add(bar)
        if values:
            g.add(cached("mono", f"{v:.2f}", 18, colour).next_to(bar, UP, buff=0.08))
    return g


def label_row(texts, xs, y, size, colours, align="center"):
    """
    Description: Builds several labels from one line of text, so they share a baseline, then moves each to its own x.
    Inputs:
    - texts: the labels, as plain text.
    - xs: each label's screen x: its centre, or its left edge when align is "left".
    - y: the screen y of the line's middle.
    - size: the font size.
    - colours: one colour per label.
    - align: "center" or "left".
    Outputs: a VGroup with one glyph group per label.
    Constraints: Pango drops spaces from the glyphs, so each label owns the next len(label without spaces) glyphs.
    A leading "(" sets the line's height and is left out, so rows of different words keep the same baseline.
    """
    line = M("(" + "".join(" " + t for t in texts), size)
    counts = [len(t.replace(" ", "")) for t in texts]
    assert len(line.submobjects) == 1 + sum(counts), "each label must map onto its own glyphs"
    line.shift((y - line.submobjects[0].get_center()[1]) * UP)
    out, k = VGroup(), 1
    for x, colour, n in zip(xs, colours, counts):
        g = VGroup(*line.submobjects[k:k + n]).set_color(colour)
        k += n
        g.shift((x - (g.get_left()[0] if align == "left" else g.get_center()[0])) * RIGHT)
        out.add(g)
    return out


def area_rects(widths, heights, colour, fill, stroke):
    """
    Description: Builds rectangles side by side with width P(y) and height in nats, whose total area is an average.
    Inputs:
    - widths: the probabilities that set each rectangle's width.
    - heights: each rectangle's height in nats.
    - colour, fill, stroke: the colour, fill opacity and stroke width.
    Outputs: (rectangles, word labels under them).
    """
    rects = VGroup(*[rect_at(j, 0, heights[j]).set_fill(colour, fill).set_stroke(colour, stroke) for j in range(len(widths))])
    labels = label_row(WORDS, [r.get_center()[0] for r in rects], RECT_BASE - 0.25, 18, (GREY_A,) * len(WORDS))
    return rects, labels


def rect_at(j, lo, hi):
    """
    Description: Builds the part of word j's column in the area picture between two heights.
    Inputs:
    - j: the word's index; its column's width is P_TRUE[j].
    - lo, hi: the bottom and top, in nats.
    Outputs: the rectangle.
    """
    left = RECT_LEFT + RECT_WIDTH * P_TRUE[:j].sum()
    w = RECT_WIDTH * P_TRUE[j]
    r = Rectangle(width=w, height=max((hi - lo) * RECT_SCALE, 1e-3))
    return r.move_to(np.array([left + w / 2, RECT_BASE + (lo + hi) / 2 * RECT_SCALE, 0]))


def polyline(pts):
    """
    Description: Joins screen points with straight segments.
    Inputs:
    - pts: the points, in order.
    Outputs: the polyline, or an empty mobject when there are fewer than two points.
    """
    return VMobject().set_points_as_corners(pts) if len(pts) > 1 else VMobject()


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
