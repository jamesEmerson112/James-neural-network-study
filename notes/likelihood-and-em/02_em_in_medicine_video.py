"""Renders a short Manim video on how expectation-maximization labels MRI voxels and where else medicine uses it.

Contents:
- EMInMedicine: the scene, with one method per beat of the video
- run_em: fits the three-tissue Gaussian mixture and records the iterations shown
- em_step, loglik, comp_pdf: one EM iteration, the mean log-likelihood and one tissue's weighted density
- sx, sy: data-to-screen coordinates
- fonts, colours, the seeded synthetic dataset and the layout constants
"""
import numpy as np
from manim import *

SERIF = "Cambria"
"""Description: Serif font for every caption and title."""
FLUID = "#58C4DD"
"""Description: Colour of the fluid curve (dark voxels)."""
GRAY_M = "#B48EDE"
"""Description: Colour of the gray-matter curve (middle voxels)."""
WHITE_M = "#FFD166"
"""Description: Colour of the white-matter curve (bright voxels)."""
GOOD = "#83C167"
"""Description: Colour of the log-likelihood readout."""
config.background_color = "#0b0d10"

RNG = np.random.default_rng(20261122)
"""Description: The seeded generator that draws the synthetic voxels."""
TRUE_MU = (40.0, 100.0, 150.0)
"""Description: The brightness means used to draw fluid, gray matter and white matter."""
TRUE_SD = (9.0, 13.0, 10.0)
"""Description: The brightness standard deviations used to draw the three tissues."""
COUNTS = (700, 1100, 900)
"""Description: How many synthetic voxels of each tissue are drawn."""
X = np.concatenate([RNG.normal(m, s, c) for m, s, c in zip(TRUE_MU, TRUE_SD, COUNTS)])
"""Description: Every synthetic voxel's brightness."""
V = len(X)
"""Description: The number of synthetic voxels."""
INIT = (np.array([1 / 3, 1 / 3, 1 / 3]), np.array([70.0, 85.0, 105.0]), np.array([30.0, 30.0, 30.0]))
"""Description: The poor starting guess as (weight, mean, standard deviation) arrays over the three tissues."""
SHOW = (0, 1, 2, 5, 20, 100)
"""Description: The EM iterations drawn on screen."""
XMIN, XMAX, BIN = 0.0, 200.0, 5.0
"""Description: The brightness range of the histogram and its bin width."""
LEFT_X, RIGHT_X, BASE_Y, TOP_Y = -6.0, 6.0, -1.85, 1.6
"""Description: The screen box the histogram fills."""


class EMInMedicine(Scene):
    """
    Description: The video's scene, three beats in order.
    Fields:
    - cap: the caption on screen, or None.
    """

    def construct(self):
        """
        Description: Plays the title, the MRI example and the closing card.
        Inputs: none.
        Outputs: none; animates the scene.
        """
        self.cap = None
        self.title_card()
        self.mri()
        self.closing()

    def title_card(self):
        """
        Description: Shows the title and the one-sentence idea behind EM.
        Inputs: none.
        Outputs: none; ends on an empty frame.
        """
        title = Text("EM in medicine", font=SERIF, font_size=60)
        sub = Text("EM is for problems where something you never measure\nwould make the fit easy if you knew it.",
                   font=SERIF, font_size=30, color=GREY_B, line_spacing=1.2).next_to(title, DOWN, buff=0.6)
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(sub), run_time=1.0)
        self.wait(5)
        self.play(FadeOut(title, sub))

    def mri(self):
        """
        Description: Draws the synthetic voxel histogram and animates EM fitting three tissue curves to it.
        Inputs: none.
        Outputs: none; ends on an empty frame.
        """
        head = Text("MRI brain tissue segmentation", font=SERIF, font_size=38).to_edge(UP, buff=0.35)
        self.play(FadeIn(head))
        self.say("The MRI image itself is built with Fourier transforms, not EM.\nEM labels the voxels afterward.", 5)

        hist = self.histogram()
        tag = Text("synthetic data", font=SERIF, font_size=22, color=GREY_B).move_to([4.9, TOP_Y + 0.3, 0])
        self.play(FadeIn(hist), FadeIn(tag))
        self.say("Measured: each voxel's brightness x.\nHidden: each voxel's tissue type z.", 5)

        hist_list = run_em(INIT, max(SHOW))
        curves = self.curves(hist_list[0][0])
        legend = self.legend()
        ll_text = self.readout(0, hist_list[0][1])
        self.play(Create(curves), FadeIn(legend), FadeIn(ll_text))
        self.say("Start from a poor guess for the three tissue curves.", 3)
        self.say("E-step: split each voxel's brightness among the three tissues.\n"
                 "M-step: refit each tissue's curve to its shares.", 4)

        for t in SHOW[1:]:
            params, ll = hist_list[t]
            self.play(Transform(curves, self.curves(params)),
                      Transform(ll_text, self.readout(t, ll)), run_time=1.6)
            self.wait(1.2)

        mu = hist_list[max(SHOW)][0][1]
        self.say(f"After {max(SHOW)} iterations the means are {mu[0]:.0f}, {mu[1]:.0f} and {mu[2]:.0f}.\n"
                 f"Each voxel then gets the tissue with the largest share.", 5)
        self.play(*[FadeOut(m) for m in (head, hist, tag, curves, legend, ll_text, self.cap)])
        self.cap = None

    def closing(self):
        """
        Description: Lists two other medical uses of EM, one sentence each.
        Inputs: none.
        Outputs: none; ends on an empty frame.
        """
        head = Text("Two other medical uses", font=SERIF, font_size=42).to_edge(UP, buff=0.8)
        lines = VGroup(
            Text("PET scans: EM builds the image itself, and the hidden data is where in\n"
                 "the body each detected emission started (Shepp and Vardi, 1982).",
                 font=SERIF, font_size=28, line_spacing=1.2),
            Text("EM also combines several doctors' diagnoses when the true\n"
                 "diagnosis is unknown (Dawid and Skene, 1979).",
                 font=SERIF, font_size=28, line_spacing=1.2),
        ).arrange(DOWN, buff=0.8, aligned_edge=LEFT).next_to(head, DOWN, buff=0.9)
        self.play(FadeIn(head))
        for line in lines:
            self.play(FadeIn(line, shift=UP * 0.2))
            self.wait(4.5)
        self.wait(2)
        self.play(FadeOut(head, lines))

    def histogram(self):
        """
        Description: Builds the bar histogram of the synthetic brightness values and its axis labels.
        Inputs: none.
        Outputs: a VGroup of bars, a baseline and tick labels.
        """
        counts, edges = np.histogram(X, bins=np.arange(XMIN, XMAX + BIN, BIN))
        bars = VGroup()
        for c, a in zip(counts, edges[:-1]):
            if c == 0:
                continue
            bars.add(Polygon([sx(a), BASE_Y, 0], [sx(a + BIN), BASE_Y, 0], [sx(a + BIN), sy(c), 0], [sx(a), sy(c), 0],
                             stroke_width=0.5, stroke_color=GREY_D, fill_color=GREY_C, fill_opacity=0.45))
        axis = Line([LEFT_X, BASE_Y, 0], [RIGHT_X, BASE_Y, 0], stroke_width=2, color=GREY_B)
        ticks = VGroup(*[Text(str(v), font=SERIF, font_size=20, color=GREY_B).move_to([sx(v), BASE_Y - 0.25, 0])
                         for v in range(0, 201, 50)])
        label = Text("voxel brightness x", font=SERIF, font_size=22, color=GREY_B).move_to([0, BASE_Y - 0.6, 0])
        return VGroup(bars, axis, ticks, label)

    def curves(self, params):
        """
        Description: Draws each tissue's weighted bell curve on the histogram's count scale.
        Inputs: params, a (weight, mean, standard deviation) tuple of arrays over the three tissues.
        Outputs: a VGroup of three curves.
        """
        g = VGroup()
        xs = np.linspace(XMIN, XMAX, 300)
        for k, col in enumerate((FLUID, GRAY_M, WHITE_M)):
            ys = comp_pdf(xs, params, k) * V * BIN
            g.add(VMobject(stroke_color=col, stroke_width=4).set_points_smoothly(
                [[sx(a), sy(min(b, 1.15 * YMAX)), 0] for a, b in zip(xs, ys)]))
        return g

    def legend(self):
        """
        Description: Names the three tissue colours.
        Inputs: none.
        Outputs: a VGroup of three coloured labels.
        """
        return VGroup(*[Text(s, font=SERIF, font_size=22, color=c) for s, c in
                        (("fluid (dark)", FLUID), ("gray matter", GRAY_M), ("white matter (bright)", WHITE_M))]
                      ).arrange(RIGHT, buff=0.6).move_to([-2.2, TOP_Y + 0.3, 0])

    def readout(self, t, ll):
        """
        Description: Builds the iteration and log-likelihood readout.
        Inputs: t, the iteration number; ll, the mean log-likelihood per voxel.
        Outputs: a Text mobject.
        """
        return Text(f"iteration {t}    mean log-likelihood per voxel: {ll:.3f}", font=SERIF, font_size=24,
                    color=GOOD).move_to([0, TOP_Y - 0.15, 0])

    def say(self, text, wait=2.0):
        """
        Description: Replaces the caption at the bottom of the frame.
        Inputs: text, the caption; wait, seconds to hold it.
        Outputs: none; animates the caption.
        """
        new = Text(text, font=SERIF, font_size=26, line_spacing=1.15).to_edge(DOWN, buff=0.25)
        if self.cap is None:
            self.play(FadeIn(new), run_time=0.6)
        else:
            self.play(FadeOut(self.cap), FadeIn(new), run_time=0.6)
        self.cap = new
        self.wait(wait)


def run_em(params, steps):
    """
    Description: Runs EM from params and records every iterate, asserting the log-likelihood never decreases.
    Inputs: params, the starting (weight, mean, standard deviation) tuple; steps, how many iterations to run.
    Outputs: a list whose entry t is (params after t iterations, mean log-likelihood).
    """
    out = [(params, loglik(params))]
    for _ in range(steps):
        params = em_step(params)
        ll = loglik(params)
        assert ll >= out[-1][1] - 1e-12, "EM log-likelihood decreased"
        out.append((params, ll))
    return out


def em_step(params):
    """
    Description: One EM iteration for the three-tissue mixture.
    Inputs: params, the current (weight, mean, standard deviation) tuple.
    Outputs: the updated (weight, mean, standard deviation) tuple.
    """
    dens = np.stack([comp_pdf(X, params, k) for k in range(3)])
    share = dens / dens.sum(axis=0)
    total = share.sum(axis=1)
    w = total / V
    mu = (share * X).sum(axis=1) / total
    sd = np.sqrt((share * (X - mu[:, None]) ** 2).sum(axis=1) / total)
    return (w, mu, sd)


def loglik(params):
    """
    Description: The mean log-likelihood of the voxels under the mixture.
    Inputs: params, a (weight, mean, standard deviation) tuple.
    Outputs: a float.
    """
    return float(np.mean(np.log(sum(comp_pdf(X, params, k) for k in range(3)))))


def comp_pdf(x, params, k):
    """
    Description: Tissue k's weight times its Gaussian density.
    Inputs: x, brightness values; params, a (weight, mean, standard deviation) tuple; k, the tissue index.
    Outputs: an array the shape of x.
    """
    w, mu, sd = params
    return w[k] * np.exp(-0.5 * ((x - mu[k]) / sd[k]) ** 2) / (sd[k] * np.sqrt(2 * np.pi))


YMAX = float(np.histogram(X, bins=np.arange(XMIN, XMAX + BIN, BIN))[0].max())
"""Description: The tallest histogram bar, which sets the vertical scale."""


def sx(v):
    """
    Description: Maps a brightness value to a screen x coordinate.
    Inputs: v, a brightness.
    Outputs: a float.
    """
    return LEFT_X + (RIGHT_X - LEFT_X) * (v - XMIN) / (XMAX - XMIN)


def sy(c):
    """
    Description: Maps a voxel count to a screen y coordinate.
    Inputs: c, a count per bin.
    Outputs: a float.
    """
    return BASE_Y + (TOP_Y - 0.45 - BASE_Y) * c / YMAX
