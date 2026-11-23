"""Renders a Manim video on three medical uses of EM: labelling MRI voxels, building a PET image and combining doctors' calls.

Contents:
- EMInMedicine: the scene, with one method per section of the video
- Chart: a plot frame whose ranges need not contain 0
- precompute: every number on screen, with asserts for the claims the captions make
- mri_slice, mri_em_step, mri_loglik, tissue_pdf: the synthetic MRI slice and its three-tissue mixture
- pet_phantom, pet_system_matrix, pet_em, pet_loglik: the synthetic PET phantom, its detector lines and ML-EM
- pet_zoom: the pixels and numbers of the close-up on one detector line
- doc_seed_survey, doctor_calls, ds_estep, ds_mstep: the synthetic doctors' calls and Dawid and Skene's EM
- strip_corners, fit_right, clip_polygon, edge_side: the outline of one detector line and text placement
- image_mob, heat, symbol_card, cell_text, M, T: drawing helpers
- plain_len, num, pct: reading time and number formatting
- fonts, colours, seeds and the sizes of the three synthetic datasets
"""
import math
import os
import re

import numpy as np
from manim import *

SERIF = "Georgia"
"""Description: Serif font for captions, headings and plain text."""
MATH = "Cambria"
"""Description: Font for formulas, because Cambria has the Greek letters, ∑, ∏ and arrows they need."""
FLUID = "#58C4DD"
"""Description: Colour of fluid, the darkest tissue."""
GRAY_M = "#B48EDE"
"""Description: Colour of gray matter, the middle tissue."""
WHITE_M = "#FFD166"
"""Description: Colour of white matter, the brightest tissue."""
GOOD = "#83C167"
"""Description: Colour of every log-likelihood readout and of EM's estimates."""
COND = "#FC6255"
"""Description: Colour of a "yes" call and of the chance that a patient has the condition."""
HIDDEN = "#F0A35E"
"""Description: Colour of the word "hidden" and of the hidden quantity on each card."""
PANEL = "#12161b"
"""Description: Fill colour of cards and tables."""
config.background_color = "#0b0d10"

MRI_SEED = 20261122
"""Description: The seed that draws the synthetic MRI brightness values."""
MRI_G = 64
"""Description: The synthetic MRI slice is MRI_G by MRI_G voxels."""
TRUE_MU = (40.0, 100.0, 150.0)
"""Description: The mean brightness used to draw fluid, gray matter and white matter."""
TRUE_SD = (9.0, 13.0, 10.0)
"""Description: The brightness standard deviation used to draw each tissue."""
MRI_INIT = (np.array([1 / 3, 1 / 3, 1 / 3]), np.array([70.0, 85.0, 105.0]), np.array([30.0, 30.0, 30.0]))
"""Description: The poor starting guess, as (fraction, mean, spread) arrays over the three tissues."""
MRI_SHOW = (0, 1, 2, 5, 20, 100)
"""Description: The MRI EM iterations drawn on screen."""
XMIN, XMAX, BIN = 0.0, 200.0, 5.0
"""Description: The brightness range of the MRI histogram and its bin width."""
H_LEFT, H_RIGHT, H_BASE, H_TOP = -6.6, 0.9, -1.55, 1.95
"""Description: The screen box the MRI histogram fills."""

PET_SEED = 7
"""Description: The seed that draws the synthetic PET counts."""
PET_G = 16
"""Description: The PET image is PET_G by PET_G pixels."""
PET_ANGLES = 18
"""Description: The number of detector-line angles, spread evenly over 180 degrees."""
PET_SUB = 8
"""Description: Each pixel is split into PET_SUB by PET_SUB points to measure how much of it lies inside a line."""
PET_SCALE = 4.0
"""Description: Multiplies the phantom's activity, which sets how many counts the scan collects."""
PET_ITERS = 200
"""Description: How many ML-EM iterations are computed."""
PET_SHOW = (0, 1, 2, 5, 20, 100)
"""Description: The ML-EM iterations shown in the gallery."""
PET_PLAY = (0, 1, 2, 3, 5, 10, 20, 50, 100)
"""Description: The ML-EM iterations stepped through in the animation."""
PET_ZOOM_ANGLE = 3
"""Description: The angle index of the one detector line examined up close (30 degrees)."""

DOC_SEED = 2
"""Description: The seed that draws the synthetic patients and doctors' calls."""
DOC_P = 100
"""Description: The number of synthetic patients."""
DOC_SHOWN = 40
"""Description: How many patients the table of calls shows."""
DOC_PREV = 0.35
"""Description: The chance that a synthetic patient has the condition."""
DOC_SE = (0.95, 0.70, 0.60)
"""Description: The chance that doctors A, B and C say yes to a patient who has the condition."""
DOC_SP = (0.95, 0.70, 0.65)
"""Description: The chance that doctors A, B and C say no to a patient who does not have it."""
DOC_ITERS = 500
"""Description: How many Dawid and Skene iterations are computed."""
DOC_PLAY = (0, 1, 2, 5, 20, 100, 500)
"""Description: The Dawid and Skene iterations stepped through in the table."""
SETTLED = 1e-4
"""Description: An estimate has settled when no parameter moves more than this in one iteration."""
DOC_SURVEY = 30
"""Description: How many seeds, 0 to DOC_SURVEY - 1, the note's seed survey of EM against the majority vote covers."""
READ_RATE = 20
"""Description: Reading speed in characters a second, which sets how long text stays on screen."""


class EMInMedicine(Scene):
    """
    Description: The video's scene. Each section method draws one part of the video and ends on an empty frame.
    Fields:
    - cap: the caption on screen, or None.
    - cap_text, cap_start: the caption's markup and the scene time it appeared.
    - head: the section heading on screen, or None.
    - d: the numbers from precompute().
    """

    def construct(self):
        """
        Description: Computes the numbers, then plays the sections in order.
        Inputs: none
        Outputs: none
        Constraints: when the EMM_ONLY environment variable names sections, separated by commas, only those play.
        """
        self.cap = None
        self.cap_text, self.cap_start = "", 0.0
        self.head = None
        self.d = precompute()
        only = [s for s in os.environ.get("EMM_ONLY", "").split(",") if s]
        for name in ("intro", "mri", "pet", "doctors", "summary"):
            if not only or name in only:
                getattr(self, name)()

    def intro(self):
        """
        Description: Says what EM needs, separates forming an image from labelling one, and previews the three examples.
        Inputs: none
        Outputs: none
        """
        self.mark("intro")
        title = T("EM in medicine", 56)
        sub = T("three worked examples, all on synthetic data", 28, GREY_B).next_to(title, DOWN, buff=0.3)
        words = VGroup(title, sub).move_to(0.3 * UP)
        self.play(Write(title), FadeIn(sub, shift=0.2 * UP), run_time=1.6)
        self.wait(2.5)
        self.play(FadeOut(words))

        self.heading("What EM needs")
        row_text = ("<b>measured</b>:  the numbers the instrument records",
                    f"<span foreground='{HIDDEN}'><b>hidden</b></span>:  a quantity never recorded, which would make the fit easy if known",
                    "<b>unknown settings</b>:  what we want to estimate from the measured numbers")
        rows = VGroup(*[M(t, 30) for t in row_text]).arrange(DOWN, aligned_edge=LEFT, buff=0.42)
        step_text = (f"<span foreground='{GOOD}'>E-step</span>:  use the current settings to split each measurement among the hidden possibilities",
                     f"<span foreground='{GOOD}'>M-step</span>:  refit the settings as if those splits were the truth, then repeat")
        steps = VGroup(*[M(t, 27) for t in step_text]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        steps.next_to(rows, DOWN, buff=0.6, aligned_edge=LEFT)
        block = VGroup(rows, steps)
        if block.width > 12.8:
            block.scale_to_fit_width(12.8)
        block.move_to(0.75 * UP)
        self.say("EM fits a model when part of the story was never recorded. It needs three pieces.", wait=0)
        for r in rows:
            self.play(FadeIn(r, shift=0.1 * RIGHT), run_time=0.8)
        self.read_hold(*row_text)
        self.say("Video 01 showed the two steps on a two-curve mixture, where the log-likelihood never went down.", wait=0)
        self.play(FadeIn(steps, shift=0.1 * UP, lag_ratio=0.4), run_time=1.4)
        self.read_hold(*step_text)
        self.clear_stage()

        self.mark("intro: the three examples")
        self.heading("Three examples")
        card_text = (("MRI segmentation", "The scanner forms the image\nwith Fourier transforms.\nEM labels each voxel\nas a tissue.", FLUID),
                     ("PET reconstruction", "There is no image at first.\nEM builds the image\nfrom detector counts.", WHITE_M),
                     ("Doctors' diagnoses", "EM estimates each patient's\ntrue diagnosis and how\nreliable each doctor is.", COND))
        cards = VGroup(*[symbol_card(t, body, col, 4.2) for t, body, col in card_text])
        cards.arrange(RIGHT, buff=0.3, aligned_edge=UP).move_to(0.5 * UP)
        self.say("Three worked examples follow.", wait=0)
        self.play(LaggedStart(*[FadeIn(c, shift=0.15 * UP) for c in cards], lag_ratio=0.4), run_time=2)
        self.read_hold(*[t + body for t, body, _ in card_text])
        self.end_section()

    def mri(self):
        """
        Description: Fits the three-tissue mixture to a synthetic MRI slice, then colours each voxel by its largest share.
        Inputs: none
        Outputs: none
        """
        self.mark("mri")
        d = self.d
        self.heading("MRI: EM labels voxels in an image that already exists")
        slice_img = image_mob(np.repeat(d["mri_gray"][:, :, None], 3, axis=2), 3.9).move_to(np.array([4.4, 0.35, 0]))
        frame = SurroundingRectangle(slice_img, buff=0, stroke_width=1.5, color=GREY_B)
        tag = T(f"synthetic slice, {MRI_G} × {MRI_G} voxels", 20, GREY_B).next_to(slice_img, DOWN, buff=0.15)
        self.play(FadeIn(slice_img), Create(frame), FadeIn(tag), run_time=1.2)
        self.say("An MRI scanner forms its image with Fourier transforms, not EM. Here is a synthetic slice.\n"
                 "EM's job comes afterward: deciding which tissue each voxel shows.", wait=6)

        card_text = ("<b>measured</b>:  x<sub>v</sub>, the brightness of voxel v",
                     f"<span foreground='{HIDDEN}'><b>hidden</b></span>:  z<sub>v</sub>, the tissue that voxel v shows",
                     "<b>unknown</b>:  π<sub>t</sub>, μ<sub>t</sub>, σ<sub>t</sub>, the fraction of voxels, mean",
                     "brightness and spread of tissue t")
        card = VGroup(*[M(t, 24) for t in card_text]).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        card[3].shift(1.25 * RIGHT)
        card.move_to(np.array([-2.85, 1.15, 0]))
        self.say(f"The fit uses the {d['mri_V']:,} voxels inside the brain.", wait=0)
        self.play(FadeIn(card, lag_ratio=0.2), run_time=1.2)
        self.read_hold(*card_text)
        self.play(FadeOut(card))

        form_text = ("E-step:   r<sub>vt</sub> = π<sub>t</sub> N(x<sub>v</sub> | μ<sub>t</sub>, σ<sub>t</sub>) / "
                     "∑<sub>u</sub> π<sub>u</sub> N(x<sub>v</sub> | μ<sub>u</sub>, σ<sub>u</sub>)",
                     "r<sub>vt</sub>: voxel v's share for tissue t",
                     "N: the bell-curve density;  u: any of the three tissues",
                     "M-step:   refit π<sub>t</sub>, μ<sub>t</sub>, σ<sub>t</sub> from the shares, as in video 01")
        e_form = M(form_text[0], 28)
        e_note = VGroup(M(form_text[1], 23, GREY_B), M(form_text[2], 23, GREY_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        m_form = M(form_text[3], 28)
        forms = VGroup(e_form, e_note, m_form).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        e_note.shift(1.3 * RIGHT)
        if forms.width > 8.4:
            forms.scale_to_fit_width(8.4)
        forms.move_to(np.array([-6.7 + forms.width / 2, 0.6, 0]))
        self.say("E-step: each voxel's share for each tissue comes from how well that tissue's curve explains\n"
                 "its brightness. M-step: each curve is refit from those shares.", wait=0)
        self.play(FadeIn(forms, lag_ratio=0.2), run_time=1.2)
        self.read_hold(*form_text)
        self.play(FadeOut(forms))

        self.mark("mri: EM fits the three tissue curves")
        hist = self.mri_histogram()
        legend = VGroup(*[T(s, 21, c) for s, c in
                          (("fluid", FLUID), ("gray matter", GRAY_M), ("white matter", WHITE_M))]
                        ).arrange(RIGHT, buff=0.5).move_to(np.array([(H_LEFT + H_RIGHT) / 2, H_TOP + 0.75, 0]))
        path = d["mri_path"]
        curves = self.mri_curves(path[0][0])
        readout = self.mri_readout(0, path[0][1])
        self.play(FadeIn(hist), run_time=0.8)
        self.say("The histogram counts the voxels at each brightness. Start EM from a poor guess:\n"
                 "three wide curves near the middle.", wait=0)
        self.play(Create(curves), FadeIn(legend), FadeIn(readout), run_time=1.4)
        self.wait(3)
        self.say("Each step shown is one E-step and one M-step. The curves move onto the three peaks.", wait=0)
        for t in MRI_SHOW[1:]:
            params, ll = path[t]
            new_read = self.mri_readout(t, ll)
            self.play(Transform(curves, self.mri_curves(params)), run_time=1.3)
            self.play(FadeOut(readout), FadeIn(new_read), run_time=0.3)
            readout = new_read
            self.wait(0.8)
        mu = d["mri_mu_text"]
        self.say(f"After {MRI_SHOW[-1]} iterations the means are {mu[0]}, {mu[1]} and {mu[2]}, close to the {num(TRUE_MU[0], '.0f')}, "
                 f"{num(TRUE_MU[1], '.0f')} and {num(TRUE_MU[2], '.0f')}\nthat drew the data. "
                 "The mean log-likelihood never went down.", wait=4)

        self.mark("mri: voxels coloured by their largest share")
        labelled = image_mob(d["mri_label_rgb"], 3.9).move_to(slice_img)
        self.say("Now colour each voxel by the tissue with its largest share. "
                 f"The result agrees with the true\nsynthetic tissue map on {d['mri_agree']} of the voxels inside the brain.", wait=0)
        self.play(FadeIn(labelled), run_time=1.5)
        self.wait(5)
        self.say("Wells and colleagues (1996) used EM to segment brain MRI while also correcting uneven\n"
                 "brightness across the image. The plain three-tissue mixture here is a simplification.", wait=8)
        self.end_section()

    def pet(self):
        """
        Description: Reconstructs a synthetic PET phantom with ML-EM, from detector lines to the image sharpening.
        Inputs: none
        Outputs: none
        """
        self.mark("pet")
        d = self.d
        self.heading("PET: EM builds the image itself")
        centre = np.array([-4.4, 0.35, 0])
        size = 3.2
        truth = image_mob(heat(d["pet_true"].reshape(PET_G, PET_G), d["pet_vmax"]), size).move_to(centre)
        box = SurroundingRectangle(truth, buff=0, stroke_width=1.5, color=GREY_B)
        ring_r = 2.45
        ring = VGroup(*[Rectangle(width=0.1, height=0.24).set_stroke(GREY_B, 1.2).set_fill(PANEL, 1)
                        .rotate(a).move_to(centre + ring_r * np.array([math.cos(a), math.sin(a), 0]))
                        for a in np.linspace(0, TAU, 48, endpoint=False)])
        tlab = T("true activity (synthetic)", 20, GREY_B).next_to(ring, DOWN, buff=0.12)
        self.play(FadeIn(truth), Create(box), FadeIn(ring), FadeIn(tlab), run_time=1.2)
        self.say("MRI hands EM a finished image to label. PET is different: the scanner records only counts,\n"
                 "and EM is one way to turn those counts into an image. The image on the left is the synthetic truth.", wait=6)
        self.mark("pet: positrons, gamma-ray pairs and detector lines")
        self.say("A PET tracer releases positrons. Each one meets an electron, and the pair becomes two\n"
                 "gamma rays that leave 180 degrees apart. Two detectors fire at once, which marks a line.", wait=0)
        flashes = []
        for (px, py), a in d["pet_events"][:5]:
            p = centre + np.array([px, py, 0]) * size / PET_G
            u = np.array([math.cos(a), math.sin(a), 0])
            tt = -np.dot(p - centre, u)
            disc = tt ** 2 - (np.dot(p - centre, p - centre) - ring_r ** 2)
            h1 = p + (tt + math.sqrt(disc)) * u
            h2 = p + (tt - math.sqrt(disc)) * u
            src = Dot(p, radius=0.06, color=WHITE)
            g1, g2 = Dot(p, radius=0.045, color=GOOD), Dot(p, radius=0.045, color=GOOD)
            self.play(FadeIn(src, scale=2), run_time=0.25)
            self.play(g1.animate.move_to(h1), g2.animate.move_to(h2), run_time=0.5)
            line = Line(h1, h2).set_stroke(GOOD, 2, opacity=0.6)
            self.play(Create(line), FadeOut(src), FadeOut(g1), FadeOut(g2), run_time=0.35)
            flashes.append(line)
        self.wait(2)
        self.play(*[FadeOut(f) for f in flashes], FadeOut(ring), run_time=0.6)

        card_text = ("<b>measured</b>:  y<sub>i</sub>, the number of counts on detector line i",
                     f"<span foreground='{HIDDEN}'><b>hidden</b></span>:  which pixel each counted emission came from",
                     "<b>unknown</b>:  λ<sub>j</sub>, the activity of pixel j: its expected",
                     "number of emissions during the scan")
        card = VGroup(*[M(t, 24) for t in card_text]).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        card[3].shift(1.2 * RIGHT)
        fit_right(card, 1.7)
        fact_text = (f"{PET_G} × {PET_G} = {PET_G * PET_G} pixels;   {PET_ANGLES} angles × {PET_G} lines = {d['pet_lines']} detector lines",
                     f"{d['pet_total']:,} counts in all, drawn as Poisson counts with a fixed seed")
        facts = VGroup(*[M(t, 22, GREY_B) for t in fact_text]).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        facts.next_to(card, DOWN, buff=0.4, aligned_edge=LEFT)
        fit_right(facts, facts.get_center()[1])
        self.say("Here is what the scan gives EM, and what EM must estimate.", wait=0)
        self.play(FadeIn(card, lag_ratio=0.2), run_time=1.2)
        self.play(FadeIn(facts, lag_ratio=0.2), run_time=1)
        self.read_hold(*card_text, *fact_text)
        self.play(FadeOut(card), FadeOut(facts))

        self.mark("pet: the system matrix a_ij, the E-step and the M-step")
        i0 = d["pet_zoom_line"]
        corners = strip_corners(PET_ZOOM_ANGLE, i0 % PET_G, centre, size)
        strip = Polygon(*corners).set_stroke(WHITE, 2).set_fill(WHITE, 0.18)
        aij_text = ("a<sub>ij</sub>:  how much pixel j contributes to line i, meaning the",
                    "chance that an emission from pixel j is counted on line i")
        aij = VGroup(*[M(t, 24) for t in aij_text]).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        aij[1].shift(0.5 * RIGHT)
        fit_right(aij, 2.2)
        zm = d["pet_zoom"]
        val_text = (f"green pixel:  {zm['a_big'][0]} of it inside line i,  a<sub>ij</sub> = {zm['a_big'][0]} / {PET_ANGLES} = {zm['a_big'][1]}",
                    f"blue pixel:  {zm['a_small'][0]} of it inside,  a<sub>ij</sub> = {zm['a_small'][0]} / {PET_ANGLES} = {zm['a_small'][1]}",
                    "pixel outside the line:  a<sub>ij</sub> = 0",
                    f"(an emission's line lies at each of the {PET_ANGLES} angles with chance 1/{PET_ANGLES})")
        a_vals = VGroup(*[M(t, 22) for t in val_text[:3]]).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        a_vals.next_to(aij, DOWN, buff=0.3, aligned_edge=LEFT).shift(0.5 * RIGHT)
        px = size / PET_G
        a_marks = VGroup(Square(side_length=px).set_stroke(GOOD, 4).move_to(centre + zm["big_xy"] * px),
                         Square(side_length=px).set_stroke(FLUID, 3).move_to(centre + zm["small_xy"] * px))
        why = M(val_text[3], 20, GREY_B)
        why.next_to(a_vals, DOWN, buff=0.2, aligned_edge=LEFT)
        fit_right(VGroup(a_vals, why), VGroup(a_vals, why).get_center()[1], left=-1.1)
        self.say(f"The model treats each detector line as a strip across the image.\n"
                 f"Line i, drawn here, recorded y<sub>i</sub> = {d['pet_zoom_y']} counts.", wait=0)
        self.play(FadeIn(strip), FadeIn(aij), run_time=1)
        self.play(FadeIn(a_vals, lag_ratio=0.2), FadeIn(why), Create(a_marks), run_time=1.2)
        self.read_hold(*aij_text, *val_text)
        self.play(FadeOut(a_vals), FadeOut(why), FadeOut(a_marks))

        z = d["pet_zoom"]
        head_row = VGroup(*[cell_text(s, 20, GREY_B) for s in ("guess", "red pixel\n(hot spot)", "white pixel\n(body)", "rest of\nthe line", "total")])
        rows = [head_row]
        for name, sh in (("flat start", z["flat"]), ("after 1 iteration", z["iter1"])):
            rows.append(VGroup(M(name, 21), M(sh[0], 22, COND), M(sh[1], 22), M(sh[2], 22), M(sh[3], 22)))
        table = VGroup(*rows)
        xs = (0.0, 2.35, 4.0, 5.65, 7.15)
        for r, row in enumerate(table):
            for c, cell in enumerate(row):
                cell.move_to(np.array([-1.6 + xs[c], 0.1 - 0.55 * r - (0.15 if r else 0), 0]), aligned_edge=LEFT)
        e_text = ("E-step:   count from line i given to pixel j  =  y<sub>i</sub> · a<sub>ij</sub> λ<sub>j</sub> / "
                  "∑<sub>k</sub> a<sub>ik</sub> λ<sub>k</sub>",
                  "k runs over every pixel; the bottom is line i's predicted count",
                  "guess red pixel (hot spot) white pixel (body) rest of the line total flat start " + " ".join(z["flat"]))
        e_form = M(e_text[0], 24)
        fit_right(e_form, 1.6)
        k_note = M(e_text[1], 20, GREY_B).next_to(e_form, DOWN, buff=0.15, aligned_edge=LEFT)
        hot = Square(side_length=size / PET_G).set_stroke(COND, 4).move_to(centre + z["hot_xy"] * size / PET_G)
        bg = Square(side_length=size / PET_G).set_stroke(WHITE, 4).move_to(centre + z["bg_xy"] * size / PET_G)
        self.say(f"E-step: split line i's {d['pet_zoom_y']} counts in proportion to a<sub>ij</sub> λ<sub>j</sub>.\n"
                 "From a flat guess the two marked pixels get almost the same share.", wait=0)
        self.play(FadeOut(aij), FadeIn(e_form), FadeIn(k_note), Create(hot), Create(bg), run_time=1)
        self.play(FadeIn(table[0]), FadeIn(table[1]), run_time=0.8)
        self.read_hold(*e_text)
        self.say("After one iteration the guess is brighter where the counts pile up, so the next E-step\n"
                 "gives the hot pixel a larger share of the same line's counts.", wait=0)
        self.play(FadeIn(table[2], shift=0.1 * DOWN), run_time=0.8)
        self.wait(6)
        self.play(FadeOut(table), FadeOut(hot), FadeOut(bg), FadeOut(strip), FadeOut(k_note))

        m_text = ("M-step:   λ<sub>j</sub>  ←  (λ<sub>j</sub> / s<sub>j</sub>)  ∑<sub>i</sub>  a<sub>ij</sub> y<sub>i</sub> / "
                  "∑<sub>k</sub> a<sub>ik</sub> λ<sub>k</sub>",
                  "s<sub>j</sub> = ∑<sub>i</sub> a<sub>ij</sub>:  pixel j's coverage, the chance that any line counts its emission",
                  "new activity  =  counts given to pixel j by all lines  /  its coverage",
                  f"coverage s<sub>j</sub>: {num(d['pet_cov_min'], '.2f')} in the corners, 1 inside the body")
        m_form = VGroup(M(m_text[0], 26), M(m_text[1], 21, GREY_B), M(m_text[2], 22))
        m_form.arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(e_form, DOWN, buff=0.35, aligned_edge=LEFT)
        fit_right(m_form, m_form.get_center()[1])
        cov = image_mob(heat(d["pet_cov"].reshape(PET_G, PET_G) - 0.5, 0.5, gray=True), 1.5)
        cov.next_to(m_form, DOWN, buff=0.3, aligned_edge=LEFT)
        cov_lab = T(m_text[3], 19, GREY_B, markup=True)
        cov_lab.next_to(cov, RIGHT, buff=0.25)
        self.say("M-step: divide the counts given to pixel j by its coverage. Corner pixels fall outside\n"
                 "some lines, so they are counted less often and need the correction.", wait=0)
        self.play(FadeIn(m_form, lag_ratio=0.3), run_time=1.4)
        self.play(FadeIn(cov), FadeIn(cov_lab))
        self.read_hold(*m_text)
        self.say("Shepp and Vardi (1982) derived this EM algorithm for emission tomography. Their model treats\n"
                 "each pixel's emissions as a Poisson random number with mean λ<sub>j</sub>.", wait=8)
        self.clear_stage()

        self.mark("pet: ML-EM iterations and the log-likelihood")
        size2 = 3.0
        left_c, mid_c = np.array([-5.2, 0.35, 0]), np.array([-1.6, 0.35, 0])
        truth2 = image_mob(heat(d["pet_true"].reshape(PET_G, PET_G), d["pet_vmax"]), size2).move_to(left_c)
        t_lab = T("true activity (synthetic)", 20, GREY_B).next_to(truth2, DOWN, buff=0.15)
        hist = d["pet_hist"]
        est = image_mob(heat(hist[0].reshape(PET_G, PET_G), d["pet_vmax"]), size2).move_to(mid_c)
        e_lab = self.pet_label(0, mid_c, size2)
        lls = d["pet_ll"]
        lo, hi = d["pet_ll_range"]
        chart = Chart((0, PET_PLAY[-1]), (lo, hi), 4.4, 3.0, np.array([4.4, 0.55, 0]), xticks=(0, 50, 100),
                      yticks=d["pet_ll_ticks"], tick_size=18)
        c_lab = M("Poisson log-likelihood of the counts", 21, GREY_B).next_to(chart.group, UP, buff=0.15)
        pts = VGroup(Dot(chart.c2p(0, lls[0]), radius=0.05, color=GOOD))
        ll_read = self.pet_ll_read(0, lls[0], chart)
        self.play(FadeIn(truth2), FadeIn(t_lab), FadeIn(est), FadeIn(e_lab), FadeIn(chart.group), FadeIn(c_lab),
                  FadeIn(pts), FadeIn(ll_read), run_time=1.2)
        self.say("Start from a flat image and repeat the two steps. Left is the true activity; the middle is\n"
                 "EM's estimate, drawn on the same colour scale.", wait=6)
        prev = 0
        for t in PET_PLAY[1:]:
            new = image_mob(heat(hist[t].reshape(PET_G, PET_G), d["pet_vmax"]), size2).move_to(mid_c)
            seg = VMobject().set_points_as_corners([chart.c2p(s, lls[s]) for s in range(prev, t + 1)]).set_stroke(GOOD, 3)
            p = Dot(chart.c2p(t, lls[t]), radius=0.05, color=GOOD)
            rt = 1.0 if t <= 3 else 0.7
            new_lab, new_read = self.pet_label(t, mid_c, size2), self.pet_ll_read(t, lls[t], chart)
            self.play(FadeIn(new), FadeOut(est), Create(seg), FadeIn(p), run_time=rt)
            self.play(FadeOut(e_lab), FadeIn(new_lab), FadeOut(ll_read), FadeIn(new_read), run_time=0.25)
            e_lab, ll_read = new_lab, new_read
            self.wait(0.9 if t <= 3 else 0.5)
            pts.add(seg, p)
            est, prev = new, t
        self.say(f"The log-likelihood rose at every one of the {PET_ITERS} iterations computed, from "
                 f"{num(lls[0], ',.0f')} to {num(lls[PET_PLAY[-1]], ',.0f')}\nat iteration {PET_PLAY[-1]}. "
                 "Every EM algorithm has this property.", wait=6)
        self.clear_stage()

        self.mark("pet: the iteration gallery, noise, and OS-EM")
        thumbs = Group()
        sz = 1.72
        items = [("true", d["pet_true"], None)] + [(f"iteration {t}", hist[t], d["pet_err"][t]) for t in PET_SHOW]
        for k, (name, img, err) in enumerate(items):
            im = image_mob(heat(img.reshape(PET_G, PET_G), d["pet_vmax"]), sz).move_to(np.array([-5.85 + 1.95 * k, 1.35, 0]))
            top = T(name, 19, GREY_B).next_to(im, UP, buff=0.15)
            bot = T("synthetic" if err is None else f"error {err:.1f}", 19, GREY_B if err is None else WHITE).next_to(im, DOWN, buff=0.15)
            thumbs.add(Group(im, top, bot))
        err_note = M("error: the root-mean-square difference from the true activity, per pixel", 20, GREY_B)
        err_note.move_to(np.array([0, -0.2, 0]))
        self.play(LaggedStart(*[FadeIn(t) for t in thumbs], lag_ratio=0.15), FadeIn(err_note), run_time=2.2)
        self.say("Side by side, the blur sharpens toward the true image over the first twenty iterations.\n"
                 "Under each image is its error against the true activity.", wait=0)
        self.read_hold("error: the root-mean-square difference from the true activity, per pixel")
        best = d["pet_best_iter"]
        self.say(f"The error is smallest near iteration {best}, then creeps up while the likelihood keeps rising.\n"
                 "With noise-free counts the error keeps falling, so the rise comes from fitting the noise.", wait=8)
        osem_text = ("Hudson and Larkin (1994): ordered subsets EM (OS-EM) updates the image after each group",
                     "of lines. In their SPECT simulations it was about ten times faster at the same image quality.")
        osem = VGroup(*[M(t, 22) for t in osem_text]).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to(np.array([0, -1.45, 0]))
        self.say("Each EM iteration above passes over all the lines before it updates the image once.\n"
                 "SPECT, single photon emission computed tomography, is a related emission scan.", wait=0)
        self.play(FadeIn(osem, lag_ratio=0.2), run_time=1.2)
        self.read_hold(*osem_text)
        self.end_section()

    def doctors(self):
        """
        Description: Runs Dawid and Skene's EM on synthetic yes/no calls from three doctors.
        Inputs: none
        Outputs: none
        """
        self.mark("doctors")
        d = self.d
        self.heading("Combining doctors' calls: Dawid and Skene (1979)")
        calls = d["doc_calls"]
        cell = 0.27
        x0 = -5.2
        names = ("A", "B", "C")
        grid = VGroup()
        for r in range(3):
            for p in range(DOC_SHOWN):
                sq = Square(side_length=cell * 0.86).move_to(np.array([x0 + cell * p, 2.45 - 0.36 * r, 0]))
                if calls[p, r]:
                    sq.set_fill(COND, 1).set_stroke(COND, 1)
                else:
                    sq.set_fill(PANEL, 1).set_stroke(GREY_C, 1)
                grid.add(sq)
        row_lab = VGroup(*[T(f"doctor {n}", 20, GREY_B).move_to(np.array([x0 - 0.95, 2.45 - 0.36 * r, 0])) for r, n in enumerate(names)])
        key = VGroup(Square(side_length=0.2).set_fill(COND, 1).set_stroke(COND, 1), T("yes", 19, GREY_B),
                     Square(side_length=0.2).set_fill(PANEL, 1).set_stroke(GREY_C, 1), T("no", 19, GREY_B),
                     T(f"(first {DOC_SHOWN} of {DOC_P} patients)", 19, GREY_B)).arrange(RIGHT, buff=0.15)
        key.next_to(grid, DOWN, buff=0.2, aligned_edge=LEFT)
        tag = T("synthetic data", 19, GREY_B).next_to(grid, UP, buff=0.18).align_to(grid, RIGHT)
        self.play(FadeIn(row_lab), LaggedStart(*[FadeIn(s) for s in grid], lag_ratio=0.004), FadeIn(key), FadeIn(tag), run_time=2)
        self.say(f"Three doctors each say yes or no to the same question about {DOC_P} patients. They disagree,\n"
                 "and nobody knows the true answers or which doctor to trust.", wait=6)
        card_text = ("<b>measured</b>:  c<sub>pd</sub>, doctor d's call on patient p (1 = yes, 0 = no)",
                     f"<span foreground='{HIDDEN}'><b>hidden</b></span>:  z<sub>p</sub>, whether patient p truly has the condition",
                     "<b>unknown</b>:  π, the fraction of patients who have it",
                     "se<sub>d</sub> (sensitivity): the chance doctor d says yes when it is there",
                     "sp<sub>d</sub> (specificity): the chance doctor d says no when it is absent")
        card = VGroup(*[M(t, 23) for t in card_text]).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        for line in card[3:]:
            line.shift(1.2 * RIGHT)
        card.move_to(np.array([0, -0.55, 0]))
        self.say("If the true diagnoses were known, scoring each doctor would be a simple count.", wait=0)
        self.play(FadeIn(card, lag_ratio=0.2), run_time=1.4)
        self.read_hold(*card_text)
        self.play(FadeOut(card))

        self.mark("doctors: the E-step for one patient and the M-step")
        z = d["doc_zoom"]
        p = z["p"]
        ring = SurroundingRectangle(VGroup(*[grid[r * DOC_SHOWN + p] for r in range(3)]), buff=0.05).set_stroke(WHITE, 2)
        calls_txt = ", ".join(f"{n} {'yes' if calls[p, r] else 'no'}" for r, n in enumerate(names))
        line_text = (f"patient {p + 1}:  {calls_txt}",
                     f"has it:   π × se<sub>A</sub> × (1 − se<sub>B</sub>) × (1 − se<sub>C</sub>)  =  {z['a0']}",
                     f"does not:   (1 − π) × (1 − sp<sub>A</sub>) × sp<sub>B</sub> × sp<sub>C</sub>  =  {z['b0']}",
                     f"q<sub>p</sub>  =  chance patient p has it  =  {z['a0']} / ({z['a0']} + {z['b0']})  =  {z['q0']}")
        lines = VGroup(M(line_text[0], 23), M(line_text[1], 23, COND), M(line_text[2], 23, FLUID), M(line_text[3], 23))
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to(np.array([-0.6, -0.5, 0]))
        self.say(f"E-step for patient {p + 1}, from settings fitted to the majority vote: multiply the chances\n"
                 "of these three calls under each answer.", wait=0)
        self.play(Create(ring), FadeIn(lines, lag_ratio=0.25), run_time=1.6)
        self.read_hold(*line_text)
        self.play(FadeOut(lines))

        qrow = VGroup()
        for p2 in range(DOC_SHOWN):
            q = d["doc_q"][0][p2]
            bar = Rectangle(width=cell * 0.7, height=max(0.02, 0.55 * q)).set_stroke(width=0).set_fill(COND, 1)
            bar.move_to(np.array([x0 + cell * p2, 1.05, 0]), aligned_edge=DOWN)
            qrow.add(bar)
        q_lab = M("q<sub>p</sub>", 22, COND).move_to(np.array([x0 - 0.95, 1.3, 0]))
        q_base = Line(np.array([x0 - cell / 2, 1.05, 0]), np.array([x0 + cell * (DOC_SHOWN - 0.5), 1.05, 0])).set_stroke(GREY_C, 1)
        self.play(FadeOut(key), run_time=0.4)
        self.play(FadeIn(q_base), FadeIn(q_lab), LaggedStart(*[GrowFromEdge(b, DOWN) for b in qrow], lag_ratio=0.02), run_time=1.4)
        m_form = VGroup(
            M("M-step:   π = average of q<sub>p</sub>", 23),
            M("se<sub>d</sub> = ∑<sub>p</sub> q<sub>p</sub> c<sub>pd</sub> / ∑<sub>p</sub> q<sub>p</sub>", 23),
            M("sp<sub>d</sub> = ∑<sub>p</sub> (1 − q<sub>p</sub>)(1 − c<sub>pd</sub>) / ∑<sub>p</sub> (1 − q<sub>p</sub>)", 23),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        m_form[1:].shift(1.15 * RIGHT)
        fit_right(m_form, -1.0, left=-6.8, right=-0.4)
        self.say("Doing this for every patient gives the bars q<sub>p</sub>. M-step: score each doctor against these soft\n"
                 "answers. A patient with q<sub>p</sub> = 0.8 counts as 0.8 of a sick patient and 0.2 of a healthy one.", wait=0)
        self.play(FadeIn(m_form, lag_ratio=0.3), run_time=1.4)
        self.wait(4)

        self.mark("doctors: the estimates converge")
        table = self.doc_table(0)
        self.play(FadeIn(table), run_time=0.8)
        self.wait(4)
        self.say("Repeat. Without any answer key, EM learns that doctor A is the most reliable. Every se and sp ends\n"
                 "closer to its true value than it started; the true values exist only because the data are synthetic.", wait=0)
        for t in DOC_PLAY[1:]:
            new_table = self.doc_table(t)
            self.play(FadeOut(table), FadeIn(new_table), run_time=0.3)
            table = new_table
            anims = []
            for p2, bar in enumerate(qrow):
                q = d["doc_q"][t][p2]
                tgt = Rectangle(width=cell * 0.7, height=max(0.02, 0.55 * q)).set_stroke(width=0).set_fill(COND, 1)
                tgt.move_to(np.array([x0 + cell * p2, 1.05, 0]), aligned_edge=DOWN)
                anims.append(Transform(bar, tgt))
            self.play(*anims, run_time=0.9 if t <= 5 else 0.6)
            self.wait(0.8)
        self.wait(2)
        self.say(f"By iteration {d['doc_settle']} no estimate moves by more than {SETTLED} per step. "
                 "The log-likelihood of the calls\nrose at every iteration.", wait=5)

        self.mark("doctors: the hidden truth against EM and the majority vote")
        truth = VGroup()
        for p2 in range(DOC_SHOWN):
            sq = Square(side_length=cell * 0.86).move_to(np.array([x0 + cell * p2, 0.55, 0]))
            if d["doc_z"][p2]:
                sq.set_fill(HIDDEN, 1).set_stroke(HIDDEN, 1)
            else:
                sq.set_fill(PANEL, 1).set_stroke(GREY_C, 1)
            truth.add(sq)
        t_lab = T("truth", 20, HIDDEN).move_to(np.array([x0 - 0.95, 0.55, 0]))
        acc = VGroup(
            M(f"EM's answer (q<sub>p</sub> &gt; 0.5) is right for {d['doc_acc_em']} of {DOC_P} patients", 23, GOOD),
            M(f"the majority vote is right for {d['doc_acc_maj']} of {DOC_P}", 23, GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        fit_right(acc, -1.0, left=-6.8, right=-0.4)
        self.say("Now reveal the hidden truth. On this dataset EM, which weights doctor A's calls more heavily,\n"
                 "gets more patients right than a simple majority vote does.", wait=0)
        self.play(LaggedStart(*[FadeIn(s) for s in truth], lag_ratio=0.01), FadeIn(t_lab), run_time=1.2)
        self.play(FadeOut(m_form), run_time=0.4)
        self.play(FadeIn(acc, lag_ratio=0.3))
        self.wait(5)
        self.say(f"Patient {p + 1}, for example, truly has the condition. The majority said no; EM ends at q<sub>p</sub> = {z['q_end']}.\n"
                 "The model assumes the doctors err independently of one another, given the true answer.", wait=8)
        self.end_section()

    def summary(self):
        """
        Description: Shows one table of the three examples: what is measured and hidden, and what each step does.
        Inputs: none
        Outputs: none
        """
        self.mark("summary")
        self.heading("Summary")
        data = (
            ("example", "measured", "hidden", "the E-step splits", "the M-step refits"),
            ("MRI\nsegmentation", "each voxel's\nbrightness", "each voxel's\ntissue", "each voxel among\nthe tissues", "each tissue's\nbrightness curve"),
            ("PET\nreconstruction", "counts on each\ndetector line", "which pixel each\ncount came from", "each line's count\namong its pixels", "each pixel's\nactivity: the image"),
            ("doctors'\ncalls", "each doctor's\nyes or no", "each patient's\ntrue diagnosis", "each patient between\nhas it and does not", "the prevalence π and\neach doctor's se and sp"),
        )
        ys = (2.4, 1.25, -0.15, -1.55)
        cells = [[cell_text(s, 24 if r else 22, GREY_B if r == 0 else (HIDDEN if c == 2 else WHITE))
                  for c, s in enumerate(row)] for r, row in enumerate(data)]
        widths = [max(cells[r][c].width for r in range(len(data))) for c in range(len(data[0]))]
        gap = 0.45
        table = VGroup()
        for r, row in enumerate(cells):
            x = 0.0
            for c, cell in enumerate(row):
                table.add(cell.move_to(np.array([x, ys[r], 0]), aligned_edge=LEFT))
                x += widths[c] + gap
        rule = Line(np.array([-0.1, 1.95, 0]), np.array([x - gap + 0.1, 1.95, 0])).set_stroke(GREY_B, 1.5)
        frame = VGroup(table, rule)
        if frame.width > 13.4:
            frame.scale_to_fit_width(13.4)
        frame.move_to(np.array([0, frame.get_center()[1], 0]))
        self.play(FadeIn(table[:5]), Create(rule))
        self.say("The same two steps run all three examples. Only what is hidden, and what gets split, changes.", wait=0)
        for r in range(1, 4):
            self.play(LaggedStart(*[FadeIn(c, shift=0.1 * RIGHT) for c in table[5 * r:5 * r + 5]], lag_ratio=0.15), run_time=1.2)
            self.wait(3)
        self.wait(3)
        last = M("MRI: the scanner forms the image and EM labels it.   PET: EM can form the image itself.", 25, GOOD)
        last.move_to(np.array([0, -2.55, 0]))
        self.say("In all three, the log-likelihood never decreased, and the code asserts it for every iteration computed.", wait=0)
        self.play(FadeIn(last, shift=0.1 * UP))
        self.wait(9)
        self.end_section()

    def mri_histogram(self):
        """
        Description: Builds the bar histogram of the brain voxels' brightness and its axis labels.
        Inputs: none
        Outputs: a VGroup of bars, a baseline, tick labels and an axis label.
        """
        x = self.d["mri_x"]
        counts, edges = np.histogram(x, bins=np.arange(XMIN, XMAX + BIN, BIN))
        bars = VGroup()
        for c, a in zip(counts, edges[:-1]):
            if c == 0:
                continue
            bars.add(Polygon([self.hx(a), H_BASE, 0], [self.hx(a + BIN), H_BASE, 0], [self.hx(a + BIN), self.hy(c), 0],
                             [self.hx(a), self.hy(c), 0], stroke_width=0.5, stroke_color=GREY_D, fill_color=GREY_C,
                             fill_opacity=0.45))
        axis = Line([H_LEFT, H_BASE, 0], [H_RIGHT, H_BASE, 0], stroke_width=2, color=GREY_B)
        ticks = VGroup(*[T(str(v), 19, GREY_B).move_to([self.hx(v), H_BASE - 0.22, 0]) for v in range(0, 201, 50)])
        label = M("voxel brightness x<sub>v</sub>", 21, GREY_B).move_to([(H_LEFT + H_RIGHT) / 2, H_BASE - 0.55, 0])
        return VGroup(bars, axis, ticks, label)

    def mri_curves(self, params):
        """
        Description: Draws each tissue's weighted bell curve on the histogram's count scale.
        Inputs:
        - params: (fraction, mean, spread) arrays over the three tissues.
        Outputs: a VGroup of three curves.
        """
        g = VGroup()
        xs = np.linspace(XMIN, XMAX, 300)
        top = 1.12 * self.d["mri_ymax"]
        for t, col in enumerate((FLUID, GRAY_M, WHITE_M)):
            ys = tissue_pdf(xs, params, t) * self.d["mri_V"] * BIN
            g.add(VMobject(stroke_color=col, stroke_width=4).set_points_smoothly(
                [[self.hx(a), self.hy(min(b, top)), 0] for a, b in zip(xs, ys)]))
        return g

    def mri_readout(self, t, ll):
        """
        Description: Builds the MRI iteration and log-likelihood readout.
        Inputs:
        - t: the iteration number.
        - ll: the mean log-likelihood per voxel.
        Outputs: the text mobject.
        """
        return M(f"iteration {t}     mean log-likelihood per voxel:  {num(ll, '.3f')}", 22, GOOD).move_to(
            np.array([(H_LEFT + H_RIGHT) / 2, H_TOP + 0.3, 0]))

    def hx(self, v):
        """
        Description: Maps a brightness to its screen x on the MRI histogram.
        Inputs:
        - v: the brightness.
        Outputs: the screen x.
        """
        return H_LEFT + (H_RIGHT - H_LEFT) * (v - XMIN) / (XMAX - XMIN)

    def hy(self, c):
        """
        Description: Maps a voxel count per bin to its screen y on the MRI histogram.
        Inputs:
        - c: the count.
        Outputs: the screen y.
        """
        return H_BASE + (H_TOP - 0.45 - H_BASE) * c / self.d["mri_ymax"]

    def pet_label(self, t, centre, size):
        """
        Description: Builds the label under EM's PET estimate.
        Inputs:
        - t: the iteration number.
        - centre, size: where the estimate is drawn and its side length.
        Outputs: the text mobject.
        """
        return T(f"EM estimate, iteration {t}", 20).move_to(centre + np.array([0, -size / 2 - 0.3, 0]))

    def pet_ll_read(self, t, ll, chart):
        """
        Description: Builds the log-likelihood readout under the PET chart.
        Inputs:
        - t: the iteration number.
        - ll: the Poisson log-likelihood after that iteration.
        - chart: the chart the readout sits under.
        Outputs: the text mobject.
        """
        return M(f"iteration {t}     log-likelihood {num(ll, ',.1f')}", 22, GOOD).next_to(chart.group, DOWN, buff=0.35)

    def doc_table(self, t):
        """
        Description: Builds the table of the doctors' estimated and true rates after iteration t.
        Inputs:
        - t: the iteration number.
        Outputs: the table, a VGroup.
        """
        d = self.d
        pi, se, sp = d["doc_path"][t]
        hdr = ("", "se<sub>d</sub> (EM)", "se<sub>d</sub> (true)", "sp<sub>d</sub> (EM)", "sp<sub>d</sub> (true)")
        rows = [[M(h, 20, GREY_B) for h in hdr]]
        for k, n in enumerate("ABC"):
            rows.append([M(f"doctor {n}", 21), M(f"{se[k]:.2f}", 22, GOOD), M(f"{d['doc_true_se'][k]:.2f}", 22, GREY_B),
                         M(f"{sp[k]:.2f}", 22, GOOD), M(f"{d['doc_true_sp'][k]:.2f}", 22, GREY_B)])
        rows.append([M("π", 22), M(f"{pi:.2f}", 22, GOOD), M(f"{d['doc_true_pi']:.2f}", 22, GREY_B), M("", 20), M("", 20)])
        g = VGroup()
        xs = (0.0, 1.35, 2.65, 3.95, 5.25)
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                cell.move_to(np.array([0.1 + xs[c], 0.05 - 0.4 * r, 0]), aligned_edge=LEFT)
                g.add(cell)
        g.add(M(f"iteration {t}", 22, GOOD).move_to(np.array([3.3, -2.25, 0])))
        return g

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
        Description: Prints the time a section or a beat within it starts, for finding it in the rendered video.
        Inputs:
        - name: the section's or beat's name.
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
        self.cap_text, self.cap_start = text, self.renderer.time
        self.play(*anims, run_time=0.6)
        self.cap = new
        # No narration, so hold long enough to read; a caption that leads into an animation gets
        # less, and read_hold() then tops up the time for the caption and the text drawn with it.
        plain = plain_len(text)
        self.wait(max(wait, plain / READ_RATE) if wait else plain / 45)

    def read_hold(self, *texts):
        """
        Description: Waits until the caption and the other text drawn with it have been on screen long enough to read.
        Inputs:
        - texts: the markup of every other text that appeared with the current caption.
        Outputs: none
        Constraints: reading time is counted from when the caption appeared, at READ_RATE characters a second,
        with at least one more second of hold.
        """
        need = plain_len(self.cap_text, *texts) / READ_RATE
        self.wait(max(1.0, self.cap_start + need - self.renderer.time))


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
            g.add(M(num(v, "g"), tick_size, GREY_B).next_to(p, DOWN, buff=0.14))
        for v in yticks:
            p = self.c2p(xr[0], v)
            g.add(Line(p + 0.07 * LEFT, p + 0.07 * RIGHT).set_stroke(GREY_B, 2))
            g.add(M(num(v, ",.0f"), tick_size, GREY_B).next_to(p, LEFT, buff=0.14))
        self.group = g

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
    Description: Computes every number the video shows and asserts the claims the captions make.
    Inputs: none
    Outputs: a dict holding the three examples' data, EM trajectories and the numbers the captions quote.
    Constraints: raises AssertionError when a claim fails, for example if any log-likelihood ever decreases.
    """
    d = {}

    tissue, x_all = mri_slice()
    inside = tissue >= 0
    x = x_all[inside]
    d["mri_x"], d["mri_V"] = x, int(inside.sum())
    d["mri_ymax"] = float(np.histogram(x, bins=np.arange(XMIN, XMAX + BIN, BIN))[0].max())
    gray = np.clip(x_all, 0, 200) / 200 * 255
    gray[~inside] = 0
    d["mri_gray"] = gray.astype(np.uint8)
    params, path = MRI_INIT, [(MRI_INIT, mri_loglik(x, MRI_INIT))]
    for _ in range(max(MRI_SHOW)):
        params = mri_em_step(x, params)
        path.append((params, mri_loglik(x, params)))
    mri_ll = np.array([p[1] for p in path])
    # Once the fit has settled, successive values agree to float rounding, so "never went down" allows 1e-12.
    assert np.all(np.diff(mri_ll) >= -1e-12), "the MRI log-likelihood went down"
    d["mri_path"] = path
    mu = path[-1][0][1]
    assert np.all(np.abs(mu - np.array(TRUE_MU)) < 3), "the fitted means should be close to the true ones"
    d["mri_mu_text"] = [num(m, ".0f") for m in mu]
    dens = np.stack([tissue_pdf(x, path[-1][0], t) for t in range(3)])
    label = np.argmax(dens, axis=0)
    agree = float((label == tissue[inside]).mean())
    d["mri_agree"] = pct(agree)
    rgb = np.zeros((MRI_G, MRI_G, 3), np.uint8)
    lab_map = np.full((MRI_G, MRI_G), -1)
    lab_map[inside] = label
    for t, col in enumerate((FLUID, GRAY_M, WHITE_M)):
        rgb[lab_map == t] = np.array(color_to_int_rgb(ManimColor(col)))
    d["mri_label_rgb"] = rgb

    A, centres = pet_system_matrix()
    true = pet_phantom()
    d["pet_true"], d["pet_vmax"] = true, float(true.max())
    cov = A.sum(0)
    assert cov.min() > 0, "every pixel must be seen by some line"
    body = true > 0
    assert np.allclose(cov[body], 1.0), "every pixel inside the body should be seen at every angle"
    d["pet_cov"], d["pet_cov_min"] = cov, float(cov.min())
    rng = np.random.default_rng(PET_SEED)
    y = rng.poisson(A @ true)
    d["pet_lines"], d["pet_total"] = len(y), int(y.sum())
    hist, lls = pet_em(A, y, PET_ITERS)
    assert np.all(np.diff(lls) > 0), "the PET log-likelihood should rise at every iteration"
    d["pet_hist"], d["pet_ll"] = hist, lls
    err = np.array([math.sqrt(float(np.mean((h - true) ** 2))) for h in hist])
    d["pet_err"] = err
    best = int(np.argmin(err[:PET_SHOW[-1] + 1]))
    d["pet_best_iter"] = best
    assert all(err[a] > err[b] for a, b in zip(PET_SHOW, PET_SHOW[1:]) if b <= 20), "the error should fall up to iteration 20"
    assert 20 < best < 100 and err[100] > err[best], "the error should bottom out between iterations 20 and 100"
    clean_hist, _ = pet_em(A, A @ true, PET_SHOW[-1])
    clean_err = [math.sqrt(float(np.mean((h - true) ** 2))) for h in clean_hist]
    assert all(b < a for a, b in zip(clean_err, clean_err[1:])), "with noise-free counts the error should keep falling"
    lo = math.floor(lls[0] / 1000) * 1000
    hi = math.ceil(lls[PET_PLAY[-1]] / 500) * 500
    d["pet_ll_range"] = (lo, hi)
    d["pet_ll_ticks"] = tuple(v for v in range(lo, hi + 1, 2000))

    at_angle = PET_ZOOM_ANGLE * PET_G + np.argsort(-y[PET_ZOOM_ANGLE * PET_G:(PET_ZOOM_ANGLE + 1) * PET_G])
    zooms = [z for z in (pet_zoom(A, y, hist, true, centres, int(i)) for i in at_angle[:6]) if z]
    assert zooms, "one of the busiest lines should give shown numbers that add up after rounding"
    zoom = zooms[0]
    assert true[np.argmin([np.abs(c - zoom["hot_xy"]).sum() for c in centres])] == true.max(), \
        "the red pixel should lie in the brightest hot spot"
    d["pet_zoom_line"], d["pet_zoom_y"] = zoom["line"], int(y[zoom["line"]])
    d["pet_zoom"] = zoom
    erng = np.random.default_rng(PET_SEED + 1)
    picks = erng.choice(PET_G * PET_G, 7, p=true / true.sum())
    d["pet_events"] = [(tuple(centres[j][:2] + erng.uniform(-0.4, 0.4, 2)), float(erng.uniform(0, math.pi))) for j in picks]

    z, calls = doctor_calls()
    d["doc_calls"], d["doc_z"] = calls, z
    d["doc_true_pi"] = float(z.mean())
    d["doc_true_se"] = [float(calls[z == 1, k].mean()) for k in range(3)]
    d["doc_true_sp"] = [float(1 - calls[z == 0, k].mean()) for k in range(3)]
    majority = (calls.sum(1) >= 2).astype(float)
    pars, qs, dlls = [ds_mstep(calls, majority)], [], []
    for _ in range(DOC_ITERS):
        ll, q = ds_estep(calls, *pars[-1])
        dlls.append(ll)
        qs.append(q)
        pars.append(ds_mstep(calls, q))
    dlls.append(ds_estep(calls, *pars[-1])[0])
    assert np.all(np.diff(dlls) > 0), "the doctors' log-likelihood should rise at every iteration"
    d["doc_path"], d["doc_q"] = pars, qs + [ds_estep(calls, *pars[-1])[1]]
    moves = [max(abs(a[0] - b[0]), *np.abs(a[1] - b[1]), *np.abs(a[2] - b[2])) for a, b in zip(pars, pars[1:])]
    settle = next(t for t in range(len(moves)) if all(m < SETTLED for m in moves[t:]))
    assert settle < DOC_PLAY[-1], "the estimates should settle before the last iteration shown"
    d["doc_settle"] = settle
    pi_e, se_e, sp_e = pars[-1]
    assert np.all(se_e + sp_e > 1), "every doctor should end better than chance, so the labels are not flipped"
    assert np.argmax(se_e + sp_e) == 0 and np.argmax(np.array(d["doc_true_se"]) + d["doc_true_sp"]) == 0, \
        "EM and the truth should both rank doctor A first"
    assert all(abs(se_e[k] - d["doc_true_se"][k]) < 0.07 and abs(sp_e[k] - d["doc_true_sp"][k]) < 0.07 for k in range(3)), \
        "the final estimates should be within 0.07 of the true rates"
    _, se_0, sp_0 = pars[0]
    assert all(abs(se_e[k] - d["doc_true_se"][k]) < abs(se_0[k] - d["doc_true_se"][k]) and
               abs(sp_e[k] - d["doc_true_sp"][k]) < abs(sp_0[k] - d["doc_true_sp"][k]) for k in range(3)), \
        "every sensitivity and specificity should end closer to its true value than it started"
    q_end = d["doc_q"][-1]
    acc_em, acc_maj = int(((q_end > 0.5) == z).sum()), int((majority == z).sum())
    assert acc_em > acc_maj, "EM should beat the majority vote on this dataset"
    d["doc_acc_em"], d["doc_acc_maj"] = acc_em, acc_maj
    cand = [p for p in range(DOC_SHOWN) if z[p] == 1 and majority[p] == 0 and q_end[p] > 0.5 and calls[p, 0] == 1]
    assert cand, "some shown patient should be one EM gets right and the majority gets wrong"
    p = cand[0]
    pi0, se0, sp0 = pars[0]
    a = pi0 * np.prod(np.where(calls[p] == 1, se0, 1 - se0))
    b = (1 - pi0) * np.prod(np.where(calls[p] == 1, 1 - sp0, sp0))
    a4, b4 = round(float(a), 4), round(float(b), 4)
    assert f"{a4 / (a4 + b4):.2f}" == f"{a / (a + b):.2f}", "rounding must not change the shown chance"
    assert abs(qs[0][p] - a / (a + b)) < 1e-12
    assert list(calls[p]) == [1, 0, 0], "the zoomed patient's calls must match the formula on screen"
    d["doc_zoom"] = {"p": p, "a0": f"{a4:.4f}", "b0": f"{b4:.4f}", "q0": f"{a / (a + b):.2f}", "q_end": f"{q_end[p]:.2f}"}
    d["doc_survey"] = doc_seed_survey()

    for k, v in d.items():
        if np.isscalar(v) or isinstance(v, (str, tuple)):
            print(f"  {k}: {v}".encode("ascii", "replace").decode())  # Windows consoles may not be UTF-8
    print(f"  doc final: pi {pi_e:.3f} se {np.round(se_e, 3)} sp {np.round(sp_e, 3)}")
    print(f"  doc seed survey (EM better, tied, worse than the majority vote): {d['doc_survey']}")
    print(f"  pet ll shown: {[round(lls[t], 1) for t in PET_PLAY]}  err: {[round(err[t], 1) for t in PET_SHOW]}")
    return d


def mri_slice():
    """
    Description: Draws the synthetic MRI slice: a tissue map with folded gray matter and two fluid-filled ventricles.
    Inputs: none
    Outputs: (tissue, brightness), two MRI_G x MRI_G arrays; tissue is -1 outside the brain, else 0, 1 or 2.
    """
    c = np.arange(MRI_G) - (MRI_G - 1) / 2
    X, Y = np.meshgrid(c, -c)
    r = np.hypot(X / 28, Y / 24)
    folds = 1 + 0.035 * np.sin(9 * np.arctan2(Y, X))
    tissue = np.full((MRI_G, MRI_G), -1)
    brain = r <= 1
    tissue[brain] = 0
    tissue[brain & (r <= 0.94)] = 1
    tissue[brain & (r <= 0.70 * folds)] = 2
    vent = (((X + 4) / 3) ** 2 + ((Y - 2) / 7) ** 2 <= 1) | (((X - 4) / 3) ** 2 + ((Y - 2) / 7) ** 2 <= 1)
    tissue[vent & brain] = 0
    rng = np.random.default_rng(MRI_SEED)
    x = np.zeros((MRI_G, MRI_G))
    for t in range(3):
        m = tissue == t
        x[m] = rng.normal(TRUE_MU[t], TRUE_SD[t], int(m.sum()))
    return tissue, x


def mri_em_step(x, params):
    """
    Description: Runs one EM iteration for the three-tissue mixture.
    Inputs:
    - x: the brain voxels' brightness.
    - params: the current (fraction, mean, spread) arrays.
    Outputs: the updated (fraction, mean, spread) arrays.
    """
    dens = np.stack([tissue_pdf(x, params, t) for t in range(3)])
    share = dens / dens.sum(axis=0)
    total = share.sum(axis=1)
    mu = (share * x).sum(axis=1) / total
    sd = np.sqrt((share * (x - mu[:, None]) ** 2).sum(axis=1) / total)
    return (total / len(x), mu, sd)


def mri_loglik(x, params):
    """
    Description: Computes the mean log-likelihood per voxel under the three-tissue mixture.
    Inputs:
    - x: the brain voxels' brightness.
    - params: (fraction, mean, spread) arrays.
    Outputs: the mean log-likelihood.
    """
    return float(np.mean(np.log(sum(tissue_pdf(x, params, t) for t in range(3)))))


def tissue_pdf(x, params, t):
    """
    Description: Evaluates tissue t's fraction times its bell-curve density.
    Inputs:
    - x: brightness values.
    - params: (fraction, mean, spread) arrays.
    - t: the tissue index.
    Outputs: an array the shape of x.
    """
    w, mu, sd = params
    return w[t] * np.exp(-0.5 * ((x - mu[t]) / sd[t]) ** 2) / (sd[t] * math.sqrt(2 * math.pi))


def pet_phantom():
    """
    Description: Builds the synthetic PET activity: an elliptical body with three hot spots of different strengths.
    Inputs: none
    Outputs: the activity of each pixel, flattened row by row from the top-left corner.
    """
    c = np.arange(PET_G) - (PET_G - 1) / 2
    X, Y = np.meshgrid(c, -c)
    img = np.zeros((PET_G, PET_G))
    img[(X / 6.6) ** 2 + (Y / 5.6) ** 2 <= 1] = 30
    img[(X + 2.5) ** 2 + (Y - 1.5) ** 2 <= 2.2 ** 2] = 160
    img[(X - 3.0) ** 2 + (Y + 1.0) ** 2 <= 1.3 ** 2] = 110
    img[(X - 0.5) ** 2 + (Y + 3.2) ** 2 <= 1.0 ** 2] = 70
    return (img * PET_SCALE).ravel()


def pet_system_matrix():
    """
    Description: Builds a_ij, the chance that an emission from pixel j is counted on detector line i.
    Inputs: none
    Outputs: (A, centres): A has one row per line (angle-major) and one column per pixel; centres holds each
    pixel's centre in pixel units, with x to the right and y up.
    Constraints: lines are strips one pixel wide; an emission's line lies at each angle with chance 1/PET_ANGLES.
    """
    c = np.arange(PET_G) - (PET_G - 1) / 2
    X, Y = np.meshgrid(c, -c)
    xs, ys = X.ravel(), Y.ravel()
    off = (np.arange(PET_SUB) + 0.5) / PET_SUB - 0.5
    A = np.zeros((PET_ANGLES * PET_G, PET_G * PET_G))
    for a in range(PET_ANGLES):
        th = a * math.pi / PET_ANGLES
        for dx in off:
            for dy in off:
                s = (xs + dx) * math.cos(th) + (ys + dy) * math.sin(th)
                b = np.floor(s + PET_G / 2).astype(int)
                ok = (b >= 0) & (b < PET_G)
                A[a * PET_G + b[ok], np.nonzero(ok)[0]] += 1 / PET_SUB ** 2
    centres = [np.array([xs[j], ys[j], 0.0]) for j in range(PET_G * PET_G)]
    return A / PET_ANGLES, centres


def pet_em(A, y, iters):
    """
    Description: Runs ML-EM from a flat image and records every iterate and its log-likelihood.
    Inputs:
    - A: the system matrix a_ij.
    - y: the counts on each line.
    - iters: how many iterations to run.
    Outputs: (images, lls): the image after each iteration, starting with the flat start, and the Poisson
    log-likelihood of each.
    """
    cov = A.sum(0)
    lam = np.full(A.shape[1], y.sum() / cov.sum())
    out, lls = [lam], [pet_loglik(A, y, lam)]
    for _ in range(iters):
        lam = lam / cov * (A.T @ (y / (A @ lam)))
        out.append(lam)
        lls.append(pet_loglik(A, y, lam))
    return out, np.array(lls)


def pet_loglik(A, y, lam):
    """
    Description: Computes the Poisson log-likelihood of the counts, including the log y_i! term.
    Inputs:
    - A: the system matrix.
    - y: the counts.
    - lam: the image.
    Outputs: the log-likelihood.
    """
    p = A @ lam
    return float((y * np.log(p) - p).sum() - sum(math.lgamma(v + 1) for v in y))


def pet_zoom(A, y, hist, true, centres, i):
    """
    Description: Picks the pixels and numbers for the close-up of detector line i.
    Inputs:
    - A, y: the system matrix and the counts.
    - hist: the ML-EM images, starting with the flat start.
    - true: the true activity.
    - centres: each pixel's centre in pixel units.
    - i: the line to examine.
    Outputs: a dict of the shown pixels and numbers, or None when rounding would change a shown sum or ratio.
    """
    row = A[i]
    frac = row * PET_ANGLES
    shows_ok = [j for j in range(len(row)) if f"{round(float(frac[j]), 2) / PET_ANGLES:.4f}" == f"{row[j]:.4f}"]
    big = [j for j in shows_ok if frac[j] > 0.6]
    small = [j for j in shows_ok if 0.1 < frac[j] < 0.35]
    if not big or not small:
        return None
    big_j, small_j = max(big, key=lambda j: frac[j]), min(small)
    shares1 = y[i] * row * hist[1] / (row @ hist[1])
    on = np.nonzero(frac > 0.6)[0]
    hot_j = int(on[np.argmax(shares1[on])])
    floor = true[true > 0].min()
    bg_on = [j for j in on if 0 < true[j] <= floor]
    bg_j = int(min(bg_on, key=lambda j: abs(row[j] - row[hot_j])))
    if true[hot_j] != true.max():
        return None
    z = {"line": i, "hot_xy": centres[hot_j], "bg_xy": centres[bg_j], "big_xy": centres[big_j],
         "small_xy": centres[small_j], "a_big": (f"{frac[big_j]:.2f}", f"{row[big_j]:.4f}"),
         "a_small": (f"{frac[small_j]:.2f}", f"{row[small_j]:.4f}")}
    for key, lam in (("flat", hist[0]), ("iter1", hist[1])):
        sh = y[i] * row * lam / (row @ lam)
        assert abs(float(sh.sum()) - y[i]) < 1e-9, "the shares must add back to the line's count"
        h, b = round(float(sh[hot_j]), 1), round(float(sh[bg_j]), 1)
        other = round(float(y[i]) - h - b, 1)  # the shown parts then add up to the shown total exactly
        if f"{other:.1f}" != f"{float(y[i] - sh[hot_j] - sh[bg_j]):.1f}":
            return None
        z[key] = (f"{h:.1f}", f"{b:.1f}", f"{other:.1f}", f"{y[i]}")
    if float(z["iter1"][0]) <= float(z["flat"][0]):
        return None
    return z


def doc_seed_survey():
    """
    Description: Runs the doctors' example on seeds 0 to DOC_SURVEY - 1 and compares EM's answers with the majority vote.
    Inputs: none
    Outputs: (better, tied, worse): how many seeds EM gets more, as many, or fewer patients right than the vote.
    """
    tally = [0, 0, 0]
    for seed in range(DOC_SURVEY):
        z, calls = doctor_calls(seed)
        majority = (calls.sum(1) >= 2).astype(float)
        pars = ds_mstep(calls, majority)
        for _ in range(DOC_ITERS):
            pars = ds_mstep(calls, ds_estep(calls, *pars)[1])
        q = ds_estep(calls, *pars)[1]
        em, vote = int(((q > 0.5) == z).sum()), int((majority == z).sum())
        tally[0 if em > vote else 1 if em == vote else 2] += 1
    return tuple(tally)


def doctor_calls(seed=DOC_SEED):
    """
    Description: Draws the synthetic patients' true status and the three doctors' yes/no calls.
    Inputs:
    - seed: the random seed; the video uses DOC_SEED.
    Outputs: (z, calls): z[p] is 1 when patient p has the condition; calls[p, d] is doctor d's call.
    Constraints: each doctor's calls are independent of the others' once the true status is fixed.
    """
    rng = np.random.default_rng(seed)
    z = (rng.random(DOC_P) < DOC_PREV).astype(int)
    u = rng.random((DOC_P, 3))
    calls = np.where(z[:, None] == 1, u < np.array(DOC_SE), u >= np.array(DOC_SP)).astype(int)
    return z, calls


def ds_estep(calls, pi, se, sp):
    """
    Description: Runs Dawid and Skene's E-step for two categories: each patient's chance of having the condition.
    Inputs:
    - calls: the patients x doctors array of calls.
    - pi, se, sp: the prevalence and each doctor's sensitivity and specificity.
    Outputs: (ll, q): the log-likelihood of all calls, and q[p], the chance patient p has the condition.
    """
    has = pi * np.prod(np.where(calls == 1, se, 1 - se), 1)
    not_ = (1 - pi) * np.prod(np.where(calls == 1, 1 - sp, sp), 1)
    return float(np.log(has + not_).sum()), has / (has + not_)


def ds_mstep(calls, q):
    """
    Description: Runs Dawid and Skene's M-step: refits the prevalence and each doctor's rates from soft answers.
    Inputs:
    - calls: the patients x doctors array of calls.
    - q: each patient's chance of having the condition.
    Outputs: (pi, se, sp).
    """
    se = (q[:, None] * calls).sum(0) / q.sum()
    sp = ((1 - q)[:, None] * (1 - calls)).sum(0) / (1 - q).sum()
    return float(q.mean()), se, sp


def strip_corners(angle, b, centre, size):
    """
    Description: Finds the corners of detector line b at an angle, clipped to the image square on screen.
    Inputs:
    - angle: the angle index.
    - b: the line's position at that angle, 0 to PET_G - 1.
    - centre, size: the image's screen centre and side length.
    Outputs: the clipped polygon's corners as screen points.
    """
    th = angle * math.pi / PET_ANGLES
    n = np.array([math.cos(th), math.sin(th)])
    t = np.array([-n[1], n[0]])
    s0, s1 = b - PET_G / 2, b + 1 - PET_G / 2
    L = PET_G
    quad = [s0 * n - L * t, s1 * n - L * t, s1 * n + L * t, s0 * n + L * t]
    h = PET_G / 2
    poly = clip_polygon(quad, [(-h, -h), (h, -h), (h, h), (-h, h)])
    return [centre + np.array([p[0], p[1], 0]) * size / PET_G for p in poly]


def fit_right(group, y, left=-1.6, right=6.85):
    """
    Description: Places a text block in the right part of the frame, shrinking it if it is too wide.
    Inputs:
    - group: the block to place.
    - y: the screen height of the block's centre.
    - left, right: the screen x range the block may use.
    Outputs: the same block, moved.
    """
    if group.width > right - left:
        group.scale_to_fit_width(right - left)
    group.move_to(np.array([left + group.width / 2, y, 0]))
    return group


def clip_polygon(poly, box):
    """
    Description: Clips a polygon to a convex polygon (Sutherland and Hodgman's method).
    Inputs:
    - poly: the polygon's corners, as 2-D points.
    - box: the convex clipping polygon's corners, counter-clockwise.
    Outputs: the clipped polygon's corners.
    """
    out = [np.asarray(p, float) for p in poly]
    for k in range(len(box)):
        a, b = np.asarray(box[k], float), np.asarray(box[(k + 1) % len(box)], float)
        inp, out = out, []
        for i in range(len(inp)):
            p, q = inp[i], inp[(i + 1) % len(inp)]
            sp, sq = edge_side(a, b, p), edge_side(a, b, q)
            if sp >= 0:
                out.append(p)
            if sp * sq < 0:
                out.append(p + (q - p) * sp / (sp - sq))
    return out


def edge_side(a, b, p):
    """
    Description: Tells which side of the directed edge from a to b a point lies on.
    Inputs:
    - a, b: the edge's start and end, as 2-D points.
    - p: the point to test.
    Outputs: a positive number when p is to the left of the edge, negative when to the right, 0 on it.
    """
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def image_mob(rgb, height):
    """
    Description: Draws an RGB array as an image with sharp pixel edges.
    Inputs:
    - rgb: an H x W x 3 uint8 array.
    - height: the image's height on screen.
    Outputs: the image mobject.
    """
    im = ImageMobject(np.ascontiguousarray(rgb, dtype=np.uint8))
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    im.height = height
    return im


def heat(img, vmax, gray=False):
    """
    Description: Colours an activity image from black through purple and orange to pale yellow.
    Inputs:
    - img: the 2-D array to colour.
    - vmax: the value drawn as the brightest colour; larger values are clipped.
    - gray: True for a plain grey ramp instead.
    Outputs: an H x W x 3 uint8 array.
    """
    v = np.clip(img / vmax, 0, 1)
    if gray:
        g = (40 + 215 * v).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)
    stops = np.array([[0, 0, 0], [70, 20, 110], [200, 60, 70], [250, 160, 40], [252, 240, 170]], float)
    pos = v * (len(stops) - 1)
    lo = np.minimum(pos.astype(int), len(stops) - 2)
    f = (pos - lo)[..., None]
    return (stops[lo] * (1 - f) + stops[lo + 1] * f).astype(np.uint8)


def symbol_card(title, body, colour, width):
    """
    Description: Draws a titled card with a few lines of plain text.
    Inputs:
    - title: the card's title.
    - body: the text, with line breaks.
    - colour: the title and border colour.
    - width: the card's width.
    Outputs: the card, a VGroup.
    """
    t = T(title, 28, colour)
    b = T(body, 24, WHITE)
    if b.width > width - 0.5:
        b.scale_to_fit_width(width - 0.5)
    inner = VGroup(t, b).arrange(DOWN, buff=0.35)
    rect = RoundedRectangle(corner_radius=0.12, width=width, height=inner.height + 0.7)
    rect.set_stroke(colour, 2).set_fill(PANEL, 1)
    return VGroup(rect, inner.move_to(rect))


def cell_text(s, size, colour):
    """
    Description: Draws one left-aligned table cell, which may span several lines.
    Inputs:
    - s: the text, with line breaks.
    - size: the font size.
    - colour: the text colour.
    Outputs: the cell, a VGroup of lines.
    """
    return VGroup(*[T(line, size, colour) for line in s.split("\n")]).arrange(DOWN, aligned_edge=LEFT, buff=0.08)


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


def T(text, size=26, color=WHITE, markup=False):
    """
    Description: Builds plain text in the serif font.
    Inputs:
    - text: the text, or Pango markup when markup is True.
    - size: the font size.
    - color: the colour.
    - markup: whether text is Pango markup.
    Outputs: the text mobject.
    """
    if markup:
        return MarkupText(text, font=SERIF, font_size=size, color=color)
    return Text(text, font=SERIF, font_size=size, color=color, line_spacing=1.1)


def plain_len(*texts):
    """
    Description: Counts the characters a viewer reads in Pango markup, ignoring tags and entities.
    Inputs:
    - texts: the markup strings.
    Outputs: the total number of visible characters.
    """
    total = 0
    for t in texts:
        total += len(re.sub(r"&[a-z]+;", "x", re.sub(r"<[^>]+>", "", t)))
    return total


def num(v, spec):
    """
    Description: Formats a number for display with a true minus sign.
    Inputs:
    - v: the number.
    - spec: a format spec, such as ".2f".
    Outputs: the formatted string.
    """
    return format(v, spec).replace("-", "−")


def pct(v):
    """
    Description: Formats a fraction as a percentage with one decimal.
    Inputs:
    - v: the fraction, between 0 and 1.
    Outputs: the formatted string, such as "98.7%".
    """
    return f"{100 * v:.1f}%"
