"""GELU and SiLU in depth, then the activation functions that recent LLMs adopted (2024 to 2026), as a Manim explainer.

Render (1080p30), from this folder:
    manim --resolution 1920,1080 --frame_rate 30 02_gelu_silu_and_recent_activations_video.py RecentActivations
Without Manim installed, prefix the command with `uvx --from manim`.
"""
from manim import *
import math
import numpy as np

SERIF = "Georgia"
MONO = "Consolas"
MATH = "Cambria"  # Georgia and Consolas have no ⊙ or arrows; Cambria does.
CURVE = "#58C4DD"
ACCENT = "#FFD166"
BAD = "#FC6255"
GOOD = "#83C167"
PURPLE = "#B48EDE"
FAINT = "#33404c"
config.background_color = "#0b0d10"

OSS_ALPHA = 1.702  # gpt-oss swiglu(): alpha
OSS_LIMIT = 7.0  # gpt-oss swiglu(): limit
XIELU_AP, XIELU_AN, XIELU_BETA = 0.8, 0.8, 0.5  # xIELU initial values (alpha_p, alpha_n, beta)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def relu(x):
    return max(0.0, x)


def relu2(x):
    return max(0.0, x) ** 2


def gelu(x):
    return x * Phi(x)


def gelu_tanh(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def gelu_sigmoid(x):
    return x * sigmoid(1.702 * x)


def silu(x):
    return x * sigmoid(x)


def swish(beta):
    return lambda x: x * sigmoid(beta * x)


def oss_gate(g):
    g = min(g, OSS_LIMIT)
    return g * sigmoid(OSS_ALPHA * g)


def xielu(x):
    if x > 0:
        return XIELU_AP * x * x + XIELU_BETA * x
    return (math.expm1(min(x, -1e-6)) - x) * XIELU_AN + XIELU_BETA * x


class RecentActivations(Scene):
    def construct(self):
        self.cap = None
        self.head = None
        self.intro()
        self.gelu_gate()
        self.gelu_approximations()
        self.swish_beta()
        self.dips()
        self.part_two()
        self.squared_relu()
        self.clamped_swiglu()
        self.xielu_section()
        self.attention_gate()
        self.summary()

    def intro(self):
        self.mark("intro")
        ax = make_axes((-4, 3), (-0.6, 3), 8.4, 4.0, 0.4 * UP)
        faint = VGroup(*[graph(ax, f, -4, b).set_stroke(c, 3, opacity=0.35)
                         for f, b, c in ((gelu, 3, CURVE), (silu, 3, ACCENT), (relu2, 1.7, GOOD), (xielu, 1.6, PURPLE))])
        self.play(Create(ax), LaggedStart(*[Create(g) for g in faint], lag_ratio=0.25), run_time=2.5)
        title = Text("GELU, SiLU and what came after", font=SERIF, font_size=50)
        sub = Text("activation functions, part 2: from 2016 to 2026", font=SERIF, font_size=30, color=GREY_B)
        sub.next_to(title, DOWN, buff=0.3)
        words = VGroup(title, sub).move_to(0.4 * UP)
        plate = BackgroundRectangle(words, color=config.background_color, fill_opacity=0.85, buff=0.35)
        self.play(FadeIn(plate), Write(title), FadeIn(sub, shift=0.2 * UP), run_time=1.6)
        self.wait(2.5)
        self.play(FadeOut(VGroup(plate, words, ax, faint)))

    def gelu_gate(self):
        self.mark("gelu_gate")
        self.heading("GELU   x · Φ(x)")
        ax = make_axes((-3, 3), (-0.5, 3), 6.4, 3.9, np.array([-3.3, 0.55, 0]),
                       xticks=(-3, -2, -1, 1, 2, 3), yticks=(1, 2, 3))
        phi = graph(ax, Phi, -3, 3).set_stroke(PURPLE, 4)
        g = graph(ax, gelu, -3, 3).set_stroke(CURVE, 5)
        legend = VGroup(legend_row("Φ(x)", PURPLE), legend_row("x · Φ(x)", CURVE))
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(ax.c2p(-1.8, 2.4))
        rule = VGroup(
            Text("Hendrycks and Gimpel (2016)", font=SERIF, font_size=26, color=GREY_B),
            Text("multiply x by a random mask m", font=MATH, font_size=26),
            Text("m = 1 with probability Φ(x), else 0", font=MATH, font_size=26),
            Text("E[m x] = Φ(x) · x + (1 − Φ(x)) · 0", font=MATH, font_size=26),
            Text("= x Φ(x)", font=MATH, font_size=30, color=CURVE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.24).move_to(np.array([3.6, 1.2, 0]))
        self.say("Hendrycks and Gimpel built GELU from a random gate. Multiply the input x\n"
                 "by a mask that is 1 with probability Φ(x), the Gaussian CDF, and 0 otherwise.", wait=0)
        self.play(Create(ax), run_time=0.8)
        self.play(Create(phi), FadeIn(legend[0]), FadeIn(rule[:3], lag_ratio=0.3), run_time=2)
        self.wait(3)
        self.say("Averaged over the mask, the output is x Φ(x). That average is GELU,\n"
                 "a fixed curve with no randomness left in it.", wait=0)
        self.play(FadeIn(rule[3]))
        self.play(FadeIn(rule[4]), Create(g), FadeIn(legend[1]), run_time=1.6)
        self.wait(3)
        t = ValueTracker(-2.5)
        guide = always_redraw(lambda: DashedLine(ax.c2p(t.get_value(), -0.5), ax.c2p(t.get_value(), 3),
                                                 dash_length=0.08).set_stroke(GREY_B, 2, opacity=0.6))
        pdot = always_redraw(lambda: Dot(ax.c2p(t.get_value(), Phi(t.get_value())), radius=0.08, color=PURPLE))
        gdot = always_redraw(lambda: Dot(ax.c2p(t.get_value(), gelu(t.get_value())), radius=0.08, color=CURVE))
        readout = always_redraw(lambda: Text(
            f"x = {num(t.get_value(), '+.2f')}    Φ(x) = {num(Phi(t.get_value()), '.2f')}    x Φ(x) = {num(gelu(t.get_value()), '+.2f')}",
            font=MATH, font_size=24, color=ACCENT).move_to(np.array([3.6, -1.1, 0])))
        k = Phi(0.5)
        self.say(f"At x = 0.5 the mask keeps the input {num(100 * k, '.0f')}% of the time, so GELU(0.5) ≈ {num(gelu(0.5), '.2f')}.\n"
                 "Large inputs are almost always kept; very negative ones are almost always dropped.", wait=0)
        self.add(guide, pdot, gdot, readout)
        self.play(t.animate.set_value(2.5), run_time=4, rate_func=smooth)
        self.play(t.animate.set_value(0.5), run_time=1.5)
        self.wait(3.5)
        self.clear_stage()

    def gelu_approximations(self):
        self.mark("approximations")
        self.heading("Two cheaper forms of GELU")
        e_tanh = max_gap(gelu_tanh, gelu, -6, 6)
        e_sig = max_gap(gelu_sigmoid, gelu, -6, 6)
        forms = VGroup(
            VGroup(Text("exact", font=SERIF, font_size=24, color=GREY_B),
                   Text("0.5 x (1 + erf(x / √2))", font=MATH, font_size=28, color=CURVE)),
            VGroup(Text("tanh form", font=SERIF, font_size=24, color=GREY_B),
                   Text("0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))", font=MATH, font_size=28, color=TEAL)),
            VGroup(Text("sigmoid form", font=SERIF, font_size=24, color=GREY_B),
                   Text("x σ(1.702 x)", font=MATH, font_size=28, color=ACCENT)),
        )
        for f in forms:
            f.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        forms.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(np.array([-3.6, 1.15, 0]))
        if forms.width > 6.2:
            forms.scale_to_fit_width(6.2).move_to(np.array([-3.6, 1.15, 0]))
        ax = make_axes((-4, 4), (-0.025, 0.025), 5.6, 3.4, np.array([3.6, 0.9, 0]),
                       xticks=(-4, -2, 2, 4), yticks=(-0.02, 0.02))
        title = Text("approximation − exact", font=MATH, font_size=24, color=GREY_B).next_to(ax, UP, buff=0.15)
        es = graph(ax, lambda x: gelu_sigmoid(x) - gelu(x), -4, 4).set_stroke(ACCENT, 4)
        et = graph(ax, lambda x: gelu_tanh(x) - gelu(x), -4, 4).set_stroke(TEAL, 4)
        self.say("The exact GELU needs the error function. The same paper gives two cheaper\n"
                 "forms, one built on tanh and one on the logistic sigmoid σ.", wait=0)
        self.play(FadeIn(forms, lag_ratio=0.3), run_time=2)
        self.wait(3.5)
        self.say(f"Over [−6, 6] the tanh form is never off by more than {num(e_tanh, '.4f')}.\n"
                 f"The sigmoid form is cruder: it is off by up to {num(e_sig, '.3f')}.", wait=0)
        self.play(Create(ax), FadeIn(title), run_time=0.8)
        self.play(Create(et), Create(es), run_time=2)
        gaps = VGroup(Text(f"largest gap {num(e_tanh, '.4f')}", font=MATH, font_size=22, color=TEAL),
                      Text(f"largest gap {num(e_sig, '.3f')}", font=MATH, font_size=22, color=ACCENT))
        gaps.arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(ax, DOWN, buff=0.55)
        self.play(FadeIn(gaps))
        self.wait(3.5)
        code = Text("def gelu(x): return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))",
                    font=MONO, font_size=20, color=GREY_A)
        if code.width > 12.6:
            code.scale_to_fit_width(12.6)
        code.move_to(np.array([0, -1.75, 0]))
        tag = Text("openai/gpt-2, src/model.py", font=SERIF, font_size=20, color=GREY_B).next_to(code, UP, buff=0.12)
        tag.align_to(code, LEFT)
        self.play(FadeOut(gaps))
        self.say("GPT-2's released code uses the tanh form, written out in full.\n"
                 "The sigmoid form will come back in part 2.", wait=0)
        self.play(FadeIn(tag), FadeIn(code, shift=0.1 * UP))
        self.wait(4)
        self.clear_stage()

    def swish_beta(self):
        self.mark("swish_beta")
        self.heading("Swish-β   x · σ(βx)")
        sx, sy = minimum(silu, -4, 0)
        ax = make_axes((-4, 3), (-0.6, 3), 7.2, 3.9, np.array([-2.7, 0.55, 0]),
                       xticks=(-4, -3, -2, -1, 1, 2, 3), yticks=(1, 2, 3))
        r = graph(ax, relu, -4, 3).set_stroke(BAD, 3, opacity=0.45)
        gd = DashedVMobject(graph(ax, gelu, -4, 3), num_dashes=70).set_stroke(CURVE, 3).set_z_index(2)
        b = ValueTracker(0.0)
        live = always_redraw(lambda: graph(ax, swish(b.get_value()), -4, 3).set_stroke(ACCENT, 5))
        readout = always_redraw(lambda: Text(f"β = {num(b.get_value(), '.3f')}", font=MATH, font_size=40, color=ACCENT)
                                .move_to(np.array([3.3, 2.3, 0]), aligned_edge=LEFT))
        legend = VGroup(legend_row("x · σ(βx)", ACCENT), legend_row("GELU, dashed", CURVE),
                        legend_row("ReLU", BAD)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        legend.move_to(np.array([3.3, 1.0, 0]), aligned_edge=LEFT)
        self.say("Swish puts a knob β inside the sigmoid. At β = 0 the gate is a constant 1/2,\n"
                 "so the curve is the straight line x/2.", wait=0)
        self.play(Create(ax), FadeIn(r), Create(gd), run_time=1.2)
        self.add(live)
        self.play(FadeIn(readout), FadeIn(legend))
        self.wait(3.5)
        self.say("At β = 1 it is SiLU, the curve from video 1.", wait=0)
        self.play(b.animate.set_value(1.0), run_time=2)
        self.wait(1.5)
        dip = always_redraw(lambda: Dot(ax.c2p(sx / b.get_value(), sy / b.get_value()), radius=0.08, color=WHITE))
        diptext = always_redraw(lambda: Text(
            f"dip {num(sy / b.get_value(), '+.3f')} at x = {num(sx / b.get_value(), '+.3f')}", font=MATH, font_size=24)
            .move_to(np.array([3.3, -0.3, 0]), aligned_edge=LEFT))
        self.say("At β = 1.702 it lies almost on top of GELU. That is GELU's sigmoid form,\n"
                 "x σ(1.702x), and 1.702 is the number that returns in part 2.", wait=0)
        self.add(dip)
        self.play(FadeIn(diptext))
        self.play(b.animate.set_value(1.702), run_time=2)
        self.wait(3.5)
        self.say("As β grows, the sigmoid sharpens into a step,\n"
                 "and the curve closes in on ReLU.", wait=0)
        self.play(b.animate.set_value(20.0), run_time=4, rate_func=rate_functions.ease_in_quad)
        self.wait(2)
        ident = Text("x σ(βx) = SiLU(βx) / β", font=MATH, font_size=28, color=WHITE)
        ident.move_to(np.array([3.3, -1.0, 0]), aligned_edge=LEFT)
        if ident.get_right()[0] > 6.8:
            ident.shift((ident.get_right()[0] - 6.8) * LEFT)
        self.say(f"Every member is SiLU rescaled, so the dip sits at x = {num(sx, '.3f')}/β with depth\n"
                 f"{num(sy, '.3f')}/β. The dip shrinks as 1/β toward ReLU's corner.", wait=0)
        self.play(FadeIn(ident))
        self.play(b.animate.set_value(1.0), run_time=3)
        self.wait(3)
        self.remove(live, dip, readout, diptext)
        self.clear_stage()

    def dips(self):
        self.mark("dips")
        self.heading("Neither curve is monotonic")
        gx, gy = minimum(gelu, -4, 0)
        sx, sy = minimum(silu, -4, 0)
        ax = make_axes((-5, 1), (-0.4, 1), 7.0, 3.9, np.array([-2.9, 0.55, 0]),
                       xticks=(-5, -4, -3, -2, -1, 1), yticks=(0.5, 1))
        g = graph(ax, gelu, -5, 1).set_stroke(CURVE, 5)
        s = graph(ax, silu, -5, 1).set_stroke(ACCENT, 5)
        legend = VGroup(legend_row("GELU", CURVE), legend_row("SiLU", ACCENT)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        legend.move_to(ax.c2p(-3.9, 0.8))
        gd = Dot(ax.c2p(gx, gy), radius=0.08, color=CURVE)
        sd = Dot(ax.c2p(sx, sy), radius=0.08, color=ACCENT)
        self.say("Look closely at the negative side. Both curves dip below zero\n"
                 "and then come back up toward it.", wait=0)
        self.play(Create(ax), run_time=0.8)
        self.play(Create(g), Create(s), FadeIn(legend), run_time=2)
        self.wait(3)
        facts = VGroup(
            Text("GELU minimum", font=SERIF, font_size=24, color=CURVE),
            Text(f"{num(gy, '.4f')} at x = {num(gx, '.4f')}", font=MATH, font_size=26),
            Text("SiLU minimum", font=SERIF, font_size=24, color=ACCENT),
            Text(f"{num(sy, '.4f')} at x = {num(sx, '.4f')}", font=MATH, font_size=26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        facts[2].shift(0.2 * DOWN)
        facts[3].shift(0.2 * DOWN)
        facts.move_to(np.array([1.6, 1.6, 0]), aligned_edge=UL)
        self.say(f"Found numerically, GELU bottoms out at {num(gy, '.3f')} near x = {num(gx, '.2f')},\n"
                 f"and SiLU at {num(sy, '.3f')} near x = {num(sx, '.2f')}.", wait=0)
        self.play(FadeIn(gd), FadeIn(sd), FadeIn(facts, lag_ratio=0.2), run_time=1.5)
        self.wait(3.5)
        p = Dot(ax.c2p(-3, silu(-3)), radius=0.08, color=WHITE)
        cmp = VGroup(Text(f"SiLU(−3) = {num(silu(-3), '.3f')}", font=MATH, font_size=26),
                     Text(f"SiLU({num(sx, '.2f')}) = {num(silu(sx), '.3f')}", font=MATH, font_size=26),
                     Text("a smaller input, a larger output", font=SERIF, font_size=24, color=GREY_B))
        cmp.arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(facts, DOWN, buff=0.5).align_to(facts, LEFT)
        self.say("Left of the minimum the slope is negative, so the output rises as the input falls.\n"
                 "GELU and SiLU are not monotonic, unlike ReLU, sigmoid and tanh.", wait=0)
        self.play(FadeIn(p), FadeIn(cmp, lag_ratio=0.3), run_time=1.5)
        self.wait(5)
        self.clear_stage()

    def part_two(self):
        self.mark("part_two")
        if self.head is not None:
            self.play(FadeOut(self.head), FadeOut(self.cap), run_time=0.6)
            self.head, self.cap = None, None
        card = VGroup(Text("Part 2", font=SERIF, font_size=48, color=ACCENT),
                      Text("activation functions in recent LLMs, 2024 to 2026", font=SERIF, font_size=32))
        card.arrange(DOWN, buff=0.3)
        self.play(FadeIn(card, shift=0.2 * UP), run_time=1)
        self.wait(2.5)
        self.play(FadeOut(card))

    def squared_relu(self):
        self.mark("squared_relu")
        self.heading("Squared ReLU   max(0, x)²")
        ax = make_axes((-3, 2), (-0.5, 4), 5.4, 3.9, np.array([-3.9, 0.55, 0]),
                       xticks=(-3, -2, -1, 1, 2), yticks=(1, 2, 3, 4))
        r = graph(ax, relu, -3, 2).set_stroke(BAD, 3, opacity=0.5)
        q = graph(ax, relu2, -3, 2).set_stroke(GOOD, 5)
        legend = VGroup(legend_row("max(0, x)²", GOOD), legend_row("ReLU", BAD)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        legend.move_to(ax.c2p(-1.7, 3.2))
        self.say("Squared ReLU keeps ReLU's zero for negative inputs and squares the positive side.\n"
                 "Its slope is 2x, so it grows with the input instead of staying at 1.", wait=0)
        self.play(Create(ax), FadeIn(r), run_time=0.8)
        self.play(Create(q), FadeIn(legend), run_time=1.5)
        self.wait(4)
        rows = VGroup(*[year_row(y, t) for y, t in (
            ("2021", "So et al., Primer: found by searching transformer designs"),
            ("2024", "Nemotron-4 15B: squared ReLU in its MLP layers"),
            ("2025", "Nemotron-H and Arcee AFM-4.5B: relu2 in config.json"),
            ("2025", "Nemotron 3 Nano: squared ReLU in its MoE experts"),
        )]).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        if rows.width > 6.9:
            rows.scale_to_fit_width(6.9)
        rows.move_to(np.array([3.25, 0.55, 0]))
        self.say("So et al. found it in 2021 by searching over transformer designs for Primer.\n"
                 "Their abstract credits it, with one other change, for most of Primer's gains.", wait=0)
        self.play(FadeIn(rows[0], shift=0.1 * RIGHT))
        self.wait(4)
        self.say("NVIDIA adopted it in 2024 for Nemotron-4 and kept it in the hybrid Nemotron-H\n"
                 "and Nemotron 3 Nano models in 2025. Arcee's AFM-4.5B uses it too.", wait=0)
        for row in rows[1:]:
            self.play(FadeIn(row, shift=0.1 * RIGHT), run_time=0.7)
        self.wait(4)
        self.clear_stage()

    def clamped_swiglu(self):
        self.mark("clamped_swiglu")
        self.heading("gpt-oss (2025): a clamped SwiGLU")
        lo_gate = minimum(oss_gate, -4, 0)[1]
        hi_gate = oss_gate(OSS_LIMIT)
        prods = [a * c for a in (lo_gate, hi_gate) for c in (1 - OSS_LIMIT, 1 + OSS_LIMIT)]
        lines = VGroup(
            Text("g ← min(g, 7)", font=MATH, font_size=30),
            Text("u ← clamp(u, −7, 7)", font=MATH, font_size=30),
            Text("out = g σ(1.702 g) · (u + 1)", font=MATH, font_size=30),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to(np.array([-3.7, 1.7, 0]))
        names = Text("g: gate input    u: linear input", font=SERIF, font_size=22, color=GREY_B)
        names.next_to(lines, DOWN, buff=0.3).align_to(lines, LEFT)
        ax = make_axes((-3, 10), (-1, 8.5), 6.0, 3.9, np.array([3.5, 0.55, 0]),
                       xticks=(-2, 2, 4, 6, 8, 10), yticks=(2, 4, 6, 8))
        free = DashedVMobject(graph(ax, swish(OSS_ALPHA), 6.5, 8.3), num_dashes=12).set_stroke(GREY_B, 3)
        gate = graph(ax, oss_gate, -3, 10, kinks=(0, OSS_LIMIT)).set_stroke(CURVE, 5)
        plabel = Text("gate side: g σ(1.702 g), capped", font=MATH, font_size=22, color=CURVE).next_to(ax, UP, buff=0.12)
        self.say("OpenAI's gpt-oss models (August 2025) change SwiGLU in three ways.\n"
                 "The gate input g is capped at 7, and the linear input u is clipped to [−7, 7].", wait=0)
        self.play(FadeIn(lines[:2], lag_ratio=0.3), FadeIn(names), run_time=1.5)
        self.play(Create(ax), FadeIn(plabel), run_time=0.8)
        self.play(Create(gate), Create(free), run_time=2)
        self.wait(3)
        self.play(FadeIn(lines[2]))
        box17 = SurroundingRectangle(lines[2][7:12], color=ACCENT, buff=0.06)
        callback = Text("x σ(1.702 x): GELU's sigmoid form, 2016", font=MATH, font_size=24, color=ACCENT)
        callback.next_to(names, DOWN, buff=0.45).align_to(lines, LEFT)
        self.say("The gate is g σ(1.702 g). It uses the same constant as GELU's sigmoid\n"
                 "approximation from 2016, so this gate traces a GELU-shaped curve, not SiLU.", wait=0)
        self.play(Create(box17), FadeIn(callback))
        self.wait(4)
        plus = Text("“unconventional, including clamping and a residual connection”",
                    font=MATH, font_size=22, color=GREY_A)
        src = Text("gpt-oss model card", font=SERIF, font_size=20, color=GREY_B)
        quote = VGroup(plus, src).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        if quote.width > 6.4:
            quote.scale_to_fit_width(6.4)
        quote.next_to(callback, DOWN, buff=0.4).align_to(lines, LEFT)
        self.say("The linear branch gets an extra 1, so the gate multiplies (u + 1) instead of u.\n"
                 "The model card describes its SwiGLU in these words.", wait=0)
        self.play(FadeIn(quote))
        self.wait(4)
        self.say(f"With both clamps, every output of the gated product lies between {num(min(prods), '.1f')} and {num(max(prods), '.1f')}.\n"
                 "DeepSeek-V4.1-Flash (2026) also sets a swiglu_limit, of 10, in its config.json.", wait=0)
        bound = Text(f"output range: {num(min(prods), '.1f')} to {num(max(prods), '.1f')}", font=MATH, font_size=24, color=WHITE)
        bound.next_to(ax, DOWN, buff=0.5)
        self.play(FadeIn(bound))
        self.wait(5)
        self.clear_stage()

    def xielu_section(self):
        self.mark("xielu")
        self.heading("xIELU (Huang and Schlag, 2024)")
        mx, my = minimum(xielu, -6, 0)
        ax = make_axes((-6, 2), (-0.5, 4.5), 6.6, 3.9, np.array([-3.2, 0.55, 0]),
                       xticks=(-6, -4, -2, 2), yticks=(1, 2, 3, 4))
        q = graph(ax, relu2, -6, 2).set_stroke(GOOD, 3, opacity=0.45)
        c = graph(ax, xielu, -6, 2).set_stroke(PURPLE, 5)
        legend = VGroup(legend_row("xIELU, initial values", PURPLE), legend_row("squared ReLU", GOOD))
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(ax.c2p(-3.6, 3.7))
        formula = VGroup(
            MarkupText("x &gt; 0:   α<sub>p</sub> x² + β x", font=MATH, font_size=28),
            MarkupText("x ≤ 0:   α<sub>n</sub> (e<sup>x</sup> − 1) − α<sub>n</sub> x + β x", font=MATH, font_size=28),
            MarkupText("drawn with α<sub>p</sub> = α<sub>n</sub> = 0.8 and β = 0.5", font=MATH, font_size=22, color=GREY_B),
            MarkupText("α<sub>p</sub> and α<sub>n</sub> are learned in each layer", font=MATH, font_size=22, color=GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        formula[2].shift(0.15 * DOWN)
        formula[3].shift(0.15 * DOWN)
        if formula.width > 6.4:
            formula.scale_to_fit_width(6.4)
        formula.move_to(np.array([3.4, 1.4, 0]))
        self.say("Huang and Schlag (2024) designed xIELU's gradient first and integrated it:\n"
                 "a slope that grows linearly for x > 0, and an ELU-style shape for x ≤ 0.", wait=0)
        self.play(FadeIn(formula[:2], lag_ratio=0.3), run_time=1.5)
        self.play(Create(ax), FadeIn(q), run_time=0.8)
        self.play(Create(c), FadeIn(legend), FadeIn(formula[2:]), run_time=2)
        self.wait(3)
        d = Dot(ax.c2p(mx, my), radius=0.08, color=WHITE)
        dl = Text(f"minimum {num(my, '.3f')} at x = {num(mx, '.2f')}", font=MATH, font_size=24).move_to(np.array([3.4, -0.4, 0]))
        self.say("The positive side is a quadratic plus a linear term, close to squared ReLU.\n"
                 "On the negative side the curve dips and then climbs again as x falls.", wait=0)
        self.play(FadeIn(d), FadeIn(dl))
        self.wait(3.5)
        uses = VGroup(year_row("2024", "paper: lower perplexity than ReLU² and SwiGLU"),
                      year_row("", "in 1.1B and 3B Llama tests at matched compute"),
                      year_row("2025", "Swiss AI's Apertus: hidden_act = xielu")).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        if uses.width > 5.7:
            uses.scale_to_fit_width(5.7)
        uses.move_to(np.array([1.0, -1.3, 0]), aligned_edge=LEFT)
        self.say("The paper reports lower perplexity than ReLU² and SwiGLU at matched compute\n"
                 "in 1.1B and 3B Llama models. Swiss AI's Apertus models (2025) use it.", wait=0)
        self.play(FadeIn(uses, lag_ratio=0.3), run_time=1.5)
        self.wait(4.5)
        self.clear_stage()

    def attention_gate(self):
        self.mark("attention_gate")
        self.heading("Beyond the FFN: a sigmoid gate on attention")
        x = node("X", CURVE).move_to(np.array([-5.8, 1.6, 0]))
        att = box("attention", 2.4, 0.9, GREY_B).move_to(np.array([-2.9, 2.3, 0]))
        wt = box("W", 1.1, 0.9, GREY_B).move_to(np.array([-3.4, 0.6, 0]))
        wsub = MarkupText("W<sub>θ</sub>", font=MATH, font_size=30).move_to(wt[1])
        wt.remove(wt[1])
        wt.add(wsub)
        sg = box("σ", 0.9, 0.9, PURPLE).move_to(np.array([-1.6, 0.6, 0]))
        prod = node("⊙", ACCENT).move_to(np.array([0.3, 1.6, 0]))
        wo = box("out proj", 1.9, 0.9, GREY_B).move_to(np.array([2.5, 1.6, 0]))
        y = node("Y′", CURVE).move_to(np.array([4.7, 1.6, 0]))
        arrows = VGroup(arrow(x, att), arrow(x, wt), arrow(wt, sg), arrow(att, prod), arrow(sg, prod),
                        arrow(prod, wo), arrow(wo, y))
        ylab = Text("Y", font=MATH, font_size=26, color=GREY_A).next_to(arrows[3].get_center(), UR, buff=0.08)
        diagram = VGroup(x, att, wt, sg, prod, wo, y, arrows, ylab)
        eq = MarkupText("Y′ = Y ⊙ σ(X W<sub>θ</sub>)", font=MATH, font_size=32, color=ACCENT).move_to(np.array([5.2, 0.4, 0]))
        if eq.get_right()[0] > 6.8:
            eq.shift((eq.get_right()[0] - 6.8) * LEFT)
        self.say("Not every new gate sits in the feed-forward block. Qiu et al. (Qwen team, 2025)\n"
                 "put a sigmoid gate on the output of attention.", wait=0)
        self.play(FadeIn(x), FadeIn(att), Create(arrows[0]), run_time=1)
        self.play(Create(arrows[1]), FadeIn(wt), Create(arrows[2]), FadeIn(sg), run_time=1.2)
        self.play(Create(arrows[3:5]), FadeIn(prod), FadeIn(ylab), run_time=1)
        self.play(Create(arrows[5:]), FadeIn(wo), FadeIn(y), run_time=1)
        self.play(FadeIn(eq))
        self.wait(2.5)
        self.say("Each head's attention output Y is multiplied by σ(X W_θ), a gate computed\n"
                 "from the same hidden state X that feeds the attention.", wait=4)
        rows = VGroup(*[year_row(yr, t) for yr, t in (
            ("2025", "Qiu et al.: head-specific sigmoid gate after attention"),
            ("2025", "Qwen3-Next: model card lists Gated Attention layers"),
            ("2026", "Qwen3.5 and Qwen3.8: attn_output_gate = true in config.json"),
        )]).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        if rows.width > 12.4:
            rows.scale_to_fit_width(12.4)
        rows.move_to(np.array([0, -1.0, 0]))
        self.say("Qwen3-Next's model card (September 2025) lists Gated Attention layers, and\n"
                 "Qwen3.5 and Qwen3.8 (both 2026) set attn_output_gate to true in config.json.", wait=0)
        self.play(FadeIn(rows, lag_ratio=0.3), run_time=1.8)
        self.wait(5)
        self.clear_stage()

    def summary(self):
        self.mark("summary")
        self.heading("Introduced, then adopted")
        data = (
            ("GELU's sigmoid form, x σ(1.702x)", "2016", "2025: the gpt-oss gate"),
            ("Squared ReLU", "2021", "2024: Nemotron-4;  2025: Nemotron-H, Nemotron 3 Nano"),
            ("xIELU", "2024", "2025: Apertus"),
            ("Clamped SwiGLU", "2025", "2025: gpt-oss;  2026: DeepSeek-V4.1-Flash limit"),
            ("Sigmoid gate on attention", "2025", "2025: Qwen3-Next;  2026: Qwen3.5, Qwen3.8"),
        )
        cols = (-6.4, -1.55, 0.3)
        hdr = VGroup(Text("function", font=SERIF, font_size=22, color=GREY_B),
                     Text("introduced", font=SERIF, font_size=22, color=GREY_B),
                     Text("in released LLMs", font=SERIF, font_size=22, color=GREY_B))
        table = VGroup()
        y0 = 2.4
        for i, h in enumerate(hdr):
            h.move_to(np.array([cols[i], y0, 0]), aligned_edge=LEFT)
        table.add(hdr)
        for k, (name, born, used) in enumerate(data):
            yk = y0 - 0.7 * (k + 1)
            cells = VGroup(Text(name, font=MATH, font_size=24),
                           Text(born, font=MONO, font_size=24, color=ACCENT),
                           Text(used, font=MATH, font_size=22, color=GREY_A))
            if cells[0].width > 4.5:
                cells[0].scale_to_fit_width(4.5)
            if cells[2].width > 6.4:
                cells[2].scale_to_fit_width(6.4)
            for i, cell in enumerate(cells):
                cell.move_to(np.array([cols[i], yk, 0]), aligned_edge=LEFT)
            table.add(cells)
        rule = Line(np.array([-6.4, y0 - 0.32, 0]), np.array([6.7, y0 - 0.32, 0])).set_stroke(GREY_B, 1.5)
        self.say("The invention year and the adoption year often differ. Squared ReLU waited three years,\n"
                 "and GELU's 1.702 constant waited nine before gpt-oss used it in a gate.", wait=0)
        self.play(FadeIn(hdr), Create(rule))
        self.play(LaggedStart(*[FadeIn(r, shift=0.1 * RIGHT) for r in table[1:]], lag_ratio=0.3), run_time=3)
        self.wait(5)
        self.say("Part 2 covered four changes since 2024: squared ReLU, a clamped SwiGLU,\n"
                 "the learned xIELU curve, and a sigmoid gate on the attention output.", wait=6)
        self.play(*[FadeOut(m) for m in self.mobjects])
        self.cap, self.head = None, None

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
    """Axes with Text tick labels; x labels sit along the bottom edge, clear of the curves."""
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
        labels.add(Text(fmt(v), font=MONO, font_size=20, color=GREY_B).next_to(ax.c2p(v, yr[0]), DOWN, buff=0.15))
    for v in yticks:
        p = ax.c2p(0, v)
        labels.add(Line(p + 0.08 * LEFT, p + 0.08 * RIGHT).set_stroke(GREY_B, 2))
        labels.add(Text(fmt(v), font=MONO, font_size=20, color=GREY_B).next_to(p, LEFT, buff=0.15))
    ax.add(labels)
    return ax


def graph(ax, f, a, b, n=240, kinks=(0,)):
    """f on [a, b] as a polyline with a vertex exactly at each kink, so corners stay sharp."""
    cuts = [a] + sorted(k for k in kinks if a < k < b) + [b]
    xs = [a]
    for lo, hi in zip(cuts, cuts[1:]):
        m = max(2, round(n * (hi - lo) / (b - a)))
        xs.extend(np.linspace(lo, hi, m)[1:])
    return VMobject().set_points_as_corners([ax.c2p(x, f(x)) for x in xs])


def minimum(f, a, b):
    """Golden-section search for the minimiser of a unimodal f on [a, b]; returns (x, f(x))."""
    r = (math.sqrt(5) - 1) / 2
    for _ in range(200):
        c, d = b - r * (b - a), a + r * (b - a)
        if f(c) < f(d):
            b = d
        else:
            a = c
    x = (a + b) / 2
    return x, f(x)


def max_gap(f, g, a, b, n=120001):
    """Largest |f - g| on a fine grid over [a, b]."""
    return max(abs(f(x) - g(x)) for x in np.linspace(a, b, n))


def num(v, spec):
    """Format a number for display with a true minus sign."""
    return format(v, spec).replace("-", "−")


def fmt(v):
    return str(int(v)) if float(v).is_integer() else str(v)


def year_row(year, text):
    y = Text(year if year else "0000", font=MONO, font_size=24, color=ACCENT)
    if not year:
        y.set_opacity(0)
    return VGroup(y, Text(text, font=SERIF, font_size=24)).arrange(RIGHT, buff=0.3)


def node(label, colour):
    c = Circle(0.42).set_stroke(colour, 3).set_fill(config.background_color, 1)
    return VGroup(c, Text(label, font=MATH, font_size=30 if len(label) < 3 else 22).move_to(c))


def box(label, w, h, colour):
    r = RoundedRectangle(corner_radius=0.12, width=w, height=h).set_stroke(colour, 3).set_fill(config.background_color, 1)
    return VGroup(r, Text(label, font=MATH, font_size=30 if len(label) < 5 else 26).move_to(r))


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
