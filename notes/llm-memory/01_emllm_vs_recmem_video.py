"""Renders a Manim video comparing EM-LLM's surprise-cut key-value memory with RecMem's recurrence-triggered text memory.

Contents:
- MemoryTwoWays: the scene, with one method per section of the video
- precompute: every toy number on screen, with asserts for the claims the captions make
- surprise_stream, moving_threshold, key_similarity, modularity, split_labels: the EM-LLM toy computations
- recmem_trace, chat_trace: the RecMem toy's write decisions
- box, chip, arrow_between, wrap, lining, P, M: drawing and text helpers
- polar, num: positions and number formatting
- fonts, colours, the toy data and the paper numbers
"""
import math
import os

import numpy as np
from manim import *

SERIF = "Georgia"
"""Description: Serif font for captions, titles and labels."""
MONO = "Consolas"
"""Description: Monospace font for tick labels."""
MATH = "Cambria"
"""Description: Font for formulas, because Cambria has the Greek letters and symbols they need."""
EM = "#58C4DD"
"""Description: Colour of everything EM-LLM."""
RC = "#83C167"
"""Description: Colour of everything RecMem."""
AMBER = "#FFD166"
"""Description: Colour of thresholds, the current chunk and highlights."""
BAD = "#FC6255"
"""Description: Colour of event boundaries and of the eager memory system."""
PURPLE = "#B48EDE"
"""Description: Colour of EM-LLM's contiguity buffer."""
GREY_T = "#7d8894"
"""Description: Colour of raw, unconsolidated items and of other methods."""
DIM = "#2a313a"
"""Description: Fill of an event block that was not fetched."""
config.background_color = "#0b0d10"

SECTIONS = ("intro", "where", "emllm_cut", "emllm_refine", "emllm_recall", "recmem_write",
            "recmem_read", "same_chat", "results", "tradeoffs", "roots", "summary")
"""Description: The video's sections in playing order."""

RNG = np.random.default_rng(20261122)
"""Description: The seeded generator behind the toy surprise stream and key similarities."""
N_TOK = 64
"""Description: Tokens in the toy surprise stream."""
SPIKES = (13, 27, 40, 52)
"""Description: The toy stream's topic shifts, where surprise spikes."""
TAU = 10
"""Description: The toy threshold's window, in tokens."""
GAMMA = 3.0
"""Description: The toy threshold's number of standard deviations above the window mean."""
BLOCKS = (7, 5, 6)
"""Description: The true event sizes in the 18-token refinement toy."""
INIT_B = (7, 14)
"""Description: The toy's surprise boundaries: the second one is two tokens late."""

THETA_SIM = 0.7
"""Description: The cosine threshold in the RecMem toy, which is the paper's LoCoMo value."""
THETA_COUNT = 3
"""Description: The toy recurrence threshold; the paper uses 5 on LoCoMo and 4 on LongMemEval-S."""
TURNS7 = (("order a birthday cake", 60), ("how to wash dark jeans?", 200), ("what to write on the cake", 84),
          ("my flight got cancelled!", 315), ("move the cake pickup to 5 pm", 40), ("which bakery is best?", 72),
          ("add candles to the order", 52))
"""Description: The RecMem write demo's turns, each with its embedding direction in degrees on the 2-D cartoon."""
CHAT6 = (("order a\nbirthday cake", "cake", 1.0), ("how to wash\ndark jeans?", "jeans", 2.9),
         ("what to write\non the cake", "cake", 2.2), ("my flight got\ncancelled!", "flight", 3.7),
         ("move the\ncake pickup", "cake", 2.5), ("thanks!", "thanks", 0.6))
"""Description: The six-turn chat: each turn's text, its topic, and its illustrative surprise."""
THR6 = 1.8
"""Description: The six-turn chat's illustrative surprise threshold."""

EM_LONGBENCH = (("RAG", 36.44, GREY_T), ("full context", 39.30, GREY_T), ("EM-LLM", 51.58, EM))
"""Description: LongBench averages with LLaMA-3.1-8B, from EM-LLM's Table 9 (RAG uses NV-Embed-v2; EM-LLM is S at 4K+4K)."""
EM_ROOTS = (("2002", "Howard, Kahana", "recall drifts to\nnearby moments"),
            ("2007", "Zacks et al.", "experience is cut\nat prediction errors"),
            ("2023", "Kumar et al.", "surprise predicts\nevent boundaries"))
"""Description: The psychology EM-LLM cites, as (year, authors, finding), oldest first."""
RC_ROOTS = (("1968", "Atkinson, Shiffrin", "fast stores feed a\nlong-term store"),
            ("1995", "McClelland, McNaughton,\nO'Reilly", "the cortex slowly\nextracts what repeats"),
            ("2016", "Kumaran, Hassabis,\nMcClelland", "learning systems\ntheory, revisited"))
"""Description: The psychology RecMem cites, as (year, authors, finding), oldest first."""
RC_LOCOMO = (("full context", 0.0, 84.18), ("naive RAG", 0.0, 54.42), ("MemoryOS", 400.7, 67.60),
             ("Mem0", 1520.8, 62.92), ("A-Mem", 1459.9, 68.83), ("RecMem", 193.2, 81.10))
"""Description: LoCoMo with GPT-4.1-mini, from RecMem's Tables 1 and 2, as (method, construction tokens in K, LLM-judge overall)."""


class MemoryTwoWays(Scene):
    """
    Description: The video's scene. Each section method draws one part of the video and ends on an empty frame.
    Fields:
    - cap: the caption on screen, or None.
    - head: the section heading on screen, or None.
    - d: the numbers from precompute().
    """

    def construct(self):
        """
        Description: Computes the toy numbers, then plays the sections in order.
        Inputs: none
        Outputs: none
        Constraints: when the MEM_ONLY environment variable names sections, separated by commas, only those play.
        """
        self.cap = None
        self.head = None
        self.d = precompute()
        only = [s for s in os.environ.get("MEM_ONLY", "").split(",") if s]
        for name in SECTIONS:
            if not only or name in only:
                getattr(self, name)()

    def intro(self):
        """
        Description: Shows a token stream outgrowing the context window, then the title and the two papers.
        Inputs: none
        Outputs: none
        """
        self.mark("intro")
        ribbon = VGroup(*[Square(0.2).set_stroke(GREY_T, 1.5).set_fill(GREY_T, 0.3) for _ in range(46)])
        ribbon.arrange(RIGHT, buff=0.08).move_to(2.3 * DOWN)
        window = SurroundingRectangle(ribbon[-14:], buff=0.1, color=AMBER, corner_radius=0.06)
        wlab = P("context window", 20, AMBER).next_to(window, UP, buff=0.12)
        title = P("Two ways to give an LLM a long memory", 46)
        em = P(f'<span foreground="{EM}">EM-LLM</span> cuts one long input where the model is surprised', 28)
        rc = P(f'<span foreground="{RC}">RecMem</span> consolidates the conversation topics that come back', 28)
        refs = P("Fountas et al., ICLR 2025      ·      Dai et al., Findings of ACL 2026", 20, GREY_B)
        words = VGroup(title, VGroup(em, rc).arrange(DOWN, buff=0.22), refs).arrange(DOWN, buff=0.45)
        words.move_to(1.1 * UP)
        self.play(LaggedStart(*[FadeIn(s, shift=0.2 * LEFT) for s in ribbon], lag_ratio=0.03), run_time=1.6)
        self.play(Create(window), FadeIn(wlab), ribbon[:-14].animate.set_opacity(0.25), run_time=0.9)
        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(em, shift=0.1 * UP), FadeIn(rc, shift=0.1 * UP), FadeIn(refs), run_time=1)
        self.wait(4)
        self.end_section()

    def where(self):
        """
        Description: Contrasts memory inside the model, as per-layer key-value events, with memory outside it, as text stores.
        Inputs: none
        Outputs: none
        """
        self.mark("where")
        self.heading("Where the memory lives")
        self.say("A model attends only to what fits in its window. Both papers bring older material back, from different places.")
        lt = P("EM-LLM: inside the model", 28, EM).move_to(np.array([-3.6, 2.5, 0]))
        widths = (0.42, 0.66, 0.5, 0.36, 0.6, 0.48)
        hot = ((1, 2), (4,), (1,))
        layers = VGroup()
        for r, name in enumerate(("layer L", "layer 2", "layer 1")):
            y = 1.5 - r * 0.95
            frame = RoundedRectangle(corner_radius=0.08, width=6.0, height=0.72).set_stroke(GREY_T, 1.5)
            frame.move_to(np.array([-3.6, y, 0]))
            row = VGroup(P(name, 18, GREY_B).move_to(np.array([-6.05, y, 0])))
            x = -5.45
            for k, w in enumerate(widths):
                blk = Rectangle(width=w, height=0.42).set_stroke(EM, 1.5)
                blk.set_fill(EM if k in hot[r] else DIM, 0.8 if k in hot[r] else 0.5)
                row.add(blk.move_to(np.array([x + w / 2, y, 0])))
                x += w + 0.06
            loc = Rectangle(width=1.25, height=0.42).set_stroke(AMBER, 1.5).set_fill(AMBER, 0.25)
            row.add(loc.move_to(np.array([-1.4, y, 0])), P("local", 16, AMBER).move_to(np.array([-1.4, y, 0])))
            layers.add(VGroup(frame, row))
        lnote = P("keys and values of past tokens, cut into events;\neach layer fetches its own", 20, GREY_B)
        lnote.move_to(np.array([-3.6, -1.55, 0]))
        rt = P("RecMem: outside the model", 28, RC).move_to(np.array([3.7, 2.5, 0]))
        stores = VGroup(*[box(t, 1.85, 0.95, RC, 19) for t in ("subconscious\nraw turns", "episodic\nnarratives", "semantic\nfacts")])
        stores.arrange(RIGHT, buff=0.12).move_to(np.array([3.7, 1.35, 0]))
        prompt = box("prompt: retrieved text + question", 4.9, 0.6, GREY_B, 20).move_to(np.array([3.7, 0.0, 0]))
        llm = box("LLM, unchanged: any chat model", 4.9, 0.6, GREY_B, 20).move_to(np.array([3.7, -1.15, 0]))
        a1 = arrow_between(stores.get_bottom(), prompt.get_top())
        a2 = arrow_between(prompt.get_bottom(), llm.get_top())
        divider = DashedLine(np.array([0.3, 2.8, 0]), np.array([0.3, -2.0, 0]), dash_length=0.1).set_stroke(GREY_T, 1.5)
        self.play(FadeIn(lt), Create(divider), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(l, shift=0.1 * UP) for l in layers], lag_ratio=0.25), run_time=1.5)
        self.play(FadeIn(lnote))
        self.say("EM-LLM keeps the keys and values of every past token inside the model, grouped into events. Each layer fetches its own.")
        self.play(FadeIn(rt), LaggedStart(*[FadeIn(s, shift=0.1 * DOWN) for s in stores], lag_ratio=0.2), run_time=1.2)
        self.play(GrowArrow(a1), FadeIn(prompt), run_time=0.7)
        self.play(GrowArrow(a2), FadeIn(llm), run_time=0.7)
        self.say("RecMem keeps memory as text in three stores, and pastes what it finds into an ordinary prompt.")
        self.say("So EM-LLM needs open weights to patch attention, while RecMem works with any chat model, even one behind an API.")
        self.end_section()

    def emllm_cut(self):
        """
        Description: Draws a toy surprise stream, its moving threshold, and the events the boundaries cut out.
        Inputs: none
        Outputs: none
        """
        self.mark("emllm_cut")
        self.heading("EM-LLM, step 1: cut the stream where surprise spikes")
        d = self.d
        s, thr, bnd = d["surprise"], d["threshold"], d["boundaries"]
        x0, dx, base, scale = -6.2, 12.4 / N_TOK, -1.7, 0.72

        def bx(t):
            return x0 + (t + 0.5) * dx

        axis = Line(np.array([x0, base, 0]), np.array([x0 + 12.4, base, 0])).set_stroke(GREY_B, 2)
        bars = VGroup()
        for t in range(N_TOK):
            h = max(s[t] * scale, 0.01)
            bars.add(Rectangle(width=dx * 0.7, height=h).set_stroke(width=0).set_fill(EM, 0.85)
                     .move_to(np.array([bx(t), base + h / 2, 0])))
        f1 = M("surprise<sub>t</sub> = −log P(x<sub>t</sub> | x<sub>1</sub> … x<sub>t−1</sub>)", 28)
        f1.move_to(np.array([0, 2.55, 0])).align_to(np.array([x0, 0, 0]), LEFT)
        tlab = M("tokens →", 18, GREY_B).next_to(axis, DOWN, buff=0.12).align_to(axis, RIGHT)
        self.play(Create(axis), FadeIn(tlab), FadeIn(f1), run_time=0.8)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.03), run_time=2.2)
        self.say("Run the text through the model. A token's surprise is how badly the model predicted it.")
        pts = [np.array([bx(t), base + thr[t] * scale, 0]) for t in range(TAU, N_TOK)]
        tline = VMobject().set_points_as_corners(pts).set_stroke(AMBER, 3)
        f2 = M("boundary if surprise<sub>t</sub> &gt; T<sub>t</sub> = μ + γσ over the last τ tokens", 28, AMBER)
        f2.move_to(np.array([0, 1.95, 0])).align_to(f1, LEFT)
        self.play(Create(tline), FadeIn(f2), run_time=1.5)
        self.say("The threshold is the recent mean plus γ standard deviations, so it adapts to how surprising this text usually is.")
        lines = VGroup(*[DashedLine(np.array([x0 + b * dx, base - 0.05, 0]), np.array([x0 + b * dx, 1.55, 0]),
                                    dash_length=0.08).set_stroke(BAD, 2.5) for b in bnd])
        self.play(*[bars[b].animate.set_fill(BAD, 1) for b in bnd], run_time=0.6)
        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.2), run_time=1)
        self.say("Every token that beats the threshold starts a new event.")
        edges = [0, *bnd, N_TOK]
        spans = VGroup()
        for k in range(len(edges) - 1):
            a, b = edges[k], edges[k + 1]
            r = Rectangle(width=(b - a) * dx - 0.06, height=0.34).set_stroke(EM, 2).set_fill(EM, 0.18)
            r.move_to(np.array([x0 + (a + b) / 2 * dx, base - 0.45, 0]))
            spans.add(VGroup(r, P(f"e{k + 1}", 18, EM).move_to(r)))
        note = P(f"toy values: γ = {GAMMA:g}, τ = {TAU}", 18, GREY_B).move_to(np.array([5.3, 2.55, 0]))
        self.play(FadeOut(tlab), LaggedStart(*[FadeIn(sp) for sp in spans], lag_ratio=0.15), FadeIn(note), run_time=1.2)
        self.say("Each event is a run of tokens of any length. Its keys and values are kept together as one block.")
        self.end_section()

    def emllm_refine(self):
        """
        Description: Moves a late surprise boundary to the position with the highest modularity on a toy key-similarity matrix.
        Inputs: none
        Outputs: none
        """
        self.mark("emllm_refine")
        self.heading("Step 2: slide each boundary so its events hang together")
        d = self.d
        A, q, init, best = d["A"], d["q_curve"], d["init_b"], d["best_b"]
        n, cell = A.shape[0], 0.25
        org = np.array([-6.3, 2.4, 0])
        heat = VGroup()
        for i in range(n):
            for j in range(n):
                c = interpolate_color(ManimColor(config.background_color), ManimColor(EM), float(A[i, j]))
                heat.add(Square(cell).set_stroke(width=0).set_fill(c, 1)
                         .move_to(org + np.array([(j + 0.5) * cell, -(i + 0.5) * cell, 0])))
        hlab = P("token i  ×  token j", 18, GREY_B).next_to(heat, DOWN, buff=0.12)
        fa = M("A<sub>ij</sub> = k<sub>i</sub> · k<sub>j</sub>   (attention-key similarity)", 26).move_to(np.array([3.6, 2.4, 0]))

        def outlines(p):
            g = VGroup()
            for a, b in ((0, init[0]), (init[0], p), (p, n)):
                sq = Square((b - a) * cell).set_stroke(AMBER, 3)
                g.add(sq.move_to(org + np.array([(a + b) / 2 * cell, -(a + b) / 2 * cell, 0])))
            return g

        self.play(FadeIn(heat), FadeIn(hlab), FadeIn(fa), run_time=1.2)
        self.say("Compare every pair of tokens by their attention keys. Tokens from the same event form bright blocks.")
        ol = outlines(init[1])
        self.play(Create(ol), run_time=1)
        self.say(f"Surprise put this boundary {init[1] - best} tokens late, so the middle event swallows the start of the next one.")
        ps = sorted(q)
        qmax = max(q.values())
        cx0, cw, cbase, ch = 1.0, 0.7, -1.55, 2.6
        qbars, qticks = VGroup(), VGroup()
        for k, p in enumerate(ps):
            h = max(q[p], 0) / qmax * ch
            bar = Rectangle(width=cw * 0.7, height=max(h, 0.01)).set_stroke(width=0).set_fill(GREY_T, 0.9)
            qbars.add(bar.move_to(np.array([cx0 + (k + 0.5) * cw, cbase + h / 2, 0])))
            qticks.add(Text(str(p), font=MONO, font_size=16, color=GREY_B).move_to(np.array([cx0 + (k + 0.5) * cw, cbase - 0.22, 0])))
        qaxis = Line(np.array([cx0, cbase, 0]), np.array([cx0 + len(ps) * cw, cbase, 0])).set_stroke(GREY_B, 2)
        qlab = P("modularity Q for each position of the boundary", 20, GREY_B).move_to(np.array([3.6, 1.6, 0]))
        cur = qbars[ps.index(init[1])]
        read = M(f"Q = {num(q[init[1]], '.3f')}", 26, AMBER).move_to(np.array([3.6, -2.25, 0]))
        self.play(Create(qaxis), FadeIn(qlab), FadeIn(qticks), run_time=0.6)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in qbars], lag_ratio=0.08), run_time=1.4)
        self.play(cur.animate.set_fill(AMBER, 1), FadeIn(read), run_time=0.6)
        self.say("Refinement tries every position after the previous boundary, up to this one, and scores each split by modularity: "
                 "similarity inside events, minus what chance would give.")
        new_read = M(f"Q = {num(q[best], '.3f')}   (was {num(q[init[1]], '.3f')})", 26, RC).move_to(read)
        self.play(Transform(ol, outlines(best)), cur.animate.set_fill(GREY_T, 0.9),
                  qbars[ps.index(best)].animate.set_fill(RC, 1), Transform(read, new_read), run_time=1.4)
        self.say("The best split moves the boundary back. This is graph arithmetic on keys, so it needs no extra LLM call.")
        self.say("It does cost time. With Mistral-7B, a 512-token chunk takes 1.40 s in InfLLM, "
                 "1.57 s with surprise alone and 2.27 s with refinement.")
        self.end_section()

    def emllm_recall(self):
        """
        Description: Shows one layer fetching events by key similarity, then their neighbours, then building its attention context.
        Inputs: none
        Outputs: none
        """
        self.mark("emllm_recall")
        self.heading("Step 3: for each new chunk, fetch events in every layer")
        widths = (0.62, 0.8, 0.55, 0.7, 0.5, 0.75, 0.6, 0.68, 0.52, 0.72)
        hits, near = (2, 6), (1, 3, 5, 7)
        evs = VGroup()
        x = -6.5
        for k, w in enumerate(widths):
            r = Rectangle(width=w, height=0.6).set_stroke(EM, 2).set_fill(DIM, 0.6)
            evs.add(VGroup(r, P(f"e{k + 1}", 18).move_to(r)).move_to(np.array([x + w / 2, 1.3, 0])))
            x += w + 0.1
        elab = P("stored events, in CPU memory", 20, GREY_B).move_to(np.array([0, 0.68, 0])).align_to(evs, LEFT)
        q = box("current chunk", 2.3, 0.62, AMBER, 22).move_to(np.array([3.9, 1.3, 0]))
        self.play(LaggedStart(*[FadeIn(e) for e in evs], lag_ratio=0.08), FadeIn(elab), FadeIn(q), run_time=1.4)
        self.say("Old events wait in CPU memory. For each new 512-token chunk, a layer compares the chunk's queries "
                 "with a few representative keys from every event.", 0)
        arrows = VGroup(*[CurvedArrow(q.get_top() + 0.05 * UP, evs[h].get_top() + 0.05 * UP, angle=0.55).set_stroke(EM, 3)
                          for h in hits])
        sl = M("similarity buffer k<sub>s</sub>: nearest events (k-NN)", 24, EM).move_to(np.array([-2.6, 0.05, 0]))
        self.play(*[Create(a) for a in arrows], *[evs[h][0].animate.set_fill(EM, 0.85) for h in hits], FadeIn(sl), run_time=1.4)
        self.wait(1.5)
        cl = M("contiguity buffer k<sub>c</sub>: their neighbours in time", 24, PURPLE).next_to(sl, DOWN, buff=0.2).align_to(sl, LEFT)
        self.play(*[evs[h][0].animate.set_fill(PURPLE, 0.75) for h in near], FadeIn(cl), run_time=1)
        self.say("It also brings each hit's neighbours. People who recall one moment tend to recall what came right after it next.")
        segs = VGroup(*[box(t, w, 0.62, c, 20) for t, w, c in (("128 initial tokens", 2.6, GREY_T), ("k<sub>s</sub> events", 2.0, EM),
                                                               ("k<sub>c</sub> events", 2.0, PURPLE), ("local context", 2.6, AMBER))])
        segs.arrange(RIGHT, buff=0.1).move_to(np.array([-0.4, -1.45, 0]))
        alab = P("attention runs over this context", 20, GREY_B).next_to(segs, DOWN, buff=0.2)
        self.play(FadeOut(arrows), LaggedStart(*[FadeIn(sg, shift=0.15 * UP) for sg in segs], lag_ratio=0.2), run_time=1.4)
        self.play(FadeIn(alab))
        self.say("Attention then runs over 128 initial tokens, the fetched events and the recent local context. "
                 "Every layer does its own fetch, so layers can recall different events.")
        self.end_section()

    def recmem_write(self):
        """
        Description: Streams seven turns onto a 2-D cartoon of embedding directions and shows when RecMem calls the LLM.
        Inputs: none
        Outputs: none
        """
        self.mark("recmem_write")
        self.heading("RecMem: store every turn raw, consolidate only what recurs")
        tr = self.d["trace7"]
        c, rad = np.array([-3.7, -0.2, 0]), 2.0
        half = math.degrees(math.acos(THETA_SIM))
        ring = Circle(radius=rad).set_stroke(GREY_T, 1.5).move_to(c)
        clab = P("turn embeddings as directions (2-D cartoon)", 18, GREY_B).move_to(c + np.array([0, -rad - 0.35, 0]))
        f1 = M("s<sub>i</sub> = (v<sub>i</sub>, u<sub>i</sub>),   v<sub>i</sub> = Φ(u<sub>i</sub>)", 26)
        f2 = M("R<sub>i</sub> = { s<sub>j</sub> : cos(v<sub>i</sub>, v<sub>j</sub>) ≥ θ<sub>sim</sub> }", 26)
        f3 = M("consolidate when |R<sub>i</sub>| ≥ θ<sub>count</sub>", 26)
        forms = VGroup(f1, f2, f3).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to(np.array([3.5, 2.05, 0]))
        tnote = P(f"toy: θ<sub>sim</sub> = {THETA_SIM:g}, θ<sub>count</sub> = {THETA_COUNT}", 18, GREY_B).next_to(forms, DOWN, buff=0.2).align_to(forms, LEFT)
        self.play(Create(ring), FadeIn(clab), FadeIn(forms), FadeIn(tnote), run_time=1)
        self.say("Each turn u<sub>i</sub> is embedded once and saved raw in the subconscious store. Saving it needs no LLM call.", 0)
        cur = P(" ", 24)
        calls = [0, 0]
        cnt = self.counters(calls)
        self.play(FadeIn(cnt), run_time=0.4)
        dots, nums, epi = [], [], None
        ep_box = fact_box = None
        for i, (text, ang) in enumerate(TURNS7):
            step = tr[i]
            p = polar(c, rad, ang)
            dt = Dot(p, radius=0.1, color=WHITE)
            nm = Text(str(i + 1), font=MONO, font_size=18, color=GREY_B).move_to(polar(c, rad + 0.3, ang))
            line = P(f"T{i + 1}  “{text}”", 24).move_to(np.array([3.5, 0.15, 0]))
            line.align_to(forms, LEFT)
            cone = AnnularSector(inner_radius=0, outer_radius=rad, angle=math.radians(2 * half),
                                 start_angle=math.radians(ang - half)).set_fill(RC, 0.16).shift(c)
            self.play(FadeTransform(cur, line), GrowFromCenter(dt), FadeIn(nm), FadeIn(cone), run_time=0.7)
            cur = line
            calls[1] += 1
            if step["kind"] == "merge":
                calls[0] += 1
                arr = Arrow(p, ep_box.get_left(), buff=0.12, stroke_width=4, color=RC)
                self.play(GrowArrow(arr), Transform(cnt, self.counters(calls)), Indicate(ep_box, color=RC), dt.animate.set_color(RC), run_time=1)
                self.say("A later cake turn matches the existing episode, so RecMem merges it in first. "
                         "One call, and the topic doesn't fork into a second episode.")
                self.play(FadeOut(arr), FadeOut(cone), run_time=0.5)
                dots.append(dt)
                continue
            r = M(f"|R| = {step['r']}", 24, RC if step["kind"] == "consolidate" else GREY_B).move_to(np.array([3.5, -0.4, 0])).align_to(forms, LEFT)
            matches = [dots[j] for j in step["matches"]]
            self.play(FadeIn(r), *[Indicate(m, color=AMBER, scale_factor=1.6) for m in matches], Transform(cnt, self.counters(calls)), run_time=0.8)
            if i == 0:
                self.say("A new turn looks for earlier turns that point the same way, inside a cone where cosine similarity is at least θ<sub>sim</sub>.")
            if step["kind"] == "consolidate":
                calls[0] += 2
                for m in matches + [dt]:
                    m.generate_target()
                    m.target.set_color(RC)
                ep_box = box("episode: the cake order", 3.0, 0.6, RC, 20).move_to(np.array([2.6, -1.25, 0]))
                fact_box = box("facts", 1.6, 0.6, RC, 20).next_to(ep_box, RIGHT, buff=0.25)
                self.play(*[MoveToTarget(m) for m in matches + [dt]], FadeIn(ep_box, shift=0.1 * DOWN), run_time=0.8)
                self.play(FadeIn(fact_box, shift=0.1 * DOWN), Transform(cnt, self.counters(calls)), run_time=0.6)
                self.say(f"Now {step['r']} earlier turns match, which meets θ<sub>count</sub>. One LLM call writes an episode, "
                         "and one more pulls out the precise facts.")
            else:
                self.wait(0.6)
            self.play(FadeOut(cone), FadeOut(r), run_time=0.4)
            dots.append(dt)
        self.say("Jeans and the flight never come back, so they stay raw. They can still be found later, by plain embedding search.")
        self.say(f"Over seven turns RecMem made {calls[0]} LLM calls, where an eager system makes at least {calls[1]}. "
                 "That gap is RecMem's saving: up to 87% fewer memory-building tokens.")
        self.say("The paper's thresholds are θ<sub>sim</sub> = 0.7 with θ<sub>count</sub> = 5 on LoCoMo, "
                 "and 0.6 with 4 on LongMemEval-S.")
        self.end_section()

    def recmem_read(self):
        """
        Description: Shows a question searching the three stores and the retrieved text going into an unchanged LLM.
        Inputs: none
        Outputs: none
        """
        self.mark("recmem_read")
        self.heading("RecMem at question time: one search, three stores")
        qn = box("“When do I pick up the cake?”", 4.2, 0.7, AMBER, 22).move_to(np.array([-4.3, 1.6, 0]))
        stores = VGroup(*[box(t, 2.9, 0.75, RC, 20) for t in ("subconscious: 10 turns", "episodic: 5 episodes", "semantic: 10 facts")])
        stores.arrange(DOWN, buff=0.22).move_to(np.array([0.0, 0.55, 0]))
        prompt = box("prompt: retrieved text\n+ question", 2.9, 1.0, GREY_B, 20).move_to(np.array([4.4, 0.55, 0]))
        llm = box("LLM, unchanged", 2.9, 0.65, GREY_B, 20).move_to(np.array([4.4, -1.15, 0]))
        a0 = VGroup(*[arrow_between(qn.get_right(), s.get_left()) for s in stores])
        a1 = VGroup(*[arrow_between(s.get_right(), prompt.get_left()) for s in stores])
        a2 = arrow_between(prompt.get_bottom(), llm.get_top())
        self.play(FadeIn(qn, shift=0.1 * RIGHT), run_time=0.6)
        self.play(LaggedStart(*[GrowArrow(a) for a in a0], lag_ratio=0.15), LaggedStart(*[FadeIn(s) for s in stores], lag_ratio=0.15), run_time=1.2)
        self.say("The question is embedded once and searched against all three stores: 10 raw turns, 5 episodes and 10 facts by default.")
        self.play(LaggedStart(*[GrowArrow(a) for a in a1], lag_ratio=0.15), FadeIn(prompt), run_time=1)
        self.play(GrowArrow(a2), FadeIn(llm), run_time=0.7)
        self.say("The answering model only ever sees text, which is why the paper can run it on GPT-4o-mini and GPT-4.1-mini through an API.")
        self.end_section()

    def same_chat(self):
        """
        Description: Runs one six-turn chat through EM-LLM's surprise rule and RecMem's recurrence rule, side by side.
        Inputs: none
        Outputs: none
        """
        self.mark("same_chat")
        self.heading("Same conversation, two rules")
        d = self.d
        xs = [-5.425 + i * 2.17 for i in range(6)]
        cards = VGroup()
        for i, (text, topic, _) in enumerate(CHAT6):
            r = RoundedRectangle(corner_radius=0.1, width=2.0, height=1.05).set_stroke(GREY_B, 1.5)
            t = VGroup(P(f"T{i + 1}", 18, GREY_B), P(text, 19)).arrange(DOWN, buff=0.08)
            cards.add(VGroup(r, t.move_to(r)).move_to(np.array([xs[i], 2.2, 0])))
        ill = P("illustrative values", 18, GREY_B).move_to(np.array([5.5, 3.0, 0]))
        self.play(LaggedStart(*[FadeIn(cd, shift=0.1 * DOWN) for cd in cards], lag_ratio=0.12), FadeIn(ill), run_time=1.4)
        self.say("Six turns of one chat. Three are about a cake, and one is a cancelled flight.")
        el = P("EM-LLM · cut where surprise beats the threshold", 22, EM).move_to(np.array([0, 1.35, 0])).align_to(cards, LEFT)
        base, sc = -0.35, 0.42
        bars = VGroup(*[Rectangle(width=0.7, height=s * sc).set_stroke(width=0).set_fill(EM, 0.85)
                        .move_to(np.array([xs[i], base + s * sc / 2, 0])) for i, (_, _, s) in enumerate(CHAT6)])
        thr = DashedLine(np.array([xs[0] - 1.0, base + THR6 * sc, 0]), np.array([xs[-1] + 1.0, base + THR6 * sc, 0]),
                         dash_length=0.1).set_stroke(AMBER, 2.5)
        self.play(FadeIn(el), LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.1), Create(thr), run_time=1.4)
        cuts = VGroup(*[DashedLine(np.array([xs[b] - 1.085, base - 0.1, 0]), np.array([xs[b] - 1.085, 1.12, 0]),
                                   dash_length=0.08).set_stroke(BAD, 2.5) for b in d["chat_bounds"]])
        labs = VGroup()
        edges = [0, *d["chat_bounds"], 6]
        for k in range(len(edges) - 1):
            mid = (xs[edges[k]] + xs[edges[k + 1] - 1]) / 2
            labs.add(P(f"e{k + 1}", 20, EM).move_to(np.array([mid, base - 0.28, 0])))
        self.play(*[bars[b].animate.set_fill(BAD, 1) for b in d["chat_bounds"]], LaggedStart(*[Create(c) for c in cuts], lag_ratio=0.15), run_time=1)
        self.play(FadeIn(labs), run_time=0.6)
        self.say("EM-LLM cuts wherever a turn surprises the model. Each topic switch spikes, and the cancelled flight gets an event of its own.")
        rl = P("RecMem · consolidate on recurrence (toy rule: fires on 1 earlier match)", 22, RC).move_to(np.array([0, -1.05, 0])).align_to(cards, LEFT)
        chips = VGroup(*[chip(state, xs[i]) for i, state in enumerate(d["chat_states"])])
        self.play(FadeIn(rl), LaggedStart(*[FadeIn(cp, shift=0.1 * UP) for cp in chips], lag_ratio=0.15), run_time=1.6)
        link = VGroup(ArcBetweenPoints(np.array([xs[0], -2.0, 0]), np.array([xs[2], -2.0, 0]), angle=0.7),
                      ArcBetweenPoints(np.array([xs[2], -2.0, 0]), np.array([xs[4], -2.0, 0]), angle=0.7)).set_stroke(RC, 2.5)
        self.play(Create(link), run_time=0.8)
        self.say("RecMem waits for repetition. T3 matches T1, so the cake topic is consolidated, and T5 is merged into that episode.")
        hl = VGroup(SurroundingRectangle(cards[3], color=AMBER, buff=0.06, corner_radius=0.1),
                    SurroundingRectangle(chips[3], color=AMBER, buff=0.06, corner_radius=0.1))
        self.play(Create(hl), run_time=0.8)
        self.say("The flight never comes back, so RecMem leaves it raw. EM-LLM marks what is new; RecMem keeps what returns.")
        self.say("EM-LLM's events stay in time order, while RecMem's episode gathers T1, T3 and T5 by meaning.")
        self.end_section()

    def results(self):
        """
        Description: Shows each paper's headline result against its own baselines, in two panels with separate axes.
        Inputs: none
        Outputs: none
        """
        self.mark("results")
        self.heading("Results: each against the baselines in its own paper")
        warn = P("different benchmarks and metrics: compare inside a panel, never across", 22, AMBER).move_to(np.array([0, 2.6, 0]))
        lt = P("EM-LLM · LongBench average · LLaMA-3.1-8B", 22, EM).move_to(np.array([-3.7, 1.95, 0]))
        base, sc = -1.6, 3.0 / 60
        bars, vals, names = VGroup(), VGroup(), VGroup()
        for k, (name, v, col) in enumerate(EM_LONGBENCH):
            x = -5.4 + k * 1.7
            b = Rectangle(width=1.0, height=v * sc).set_stroke(width=0).set_fill(col, 0.9).move_to(np.array([x, base + v * sc / 2, 0]))
            bars.add(b)
            vals.add(Text(num(v, ".2f"), font=MONO, font_size=20).next_to(b, UP, buff=0.08))
            names.add(P(name, 19, GREY_B).move_to(np.array([x, base - 0.25, 0])))
        laxis = Line(np.array([-6.2, base, 0]), np.array([-1.2, base, 0])).set_stroke(GREY_B, 2)
        lnote = P("and ahead of InfLLM with all five base models", 19, GREY_B).move_to(np.array([-3.7, -2.25, 0]))
        divider = DashedLine(np.array([-0.35, 2.2, 0]), np.array([-0.35, -2.5, 0]), dash_length=0.1).set_stroke(GREY_T, 1.5)
        self.play(FadeIn(warn), FadeIn(lt), Create(laxis), Create(divider), run_time=0.8)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.2), FadeIn(names), run_time=1.2)
        self.play(FadeIn(vals), FadeIn(lnote), run_time=0.6)
        self.say("EM-LLM with LLaMA-3.1-8B averages 51.58 on LongBench, against 39.30 for the full context and 36.44 for RAG.")
        rt = P("RecMem · LoCoMo · GPT-4.1-mini", 22, RC).move_to(np.array([3.5, 1.95, 0]))
        ox, oy, w, h = 0.9, -1.6, 5.4, 3.0

        def pt(k, acc):
            return np.array([ox + k / 1600 * w, oy + (acc - 50) / 40 * h, 0])

        axes = VGroup(Line(pt(0, 50), pt(1600, 50)), Line(pt(0, 50), pt(0, 90))).set_stroke(GREY_B, 2)
        ticks = VGroup(*[Text(s, font=MONO, font_size=16, color=GREY_B).next_to(pt(k, 50), DOWN, buff=0.1) for k, s in ((0, "0"), (800, "800K"), (1600, "1.6M"))],
                       *[Text(str(a), font=MONO, font_size=16, color=GREY_B).next_to(pt(0, a), LEFT, buff=0.1) for a in (50, 70, 90)])
        xl = P("tokens spent building memory, per conversation", 18, GREY_B).move_to(np.array([3.6, -2.3, 0]))
        yl = P("accuracy", 18, GREY_B).next_to(pt(0, 90), UP, buff=0.12)
        pts = VGroup()
        for name, k, acc in RC_LOCOMO:
            col = RC if name == "RecMem" else GREY_B
            dt = Dot(pt(k, acc), radius=0.11 if name == "RecMem" else 0.08, color=col)
            lab = P(f"{name} {num(acc, '.2f')}", 17, col)
            side = {"Mem0": LEFT, "A-Mem": UP}.get(name, RIGHT)
            lab.next_to(dt, side, buff=0.12)
            pts.add(VGroup(dt, lab))
        self.play(FadeIn(rt), Create(axes), FadeIn(ticks), FadeIn(xl), FadeIn(yl), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(p, scale=0.8) for p in pts], lag_ratio=0.15), run_time=1.6)
        self.say("RecMem on LoCoMo scores 81.10 after spending 193.2K tokens building memory. Mem0 spends 1,520.8K and scores 62.92.")
        self.say("Full context still wins on LoCoMo's short chats of about 16K tokens, but then every question pays for the whole history.")
        self.end_section()

    def tradeoffs(self):
        """
        Description: Lays out the two methods' design choices as a seven-row table.
        Inputs: none
        Outputs: none
        """
        self.mark("tradeoffs")
        self.heading("Side by side")
        rows = (("memory unit", "events of keys and values", "turns, episodes and facts, as text"),
                ("written when", "surprise beats a moving threshold", "similar turns recur θ<sub>count</sub> times"),
                ("LLM calls to write", "none: graph math on keys", "only for topics that recur"),
                ("recall", "k-NN on keys, every layer and chunk", "embedding search, once per question"),
                ("model access", "open weights, attention patched", "any chat model plus an embedder"),
                ("main cost", "KV storage: 488 GB per 1M tokens", "LLM calls for each recurring topic"),
                ("blind spot", "no summaries and no merging", "one-off events stay raw"))
        xa, xe, xr, y0, dy = -6.6, -3.3, 1.75, 1.95, 0.6
        hdr = VGroup(P("EM-LLM", 26, EM).move_to(np.array([xe, 2.65, 0]), aligned_edge=LEFT),
                     P("RecMem", 26, RC).move_to(np.array([xr, 2.65, 0]), aligned_edge=LEFT))
        hdr[0].align_to(np.array([xe, 0, 0]), LEFT)
        hdr[1].align_to(np.array([xr, 0, 0]), LEFT)
        lines = VGroup()
        for k, (a, e, r) in enumerate(rows):
            y = y0 - k * dy
            g = VGroup(P(a, 20, GREY_B), P(e, 20), P(r, 20))
            for t, x in zip(g, (xa, xe, xr)):
                t.move_to(np.array([0, y, 0])).align_to(np.array([x, 0, 0]), LEFT)
            lines.add(g)
        rules = VGroup(*[Line(np.array([xa, y0 - (k + 0.5) * dy, 0]), np.array([6.8, y0 - (k + 0.5) * dy, 0])).set_stroke(GREY_T, 1, 0.5)
                         for k in range(len(rows) - 1)])
        self.play(FadeIn(hdr), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(l, shift=0.1 * RIGHT) for l in lines], lag_ratio=0.25), Create(rules), run_time=3)
        self.say("They work at different layers of the stack. EM-LLM changes attention inside one forward pass; "
                 "RecMem manages text between the turns of an agent.")
        self.say("So they could be stacked: RecMem for memory across sessions, and EM-LLM for reading one very long input.")
        self.end_section()

    def roots(self):
        """
        Description: Shows the psychology each method borrows, as two lanes of dated findings that end at each paper.
        Inputs: none
        Outputs: none
        """
        self.mark("roots")
        self.heading("Where the ideas come from")
        lanes = []
        for col, y, side, title, paper, roots in ((EM, 1.45, UP, "EM-LLM's roots", "EM-LLM\nICLR 2025", EM_ROOTS),
                                                   (RC, -0.7, DOWN, "RecMem's roots", "RecMem\nACL Findings 2026", RC_ROOTS)):
            t = P(title, 24, col).move_to(np.array([-5.45, y, 0]))
            line = Arrow(np.array([-4.25, y, 0]), np.array([4.0, y, 0]), buff=0, stroke_width=3, color=col,
                         tip_length=0.2, max_tip_length_to_length_ratio=0.05)
            pb = box(paper, 2.4, 0.95, col, 22).move_to(np.array([5.45, y, 0]))
            stations = VGroup()
            for k, (year, who, idea) in enumerate(roots):
                dot = Dot(np.array([-3.3 + 2.9 * k, y, 0]), radius=0.09, color=col)
                # One mobject per line, so each line is centred under the year.
                rows = [Text(year, font=MONO, font_size=20, color=col)]
                rows += [P(s, 16, GREY_B) for s in who.split("\n")] + [P(s, 18) for s in idea.split("\n")]
                lab = VGroup(*rows).arrange(DOWN, buff=0.06).next_to(dot, side, buff=0.18)
                stations.add(VGroup(dot, lab))
            lanes.append((t, line, pb, stations))
        self.play(*[FadeIn(t) for t, _, _, _ in lanes], *[GrowArrow(l) for _, l, _, _ in lanes], run_time=1)
        self.say("Each method borrows from a different corner of memory research.")
        for (t, line, pb, stations), text in zip(lanes, (
                "EM-LLM: people cut experience into events at prediction errors, surprise predicts where they "
                "mark boundaries, and recall drifts to neighbouring moments.",
                "RecMem: fast stores feed a stable long-term store, and the cortex slowly extracts what repeats. "
                "That is why recurrence triggers consolidation.")):
            self.play(LaggedStart(*[FadeIn(s, shift=0.1 * RIGHT) for s in stations], lag_ratio=0.3), run_time=1.5)
            self.play(FadeIn(pb, shift=0.1 * LEFT), run_time=0.6)
            self.say(text)
        self.end_section()

    def summary(self):
        """
        Description: Lists the video's conclusions and one labelled synthesis.
        Inputs: none
        Outputs: none
        """
        self.mark("summary")
        self.heading("Summary")
        lines = VGroup(*[P(s, 25) for s in (
            f'<span foreground="{EM}">EM-LLM</span> cuts one long input into events where the model is surprised,\n'
            "refines the cuts on attention keys, and recalls events per layer.",
            f'<span foreground="{RC}">RecMem</span> stores every turn as raw text and spends LLM calls only on\n'
            "topics that recur, turning them into episodes and facts.",
            "One keys on novelty and the other on repetition.",
        )])
        syn = P("Synthesis, not a claim in either paper: EM-LLM's surprise could be the salience signal\n"
                "that RecMem's limitations section asks for, so rare but important events get kept.", 23, AMBER)
        for ln in [*lines, syn]:
            if ln.width > 13:
                ln.scale_to_fit_width(13)
        VGroup(*lines, syn).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to(0.2 * UP)
        self.play(LaggedStart(*[FadeIn(ln, shift=0.1 * RIGHT) for ln in lines], lag_ratio=0.5), run_time=3)
        self.wait(9)
        self.play(FadeIn(syn, shift=0.1 * RIGHT), run_time=0.8)
        self.wait(7)
        self.end_section()

    def counters(self, calls):
        """
        Description: Builds the two LLM-call counters of the RecMem write demo.
        Inputs:
        - calls: [RecMem's calls so far, an eager system's calls so far].
        Outputs: the two-line text group.
        """
        g = VGroup(M(f"LLM calls · RecMem: {calls[0]}", 24, RC), M(f"LLM calls · eager memory: {calls[1]}", 24, BAD))
        return g.arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(np.array([3.6, -2.2, 0]))

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
        new = MarkupText(lining(text), font=SERIF, font_size=26)
        if new.width > 13.4:
            new = MarkupText(lining(wrap(text)), font=SERIF, font_size=26)
            if new.width > 13.4:
                new.scale_to_fit_width(13.4)
        new.to_edge(DOWN, buff=0.4)
        anims = [FadeIn(new, shift=0.15 * UP)]
        if self.cap is not None:
            anims.append(FadeOut(self.cap, shift=0.15 * UP))
        self.play(*anims, run_time=0.6)
        self.cap = new
        # No narration, so hold long enough to read (~20 characters a second);
        # captions that lead into an animation get less, the animation covers the rest.
        plain = len(text.replace("<sub>", "").replace("</sub>", ""))
        self.wait(max(wait, plain / 20) if wait else plain / 45)


def precompute():
    """
    Description: Computes every toy number on screen and asserts the claims the captions make about them.
    Inputs: none
    Outputs: a dict with the surprise stream, threshold, boundaries, key-similarity matrix, modularity curve and RecMem traces.
    """
    d = {}
    s = surprise_stream()
    thr = moving_threshold(s, TAU, GAMMA)
    bnd = [t for t in range(TAU, N_TOK) if s[t] > thr[t]]
    assert tuple(bnd) == SPIKES, f"surprise boundaries {bnd} differ from the designed spikes {SPIKES}"
    d.update(surprise=s, threshold=thr, boundaries=bnd)

    A = key_similarity()
    n = A.shape[0]
    # Algorithm 1 searches the late boundary over (alpha, beta]: after the previous boundary, up to its surprise position.
    q = {p: modularity(A, split_labels(n, (INIT_B[0], p))) for p in range(INIT_B[0] + 1, INIT_B[1] + 1)}
    best = max(q, key=q.get)
    assert best == BLOCKS[0] + BLOCKS[1], f"modularity peaks at {best}, not at the true boundary"
    assert INIT_B[1] - best == 2 and q[best] > q[INIT_B[1]]
    d.update(A=A, q_curve=q, init_b=INIT_B, best_b=best)

    tr = recmem_trace(THETA_COUNT)
    assert [t["kind"] for t in tr] == ["raw", "raw", "raw", "raw", "raw", "consolidate", "merge"], tr
    assert [t["r"] for t in tr[:6]] == [0, 0, 1, 0, 2, 3]
    d["trace7"] = tr

    bounds = [i for i in range(1, 6) if CHAT6[i][2] > THR6]
    assert bounds == [1, 2, 3, 4], bounds
    states = chat_trace(1)
    assert states == ["raw", "raw", "consolidate", "raw", "merge", "raw"], states
    d.update(chat_bounds=bounds, chat_states=states)
    return d


def surprise_stream():
    """
    Description: Builds the toy per-token surprise: a gentle wave with noise, plus a spike at each designed topic shift.
    Inputs: none
    Outputs: an array of N_TOK surprise values.
    """
    t = np.arange(N_TOK)
    s = 1.0 + 0.12 * np.sin(0.45 * t) + RNG.normal(0, 0.05, N_TOK)
    for k, b in enumerate(SPIKES):
        s[b] += 2.0 + 0.35 * (k % 2)
        s[b + 1] += 0.5
    return np.clip(s, 0.2, None)


def moving_threshold(s, tau, gamma):
    """
    Description: Computes EM-LLM's boundary threshold T_t = mean + gamma * std over the previous tau tokens.
    Inputs:
    - s: the surprise values.
    - tau: the window length.
    - gamma: the number of standard deviations.
    Outputs: an array of thresholds, infinite for the first tau tokens.
    """
    thr = np.full(len(s), np.inf)
    for t in range(tau, len(s)):
        w = s[t - tau:t]
        thr[t] = w.mean() + gamma * w.std()
    return thr


def key_similarity():
    """
    Description: Builds a symmetric toy key-similarity matrix, high inside each true event and low across events.
    Inputs: none
    Outputs: an n-by-n array with values in [0, 1].
    """
    n = sum(BLOCKS)
    lab = np.repeat(np.arange(len(BLOCKS)), BLOCKS)
    noise = RNG.uniform(0, 1, (n, n))
    noise = (noise + noise.T) / 2
    A = np.where(lab[:, None] == lab[None, :], 0.7 + 0.3 * noise, 0.05 + 0.2 * noise)
    np.fill_diagonal(A, 1.0)
    return A


def modularity(A, labels):
    """
    Description: Computes Newman's modularity of a partition of a weighted graph.
    Inputs:
    - A: the symmetric weight matrix.
    - labels: each node's community.
    Outputs: Q, the weight inside communities minus what a random graph with the same degrees would put there, over the total.
    """
    k = A.sum(1)
    two_m = A.sum()
    same = labels[:, None] == labels[None, :]
    return float(((A - np.outer(k, k) / two_m) * same).sum() / two_m)


def split_labels(n, cuts):
    """
    Description: Turns boundary positions into one community label per token.
    Inputs:
    - n: the number of tokens.
    - cuts: the boundary positions, each the first token of a new event.
    Outputs: an array of n labels.
    """
    lab = np.zeros(n, dtype=int)
    for c in cuts:
        lab[c:] += 1
    return lab


def recmem_trace(theta_count):
    """
    Description: Replays RecMem's write rule on the seven demo turns: merge into a matching episode first, else consolidate on recurrence.
    Inputs:
    - theta_count: how many earlier similar turns trigger consolidation.
    Outputs: one dict per turn with its kind ("raw", "consolidate" or "merge"), |R| and the earlier turns it matched.
    """
    out, episodes = [], []
    angs = [a for _, a in TURNS7]
    for i, a in enumerate(angs):
        if any(math.cos(math.radians(a - e)) >= THETA_SIM for e in episodes):
            out.append({"kind": "merge", "r": None, "matches": []})
            continue
        m = [j for j in range(i) if math.cos(math.radians(a - angs[j])) >= THETA_SIM]
        if len(m) >= theta_count:
            episodes.append(float(np.mean([angs[j] for j in m + [i]])))
            out.append({"kind": "consolidate", "r": len(m), "matches": m})
        else:
            out.append({"kind": "raw", "r": len(m), "matches": m})
    return out


def chat_trace(theta_count):
    """
    Description: Replays RecMem's write rule on the six-turn chat, where two turns are similar when they share a topic.
    Inputs:
    - theta_count: how many earlier same-topic turns trigger consolidation.
    Outputs: each turn's state, "raw", "consolidate" or "merge".
    """
    out, episodes = [], set()
    topics = [t for _, t, _ in CHAT6]
    for i, tp in enumerate(topics):
        if tp in episodes:
            out.append("merge")
        elif sum(topics[j] == tp for j in range(i)) >= theta_count:
            episodes.add(tp)
            out.append("consolidate")
        else:
            out.append("raw")
    return out


def box(label, w, h, colour, size=24):
    """
    Description: Draws a labelled rounded box.
    Inputs:
    - label: the text inside, as Pango markup, with \\n for line breaks.
    - w, h: the box's width and height.
    - colour: the border colour.
    - size: the label's font size.
    Outputs: the box and its label.
    """
    r = RoundedRectangle(corner_radius=0.1, width=w, height=h).set_stroke(colour, 2.5).set_fill(config.background_color, 1)
    t = P(label, size)
    if t.width > w - 0.2:
        t.scale_to_fit_width(w - 0.2)
    return VGroup(r, t.move_to(r))


def chip(state, x):
    """
    Description: Draws the RecMem status chip under one turn of the six-turn chat.
    Inputs:
    - state: "raw", "consolidate" or "merge".
    - x: the chip's centre x.
    Outputs: the chip.
    """
    text, col = {"raw": ("stays raw", GREY_T), "consolidate": ("episode + facts", RC), "merge": ("merged in", RC)}[state]
    r = RoundedRectangle(corner_radius=0.12, width=2.0, height=0.55).set_stroke(col, 2).set_fill(col, 0.15)
    return VGroup(r, P(text, 18, col if state != "raw" else GREY_B).move_to(r)).move_to(np.array([x, -1.65, 0]))


def arrow_between(a, b):
    """
    Description: Draws a grey arrow between two points.
    Inputs:
    - a, b: the start and end points.
    Outputs: the arrow.
    """
    return Arrow(a, b, buff=0.08, stroke_width=3, color=GREY_B, tip_length=0.18, max_tip_length_to_length_ratio=0.2)


def wrap(text):
    """
    Description: Breaks a long caption into two lines at the space nearest its middle.
    Inputs:
    - text: the caption.
    Outputs: the caption with one newline.
    """
    mid = len(text) // 2
    left, right = text.rfind(" ", 0, mid), text.find(" ", mid)
    cut = left if right < 0 or (left >= 0 and mid - left <= right - mid) else right
    return text[:cut] + "\n" + text[cut + 1:]


def lining(text):
    """
    Description: Sets the 0 of "Mem0" in Cambria, because Georgia's old-style zero reads as a letter o.
    Inputs:
    - text: Pango markup.
    Outputs: the markup with every "Mem0" fixed.
    """
    return text.replace("Mem0", 'Mem<span font_family="Cambria">0</span>')


def P(text, size=26, color=WHITE):
    """
    Description: Builds serif text from Pango markup.
    Inputs:
    - text: the markup.
    - size: the font size.
    - color: the default colour.
    Outputs: the text mobject.
    """
    return MarkupText(lining(text), font=SERIF, font_size=size, color=color)


def M(text, size=30, color=WHITE):
    """
    Description: Builds a formula from Pango markup in Cambria, so <sub> gives real subscripts.
    Inputs:
    - text: the markup.
    - size: the font size.
    - color: the default colour.
    Outputs: the text mobject.
    """
    return MarkupText(text, font=MATH, font_size=size, color=color)


def polar(c, r, deg):
    """
    Description: Finds the point at a given angle and distance from a centre.
    Inputs:
    - c: the centre.
    - r: the distance.
    - deg: the angle in degrees, counter-clockwise from the right.
    Outputs: the point.
    """
    return c + r * np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg)), 0])


def num(v, spec):
    """
    Description: Formats a number for display with a true minus sign.
    Inputs:
    - v: the number.
    - spec: a format spec, such as ".2f".
    Outputs: the formatted string.
    """
    return format(v, spec).replace("-", "−")
