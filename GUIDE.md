# MedFSCIL Studio — driving the app

A reference for you. The app itself is written to stand alone for anyone who opens it
cold, so nothing on screen assumes this document.

---

## The point

The model is never retrained. It turns each image into a list of numbers, and represents
a disease by the **average** of those numbers across its examples. Recognising an image
means finding the nearest average. So adding a disease is one averaging step — which is
why the timer reads in microseconds, and why the demo can do it live.

---

## Tab 1 — Teach a disease

The core demo. All six controls are live; changing any of them restarts the session.

| Control | Does |
|---|---|
| **Image collection** | Which of the four medical image sets. Changes the diseases and the starting/added split. |
| **Model** | Which frozen feature extractor. Tick two or more and they get combined. |
| **How to combine them** | Greyed out until 2+ are ticked. *Join their features* merges what each model saw before deciding; the other two let each decide and pool the votes. |
| **Examples per new disease** | 1–20. The few-shot constraint itself. |
| **Which examples get picked** | Changes which examples are drawn. On rare diseases this swings the result by ten points — that is a finding, not noise. |
| **Correction method** | How the new disease's average is nudged toward the ones it resembles, compensating for how noisy an average of five is. Set to *Off* to show what it's worth. |

**Buttons**: teach the next disease · teach all remaining · start over.

**The cards**: accuracy; balanced (all diseases weighted equally — matters because one
skin class is 67% of that test set); accuracy on diseases known from full data vs from a
few images; how many diseases it knows; and how long the last one took.

**The chips** show every disease colour-coded: solid teal = learned from thousands,
orange = learned from a handful, dashed = not shown yet.

**Charts** — accuracy as diseases are added; per-disease recognition rate; and the grid of
what gets mistaken for what (rows sum to 100%, diagonal is the recognition rate).

> **The thing to point at:** some diseases score badly even though the model saw thousands
> of examples. On the skin set, `bkl` sits near 24% and is a *solid teal* bar. The
> difficulty is not only scarcity.

---

## Tab 2 — Which model to use

Pick a collection; everything updates. You get a **decision card** (the verdict), a table,
and a chart. The table's last column is the important one: how much accuracy the
averaging step throws away, measured as the gap between what the pipeline scores now and
what a plain classifier reaches on exactly the same frozen numbers.

On the tissue-slide set the striped bars sit *below* the solid ones — retraining makes an
already-capable model worse. On the skin set they sit far above. Same code, opposite
answer. That contrast is the single best thing to demo here.

Detail is behind the **Why** panel.

---

## Tab 3 — Combining models

Pick models, press **Measure**. Cards give the combined score, the best single model, the
difference, the **ceiling** (share of images at least one model gets right), and what is
still unclaimed.

The chart counts how often two models disagree — high means they see different things,
which is the precondition for combining them helping at all. Orange ticks are stored
values from earlier runs, for cross-checking.

**Show images they read differently** is the tangible version: images where one model is
right and another is wrong, with a table of what each said. On the skin set that's **50.8%
of all test images**.

**Try every combination** ranks all of them. Slow on the tissue set — run it before you
present.

---

## Tab 4 — Where it goes wrong

Press **Examine mistakes**. Four cards: average certainty when right, when wrong, how
often it answers confidently, and **how many of those confident answers are wrong**.

Left chart: the two certainty distributions. Where they overlap, certainty can't separate
right from wrong. Right chart: if it only answered its most confident cases and passed the
rest to a person, how accurate would those be? Flat = the certainty score isn't worth
acting on.

The gallery shows mistakes sorted most-confident-first — the errors that would pass
unchallenged.

> On the retina set, ~38% of answers come out above 0.9 certainty and about one in four of
> those is wrong. That's the strongest argument in the whole demo.

---

## Tab 5 — Reference

Collections, models, and the stored five-run averages. The **How the numbers here are
produced** panel is the honesty statement — read it if anyone asks whether you retrained
anything.

---

## A 5-minute run

1. **Teach a disease**, skin set, one model. Press *Teach the next disease*. Point at the
   five images and the timer. *"That's the whole learning step."*
2. Twice more. Show the accuracy holding as the disease count grows.
3. Move **which examples get picked**. *"Same method, different five images, ten points of
   swing."*
4. **Which model to use** — switch between the tissue and skin sets. *"Same code, opposite
   recommendation."* Point at the striped bars below the solid ones.
5. **Combining models** — all three, measure, point at the ceiling, then show the images
   they read differently.
6. **Where it goes wrong** — retina. *"One in four confident answers is wrong."*

## Questions to expect

- *"Did you retrain anything?"* No. The models are frozen; adding a disease is arithmetic.
  The *adapted* rows in Reference are stored results from a slower approach, marked as such.
- *"Why doesn't this match your written results?"* The app runs one draw of examples that
  you control; the stored table averages five. Under matched settings the engine reproduces
  the recorded numbers to within 0.05 pp — see `README.md`.
- *"Why is accuracy low on the skin set?"* Tab 2 answers it: a plain classifier reaches
  79–82% on the same frozen numbers, so the loss is in the averaging step, not the model.
