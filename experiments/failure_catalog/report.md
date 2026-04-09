# Failure Catalog — Two Types of Failure When VLMs Trust Data Over Their Eyes

**Model:** Gemini 2.5 Flash
**Context:** These are stress tests -- deliberately adversarial conditions used during development to probe how the system breaks. They are not controlled experiments. They preceded the formal EXP1 study and are the observations that motivated its design.

**Conditions stacked:** Weaker model (Gemini 2.5 Flash vs Part 1's Gemini 3 Flash Preview), intentionally incorrect or incomplete tracking data.

---

## Failure type 1: Identity misattribution

**Experiment:** E2 -- Partial Data
**Scene:** Shoplifting (85--95s, sample1)
**Manipulation:** Tracking data included only for T1 (a mannequin in the background). No data for T2 (the actual shoplifter).
**Text:image token ratio:** ~1.65

The model correctly saw the shoplifting sequence -- someone took a garment. It assigned the entire action to T1:

> *"A woman (tracked as T1), wearing a red top and dark pants, is standing at a clothing rack."*

T1 is not a woman in red. T1 is a mannequin. In other conditions with full data, the same model family correctly identified T1 as a mannequin. But here, T1 was the only tracked entity -- and the tracking ID overrode visual perception.

Then the model cited T1's stationary data as confirmation:

> *"The tracking data confirms she remained stationary throughout this entire period of selection and concealment."*

T1's near-zero speed (a measurement of a mannequin being a mannequin) was repurposed as evidence of "careful, deliberate stillness during concealment."

**What this failure type looks like:** Tracking IDs act as identity magnets. Actions flow to whichever entity the data names, regardless of who -- or what -- actually performed them.

**Raw data:** `wrong_suspect/sample1/`

---

## Failure type 2: Entity fabrication

**Experiment:** E1 -- Contradictory Data (v2, internally consistent)
**Scene:** Shoplifting (85--95s, sample1)
**Manipulation:** T2's tracking data replaced with fabricated data: all fields frozen to one position (cx=307, cy=253), zero speed, zero area change.
**Text:image token ratio:** ~7.32 (high -- includes full tracking data for all entities)

The model invented two people:
- **"Foreground Activity (Woman in Red -- Not explicitly tracked)"** -- the real shoplifter, demoted to untracked. Shoplifting rewritten as "appears to place them back."
- **"Mid-ground Activity (Track T2 -- Woman in Pink)"** -- a ghost entity. The frozen coordinates point center-left (nearest T4, red dress mannequin), but the model described its ghost as "Woman in Pink" on the "right side" -- matching T1 (pink mannequin), 240px from the data's coordinates. The model recruited visual appearance from across the frame.

Conclusion: *"There is no overtly suspicious or significant activity observed."*

**What this failure type looks like:** When data describes an entity that doesn't match anything visible, the model fabricates a visual counterpart rather than question the data. It stitches together appearance from one location and identity from another to produce a coherent but fictional entity.

**Raw data:** `ghost/v2/`

---

## Root cause

Both failures share one mechanism: **the model treats structured data as ground truth and adjusts its visual interpretation to match.**

- **Identity misattribution** bends *who* to fit the data.
- **Entity fabrication** bends *what exists* to fit the data.

The model never says "the data appears to be wrong based on what I can see."
