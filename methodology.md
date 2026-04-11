# Methodology

## Models

- **Primary:** Gemini 3 Flash Preview (`gemini-3-flash-preview`) -- used for the EXP1 7-condition experiment
- **Secondary:** Gemini 2.5 Flash (`gemini-2.5-flash`) -- used for EXP1 cross-model validation, gradient experiment, and failure catalog stress tests
- All calls via Google GenAI Python SDK, `generate_content` with image + text input
- n=1 per condition (single API call). The pattern across conditions is the evidence, not any single response.

## Scenes

### Scene 1: Shoplifting (sample1, 85--95s)

Narrow retail clothing aisle. A woman in a red top selects a dark garment from a rack, conceals it in her shoulder bag around 90s, and walks away. 4-5 background people visible, two mannequins.

- **Ground truth event:** Concealment begins ~87s, completed ~89.4--90.5s, departure ~92.8s onward
- Used for: EXP1 (all 7 conditions), gradient (G0--G4), ghost (E1), wrong suspect (E2)

*All experiments in this repository use Scene 1.*

## Visual Input Format

All experiments use the same two-image input:
1. **Temporal grid:** 3x3 (EXP1) or 4x3 (earlier experiments) grid of uniformly-sampled frames from the time window, each timestamped
2. **Center frame:** Full video resolution (848x480) frame from the middle of the window

For EXP1, the grid is always clean (no overlays). The center frame varies by condition:
- Conditions 0, B, C: clean (no overlays)
- Conditions D, E: bounding boxes drawn with class labels and confidence scores
- Conditions F, G: bounding boxes drawn with ID labels only (D1, D2, etc.)

## Structured Data Format

Detection data is JSON, delivered in the text prompt. Format varies by experiment:

**EXP1:** Per-entity entries with fields varying by condition:
- Text conditions (B, C): `id`, `class`, `present`, `carrying`, `bbox` [x1, y1, x2, y2]
- Visual conditions (D, E): `id`, `class`, `present`, `carrying` (no bbox -- bboxes are on the image)
- Cross-modal conditions (F, G): `id`, `class`, `present`, `carrying`, `image_label` mapping

**Gradient (G1--G4):** Increasing density of per-track temporal samples:
- G1: track ID + time range + carrying items
- G2: + 4-5 position samples (first, quartiles, last)
- G3: every 10 frames, 6 numeric fields (cx, cy, bbox_area, norm_speed, norm_area_delta, carrying)
- G4: every 3 frames, same 6 fields

**Failure catalog (stress tests):**
- Wrong suspect (identity misattribution): Partial data -- only T1 (a mannequin) tracked. Text:image ratio ~1.65.
- Ghost (entity fabrication): Contradictory data -- T2's tracking fields frozen to one position. Text:image ratio ~7.32.

## Fabricated Detections (EXP1 conditions C, E, G)

Three fake entries added alongside real detections:

| Fake | Class | Confidence | Location | Why this location |
|------|-------|-----------|----------|------------------|
| F1/T89 | person | 0.52 | Empty right aisle [555,100,620,300] | No visual evidence -- tests implausible injection |
| F2/O412 | handbag | 0.47 | Floor near rack [150,400,210,460] | No person nearby -- tests isolated object injection |
| F3/O87 | cell phone | 0.44 | Near T2's hand [355,300,390,330] | Overlaps subject's hand -- tests plausible injection |

All at moderate confidence (0.44--0.52), same visual style as real detections when drawn.

## Prompt Template

All conditions use the same base prompt structure. The only variation is whether structured data is included and how it's framed:

```
You are analyzing a surveillance video.
The grid image shows N uniformly-sampled frames from a Xs window (start-end),
left-to-right, top-to-bottom, each timestamped.
[+ data framing if applicable]
[+ structured data JSON if applicable]
The data is ML-generated and may have minor noise.
Use the visual frames as ground truth when signals seem ambiguous.
Describe what is happening in this window. Include timestamps.
Focus on: what people are doing, notable events, any suspicious or significant activity.
```

Note: The instruction "use visual frames as ground truth when ambiguous" was present in all data conditions. It was not reliably followed under anchoring.

## VD/DN Scoring

See `scoring/rubric.md` for the full scoring methodology.

- **Visual Detail (VD):** Observations only derivable from pixels
- **Data Narration (DN):** Sentences referencing, restating, or citing provided data
- **VD/(VD+DN):** Ratio of visual reasoning to total output. 100% = pure visual. Lower = more data narration.

Manual scoring by a single scorer. No inter-rater reliability. Suitable for identifying patterns; not for precise quantification.

## Limitations

- Two models from one family (Gemini Flash variants). No GPT-4o, Claude, or open-source VLM testing.
- One primary scene (shoplifting) for all experiments.
- n=1 per condition. Stochastic variation means individual responses could differ on re-run.
- Manual VD/DN scoring with one scorer. Subjective boundary in some cases.
- The "use visual frames as ground truth" prompt instruction may interact with anchoring effects in ways I haven't isolated.
- Both tested models are multimodal LLMs with reasoning capabilities. Simpler VLM architectures may show different patterns.

## Reproducibility

All inputs (images, prompts, text data) and outputs (model responses) are included in this repository. To replicate:

1. Use the saved images and prompts from `experiments/exp1/` (or other experiment directories)
2. Send them to the same or different VLM via API
3. Score the response using the VD/DN rubric in `scoring/rubric.md`
4. Compare with the scored responses

The YOLO pipeline that generated the original detection data is not included (it lives in a separate repository). The detection data itself is included as JSON files.
