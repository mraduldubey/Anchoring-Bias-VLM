# Gradient Experiment — Data Density vs Visual Reasoning

**Model:** Gemini 2.5 Flash
**Design:** 5 data density levels (G0--G4) on two scenes. Measures how visual reasoning degrades as structured data volume increases.

## Density Levels

| Level | Data provided | Description |
|-------|--------------|-------------|
| G0 | None | Vision only |
| G1 | IDs + time range + carrying | Minimal metadata |
| G2 | + 4-5 position samples | Summary positions |
| G3 | + every 10 frames, 6 fields | Full Mode B density |
| G4 | + every 3 frames, 6 fields | 3x Mode B density |

## Sample 1 — Shoplifting (85--95s)

| Level | VD/(VD+DN) | Shoplifting detected? | Concealment timing | Notable artifact |
|-------|-----------|----------------------|-------------------|-----------------|
| G0 | ~100% | Yes -- confident | 91.7--92.8s (accurate) | None |
| G1 | ~85% | Yes -- confident | 92.8s (accurate) | Low data narration |
| G2 | ~65% | Yes -- with noise | 88.3--89.4s (slightly early) | Phone hallucination appears |
| G3 | ~45% | Yes -- but late | 92.7--93.0s (slightly late) | Heavy data citation throughout |
| G4 | ~20% | **No** -- "No suspicious activity" | -- | Latched on carrying-label artifact |

**G4 failure:** The model treated a YOLO misdetection (carrying label change "handbag" to "backpack" at 93.7s) as the primary event. The actual shoplifting, visible in the same frames, was not mentioned.

## Key Finding

The VD/(VD+DN) curve is monotonic. Every additional field of structured data costs visual perception. There is no level where "a little data helps" -- even G1 (minimal metadata) drops visual reasoning by ~15 points.

## Raw Data

Each level has its own directory under `sample1/` containing:
- `gradient_gN_prompt.txt` — the exact prompt sent
- `gradient_gN_input_data.json` — the structured data (absent for G0)
- `gradient_gN_response.txt` — the model's response
