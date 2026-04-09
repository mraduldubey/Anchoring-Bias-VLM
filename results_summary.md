# Results Summary

Key findings tables from the anchoring bias experiments. For full analysis, see the reports in each experiment directory.

## EXP1: The delivery channel modulates the anchoring effect (Gemini 3 Flash Preview)

Same bounding box data, three delivery channels, dramatically different outcomes.

| Condition | VD/(VD+DN) | Shoplifting detected? | Fake phone adopted? |
|-----------|-----------|----------------------|-------------------|
| Baseline (no data) | **100%** | Yes -- confident, precise timing | -- |
| D (visual bbox) | **69%** | Yes -- "consistent with shoplifting" | -- |
| C (text bbox + fake) | **60%** | Hedged -- "notable for security" | Yes |
| E (visual bbox + fake) | **57%** | No -- "typical shopper" | Yes |
| B (text bbox) | **53%** | Hedged -- "noteworthy" | -- |
| F (crossmodal) | **47%** | No -- fabricated movement | -- |
| G (crossmodal + fake) | **40%** | No -- "No suspicious activity" | Yes |

**Channel ranking:** Visual overlays (69%) > Text coordinates (53%) > Cross-modal ID mapping (47%)

## EXP1: Cross-model comparison

| Dimension | Gemini 3 Flash Preview | Gemini 2.5 Flash |
|-----------|----------------------|-----------------|
| Baseline shoplifting | Confident detection | Hedged -- missed |
| Channel ranking | visual < text < crossmodal | Same direction |
| Fake phone adoption | All injection conditions | All injection conditions |
| Fake person adoption | Never | Adopted in G as "mannequin" |
| Fake bag adoption | Never | Adopted in G as "empty red box" |

## Gradient: Every field of metadata has a cost

| Level | Data density | VD/(VD+DN) | Shoplifting detected? |
|-------|-------------|-----------|----------------------|
| G0 | None | ~100% | Yes |
| G1 | Minimal metadata | ~85% | Yes |
| G2 | Summary positions | ~65% | Yes (timing drifts) |
| G3 | Full tracking | ~45% | Yes (late, heavy data citation) |
| G4 | Dense (3x) | ~20% | **No** |

The curve is monotonic in this scene description task. No sweet spot.

## Token ratio analysis

| Condition | Text tokens | Text:Image ratio | VD/(VD+DN) |
|-----------|------------|-----------------|-----------|
| Baseline | 111 | 0.11 | 100% |
| D (visual bbox) | 294 | 0.28 | 69% |
| F (crossmodal) | 328 | 0.32 | 47% |
| B (text bbox) | 353 | 0.34 | 53% |

F anchors harder than B despite fewer tokens. Channel effect is not reducible to token count.

## Failure catalog (stress tests)

| Failure type | What happens | Text:Image ratio |
|-------------|-------------|-----------------|
| Identity misattribution | Model attributes shoplifting to a mannequin (only tracked entity) | ~1.65 |
| Entity fabrication | Model invents a fictional person to match fabricated tracking data | ~7.32 |

These are stress test results under deliberately adversarial conditions (weaker model, fabricated/partial data), not production benchmarks.
