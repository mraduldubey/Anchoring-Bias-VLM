# EXP1 Results — Channel Differentiation of Anchoring Bias

**Scene:** Shoplifting surveillance (85s--95s). Woman in red top selects dark garment, conceals in shoulder bag ~90s, walks away.
**Models:** Gemini 3 Flash Preview (primary), Gemini 2.5 Flash (secondary)
**Design:** 7 conditions, same images, varying bbox delivery channel. 3 conditions include fabricated detections.

## Condition Matrix

| Condition | Name | Center image | Text data | Bbox delivery |
|-----------|------|-------------|-----------|--------------|
| 0 | baseline_vision_only | Clean | None | -- |
| B | text_bbox_real | Clean | Real detections + bboxes | Text JSON |
| C | text_bbox_fake | Clean | Real + 3 fakes + bboxes | Text JSON |
| D | visual_bbox_real | Real bboxes drawn | Real detections (no bboxes) | Visual overlay |
| E | visual_bbox_fake | Real + fake bboxes drawn | Real detections (no bboxes) | Visual overlay |
| F | crossmodal_bbox_real | ID labels drawn | Real detections + ID:label map | Cross-modal |
| G | crossmodal_bbox_fake | Real + fake ID labels | Real + fakes + ID:label map | Cross-modal |

## Injected Fake Detections (conditions C, E, G)

| Fake | Class | Confidence | Location | Rationale |
|------|-------|-----------|----------|-----------|
| F1/T89 | person | 0.52 | Empty right aisle | No visual evidence -- tests implausible injection |
| F2/O412 | handbag | 0.47 | Floor near rack | No person nearby -- tests isolated object injection |
| F3/O87 | cell phone | 0.44 | Near T2's hand | Overlaps subject's hand region -- tests plausible injection |

## Results — Gemini 3 Flash Preview

| Condition | VD | DN | VD/(VD+DN) | Shoplifting | Phone adopted | Person adopted | Bag adopted |
|-----------|----|----|-----------|------------|--------------|---------------|------------|
| 0 (baseline) | 16 | 0 | **100%** | Yes -- confident | -- | -- | -- |
| D (visual bbox) | 11 | 5 | **69%** | Yes -- "consistent with shoplifting" | -- | -- | -- |
| C (text + fake) | 9 | 6 | **60%** | Hedged -- "notable for security" | Yes | No | No |
| E (visual + fake) | 8 | 6 | **57%** | No -- "typical shopper" | Yes | No | No |
| B (text bbox) | 8 | 7 | **53%** | Hedged -- "noteworthy" | -- | -- | -- |
| F (crossmodal) | 7 | 8 | **47%** | No -- fabricated movement | -- | -- | -- |
| G (crossmodal + fake) | 6 | 9 | **40%** | No -- "No suspicious activity" | Yes | No | No |

## Results — Gemini 2.5 Flash

| Condition | VD | DN | VD/(VD+DN) | Shoplifting | Phone adopted | Person adopted | Bag adopted |
|-----------|----|----|-----------|------------|--------------|---------------|------------|
| 0 (baseline) | 14 | 0 | **100%** | No -- hedged | -- | -- | -- |
| D (visual bbox) | 8 | 7 | **53%** | Hedged -- "potentially suspicious" | -- | -- | -- |
| C (text + fake) | 5 | 8 | **38%** | No | Yes | Partial | No |
| B (text bbox) | 5 | 9 | **36%** | No | -- | -- | -- |
| E (visual + fake) | 5 | 9 | **36%** | No | Yes | No | No |
| F (crossmodal) | 4 | 10 | **29%** | No -- fabricated rummaging | -- | -- | -- |
| G (crossmodal + fake) | 4 | 11 | **27%** | No | Yes | Yes | Yes |

## Input Token Breakdown

Image tokens constant at 1,032 (516 per image x 2). Only text tokens vary:

| Condition | Text tokens | Text:Image ratio |
|-----------|------------|-----------------|
| 0 (baseline) | 111 | 0.11 |
| D (visual bbox) | 294 | 0.28 |
| F (crossmodal) | 328 | 0.32 |
| B (text bbox) | 353 | 0.34 |
| G (crossmodal + fake) | 390 | 0.38 |
| C (text + fake) | 423 | 0.41 |

Key: F (328 tokens) anchors harder than B (353 tokens) despite fewer text tokens. The channel effect is not reducible to token count.

## Key Findings

### 1. Channel ranking: visual < text < crossmodal
Same spatial information, three delivery channels, dramatically different anchoring. Visual overlays preserve 69% visual reasoning (3-flash) vs 53% for text vs 47% for crossmodal. The ranking holds across both models.

### 2. Fake phone adopted universally; fake person/bag rejected
The cell phone (bbox overlapping T2's hand) was adopted in every injection condition by both models. The person and bag (in empty regions) were rejected by 3-flash and mostly rejected by 2.5-flash. Positional plausibility -- not confidence -- is the adoption gate.

### 3. Cross-modal delivery produces fabrication
Both models fabricated observations under cross-modal load:
- 3-flash F: invented "back-and-forth movement" that never occurred
- 2.5-flash F: invented "crouching on ground, rummaging through objects"

### 4. Weaker model = more vulnerable
2.5-flash VD ratios are lower across every condition. It adopted all 3 fakes in condition G (inventing "mannequin T89" and "empty red box O412"). Weaker visual grounding lowers the plausibility gate.

### 5. Selective data criticism asymmetry
3-flash correctly identified mannequins misclassified as persons but uncritically adopted the fake phone at lower confidence. The model corrects data contradicted by strong visual evidence but trusts data in ambiguous regions.

## Limitations

- Single scene (shoplifting). Needs multi-scene replication.
- n=1 per condition per model (single API call).
- Manual VD/DN scoring, single scorer.
- "Use visual frames as ground truth" instruction was present in all data conditions and was not reliably followed.

## Raw Data

- Prompts: `prompts/exp1_{0,b,c,d,e,f,g}_prompt.txt`
- Text data: `text_data/exp1_{0,b,c,d,e,f,g}_text_data.json`
- Images: `images/` (grid + center frames)
- Responses: `responses/{gemini-3-flash-preview,gemini-2.5-flash}/`
- Detection references: `reference/`
