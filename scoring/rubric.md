# VD/DN Scoring Rubric

## Overview

Every model response is scored on two dimensions to measure how much of the output comes from looking at the image versus narrating the provided data.

## Visual Detail (VD)

A sentence (or clause) counts as VD if the observation is **only derivable from pixels** -- it could not be produced from the structured data alone.

Examples:
- Posture descriptions: "she bends down," "crouching near the rack"
- Gaze and attention: "looks directly at the camera," "checking her surroundings"
- Hand/arm actions: "stuffing the garment into her bag," "reaching for the rack"
- Clothing descriptions not in the data: "red top," "dark pants," "colorful pink outfit"
- Spatial relationships from the image: "standing near the mannequin," "walking down the hallway"
- Environmental details: "narrow retail aisle," "patterned pavement," "parked motorcycles"
- Temporal observations from the grid: "by 90.0s she has moved to..."

## Data Narration (DN)

A sentence (or clause) counts as DN if it **references, restates, or cites provided detection data** -- track IDs, bounding box coordinates, carrying labels, confidence scores, or motion signals.

Examples:
- Track ID citations: "T2," "O164," "D1"
- Bbox/coordinate references: "cx: 700.0," "bbox_area: 2308"
- Carrying label imports: "carrying a backpack (from detection data)"
- Confidence score references: "cell phone (confidence 0.44)"
- Data quality observations: "the detection system misidentifies these as persons"
- Direct data restating: "the tracking data confirms she remained stationary"

## Scoring Rules

1. **Unit of scoring:** Individual sentences or independent clauses. A sentence with both VD and DN content is split and each part counted separately.
2. **Ambiguous cases:** If an observation could come from either pixels or data, score as VD if the specific detail (e.g., posture, timing precision) goes beyond what the data contains.
3. **Conclusions and interpretations** (e.g., "suggests shoplifting") are scored based on what evidence they cite -- visual evidence = VD, data evidence = DN.
4. **Boilerplate** (e.g., "Here is a description of the scene") is excluded from both counts.

## The VD/(VD+DN) Ratio

- **100%** = pure visual reasoning, no data narration
- **~70%** = mostly looking, some data reference
- **~50%** = equal parts looking and narrating
- **~30%** = mostly narrating the data
- **~20%** = the model is reading back its input

In the experiments, shoplifting detection survives above ~60% and is lost below ~50%.

## Limitations

- Manual scoring by a single scorer. No inter-rater reliability calculated.
- Subjective boundary between VD and DN in some cases.
- Suitable for blog-level evidence. Would need inter-rater agreement and larger n for publication.
