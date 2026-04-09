# Anchoring Bias in Vision-Language Models

Companion repository for the blog series on how structured detection data suppresses visual reasoning in VLMs.

## Coming from the blog?

This repo contains every input and output from the experiments described in the posts. Nothing is summarized or paraphrased -- you can read every prompt sent and every response received.

**Blog posts:**
- [Part 1: The More You Tell It, The Less It Sees](LINK) -- defining anchoring bias in VLMs, the controlled experiment, and engineering guidance
- [Part 2: The Failure Catalog](LINK) -- identity misattribution and entity fabrication under stress test conditions

## Coming from GitHub?

Start with the blog. [Part 1](LINK) presents the findings and context you need to make sense of this data. The short version:

I fed a VLM the same surveillance frames with structured object detection data (YOLOv8 + BoT-SORT). Without the data, the model confidently identified a shoplifting event. With the data, it declared "No suspicious activity observed." I designed experiments to understand why, and found three patterns:

1. **The delivery channel modulates the anchoring effect.** Same bounding box data delivered as text, visual overlay, or cross-modal reference scheme produced dramatically different suppression of visual reasoning.
2. **Plausible metadata passes unchallenged.** The model doesn't verify metadata against the image -- it only rejects what the image actively disproves.
3. **Every field of metadata has a cost.** In the scene description task, the degradation is monotonic. Even minimal metadata reduces visual reasoning.

## Repository structure

| You want to see... | Go to... |
|-------------------|----------|
| How the experiment was designed | [methodology.md](methodology.md) |
| Key results tables | [results_summary.md](results_summary.md) |
| How VD/DN was scored | [scoring/rubric.md](scoring/rubric.md) |
| All scores in one file | [scoring/scores.csv](scoring/scores.csv) |
| The 7-condition channel experiment (Part 1) | [experiments/exp1/](experiments/exp1/) |
| The data density gradient (Part 1, Pattern 3) | [experiments/gradient/](experiments/gradient/) |
| Identity misattribution & entity fabrication (Part 2) | [experiments/failure_catalog/](experiments/failure_catalog/) |
| Scripts used to generate inputs and run experiments | [scripts/](scripts/) |

### EXP1 condition reference

| Code | Center image | Bbox delivery |
|------|-------------|--------------|
| 0 (baseline) | Clean | None |
| B | Clean | Text JSON |
| C | Clean | Text JSON (+ 3 fakes) |
| D | Bboxes drawn | Visual overlay |
| E | Bboxes drawn (+ 3 fakes) | Visual overlay |
| F | ID labels drawn | Cross-modal (image IDs + text map) |
| G | ID labels drawn (+ 3 fakes) | Cross-modal (image IDs + text map) |

### Failure catalog experiments

| Experiment | What was manipulated | Blog section |
|-----------|---------------------|-------------|
| wrong_suspect | Tracking data for only T1 (a mannequin). No data for T2 (shoplifter). | Part 2: Identity misattribution |
| ghost | T2's tracking data replaced with fabricated frozen-position data. | Part 2: Entity fabrication |

## Models tested

- **Gemini 3 Flash Preview** -- primary model for EXP1
- **Gemini 2.5 Flash** -- secondary model for EXP1 cross-validation; primary for gradient and failure catalog experiments

## Replication

All experiment inputs (images, prompts, text data) are included. To replicate on your own model:

1. Pick an experiment directory (e.g., `experiments/exp1/`)
2. For each condition: send the grid image + center image + prompt + text data to your VLM
3. Score the response using the [VD/DN rubric](scoring/rubric.md)
4. Compare with the responses in `responses/`

The scripts in `scripts/` generated the experiment inputs but require the full YOLO pipeline to run. You don't need them for replication -- all generated artifacts are already here.

## Caveats

This is an engineer's investigation, not a peer-reviewed study. Two models from one family, one primary scene, n=1 per condition. The patterns are consistent and reproducible within scope. They are not a proof of universality. Replicate before you ship.

## License

[CC-BY-4.0](LICENSE) -- use freely with attribution.

## References cited in the blog

All references below are cited in Part 1. Part 2 cites no external papers.

### Anchoring bias in LLMs

| Citation | Where cited | What it's cited for | Link |
|----------|------------|--------------------|----|
| Jones & Steinhardt, 2022 | Pattern 1 "How this differs" | LLM anchoring baseline — models over-weight reference values in reasoning tasks | https://arxiv.org/abs/2206.02339 |

### VLM visual grounding & language-prior override

| Citation | Where cited | What it's cited for | Link |
|----------|------------|--------------------|----|
| "Seeing but Not Believing" (Li et al., 2025) | Pattern 1 "How this differs" | VLM visual encoders capture correct info but language backbone overrides during generation | https://arxiv.org/abs/2510.17771 |
| M3ID — Favero et al., CVPR 2024 | Pattern 1 "How this differs", Pattern 3 | VLM reliance on visual input decays as more output tokens are generated | https://arxiv.org/abs/2403.14003 |

### Visual prompting

| Citation | Where cited | What it's cited for | Link |
|----------|------------|--------------------|----|
| Set-of-Mark Prompting (Yang et al., 2023) | Pattern 1 "How this differs" | Visual overlays direct VLM attention to specific regions | https://arxiv.org/abs/2310.11441 |
| Contrastive Region Guidance (Wan et al., ECCV 2024) | Pattern 1 "How this differs" | Naive bounding box overlays can hurt VLM performance | https://arxiv.org/abs/2403.02325 |
| "Biasing VLM Response with Visual Stimuli" | Pattern 1 "How this differs" | Visual highlighting shifts VLM answers toward marked options | https://www.lesswrong.com/posts/dktDLahikgoK4sen3/biasing-vlm-response-with-visual-stimuli |

### VLM hallucination

| Citation | Where cited | What it's cited for | Link |
|----------|------------|--------------------|----|
| Multi-Object Hallucination (Chen et al., NeurIPS 2024) | Pattern 3 | Hallucination rates increase with number of object categories | https://arxiv.org/abs/2406.11649 |

### Adversarial attacks on VLMs

| Citation | Where cited | What it's cited for | Link |
|----------|------------|--------------------|----|
| AdvEDM (NeurIPS 2025) | Pattern 2 "How this differs" | Injects/removes object semantics via adversarial image perturbations | https://arxiv.org/abs/2509.16645 |
| Shadowcast (NeurIPS 2024) | Pattern 2 "How this differs" | Data poisoning of VLMs at training time | https://vlm-poison.github.io |
| Prompt injection in oncology VLMs (Nature Communications, 2024) | Pattern 2 "How this differs" | Sub-visual adversarial prompts embedded in medical images | https://www.nature.com/articles/s41467-024-52258-2 |
