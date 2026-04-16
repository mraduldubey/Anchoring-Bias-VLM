# Overview — narrative companion to the README

The main [README.md](README.md) is the formal, research-document-style summary of this repository. This page preserves the original walk-through-style intro for readers who want the plainer narrative, the condensed three-patterns framing, and the tabular reference index.

## Coming from the blog?

This repo contains every input and output from the experiments described in the posts. Nothing is summarized or paraphrased — you can read every prompt sent and every response received.

**Blog posts:**
- Part 1: *The More You Tell It, The Less It Sees* — defining anchoring bias in VLMs, the controlled experiment, and engineering guidance. Under publishing submission. [LINK]
- Upcoming Part 2: *The Failure Catalog* — identity misattribution and entity fabrication under stress-test conditions.

## Coming from GitHub?

Start with the blog. Part 1 presents the findings and context you need to make sense of this data. The short version:

I fed a VLM the same surveillance frames with structured object detection data (YOLOv8 + BoT-SORT). Without the data, the model confidently identified a shoplifting event. With the data, it declared "No suspicious activity observed." I designed experiments to understand why, and found three patterns:

1. **The delivery channel modulates the anchoring effect.** Same bounding-box data delivered as text, visual overlay, or cross-modal reference scheme produced dramatically different suppression of visual reasoning.
2. **Plausible metadata passes unchallenged.** The model doesn't verify metadata against the image — it only rejects what the image actively disproves.
3. **Every field of metadata has a cost.** In the scene-description task, the degradation is monotonic. Even minimal metadata reduces visual reasoning.

## Models tested

- **Gemini 3 Flash Preview** — primary model for EXP1
- **Gemini 2.5 Flash** — secondary model for EXP1 cross-validation; primary for gradient and failure-catalog experiments

## References cited in the blog (tabular index)

All references below are cited in Part 1. Part 2 cites no external papers. The main README's "Related work" section presents these as thematic prose; the tables below preserve the "where cited / what it's cited for" metadata useful for anyone hunting a specific citation.

### Anchoring bias in LLMs

| Citation | Where cited | What it's cited for | Link |
|----------|------------|--------------------|----|
| Jones & Steinhardt, 2022 | Pattern 1 "How this differs" | LLM anchoring baseline — models over-weight reference values in reasoning tasks | https://arxiv.org/abs/2202.12299 |

### VLM visual grounding & language-prior override

| Citation | Where cited | What it's cited for | Link |
|----------|------------|--------------------|----|
| "Seeing but Not Believing" (Li et al., 2025) | Pattern 1 "How this differs" | VLM visual encoders capture correct info but language backbone overrides during generation | https://arxiv.org/abs/2510.17771 |
| M3ID — Favero et al., CVPR 2024 | Pattern 1 "How this differs", Pattern 3 | VLM reliance on visual input decays as more output tokens are generated | https://arxiv.org/abs/2403.14003 |

### Visual prompting

| Citation | Where cited | What it's cited for | Link |
|----------|------------|--------------------|----|
| Set-of-Mark Prompting (Yang et al., 2023) | Pattern 1 "How this differs" | Visual overlays direct VLM attention to specific regions | https://arxiv.org/abs/2310.11441 |
| Contrastive Region Guidance (Wan et al., ECCV 2024) | Pattern 1 "How this differs" | Naive bounding-box overlays can hurt VLM performance | https://arxiv.org/abs/2403.02325 |
| "Biasing VLM Response with Visual Stimuli" | Pattern 1 "How this differs" | Visual highlighting shifts VLM answers toward marked options | https://www.lesswrong.com/posts/dktDLahikgoK4sen3/biasing-vlm-response-with-visual-stimuli |

### VLM hallucination

| Citation | Where cited | What it's cited for | Link |
|----------|------------|--------------------|----|
| Multi-Object Hallucination (Chen et al., NeurIPS 2024) | Pattern 3 | Hallucination rates increase with number of object categories | https://arxiv.org/abs/2407.06192 |

### Adversarial attacks on VLMs

| Citation | Where cited | What it's cited for | Link |
|----------|------------|--------------------|----|
| AdvEDM (NeurIPS 2025) | Pattern 2 "How this differs" | Injects/removes object semantics via adversarial image perturbations | https://arxiv.org/abs/2509.16645 |
| Shadowcast (NeurIPS 2024) | Pattern 2 "How this differs" | Data poisoning of VLMs at training time | https://vlm-poison.github.io |
| Prompt injection in oncology VLMs (Clusmann et al., Nature Communications, 2025) | Pattern 2 "How this differs" | Sub-visual adversarial prompts embedded in medical images | https://www.nature.com/articles/s41467-024-55631-x |
