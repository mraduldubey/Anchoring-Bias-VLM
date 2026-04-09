# Scripts

These scripts are included for methodology transparency. They were used to generate the experiment inputs and run the VLM calls.

**They cannot be run standalone.** They depend on:
- The full video-search pipeline (YOLOv8 tracking CSVs, video files)
- A Google GenAI API key configured for Gemini access
- Python packages: `google-genai`, `opencv-python`, `pandas`, `numpy`, `Pillow`

## Files

| Script | Purpose |
|--------|---------|
| `prepare_exp1.py` | Generates EXP1 images (grid, center frames with overlays), prompts, and text data JSONs |
| `run_exp1.py` | Runs all 7 EXP1 conditions against the Gemini API and saves responses |
| `experiment_anchoring.py` | Runs E0-E3 experiments (elephant room, contradictory data, partial data, gradient) |

## If you want to replicate

The experiment inputs (images, prompts, text data) and outputs (model responses) are all included in the `experiments/` directory. You can re-run the VLM calls using the saved prompts and images without needing the YOLO pipeline.
