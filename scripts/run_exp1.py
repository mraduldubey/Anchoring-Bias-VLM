"""EXP1 — Runner script.

Calls Gemini for each condition and saves responses + a conclusion report.

Conditions:
  exp1_0  baseline_vision_only       — clean images, no text data
  exp1_b  text_bbox_real             — clean images + text data with bbox coords
  exp1_c  text_bbox_fake             — clean images + text data with injected bboxes
  exp1_d  visual_bbox_real           — center with labeled bbox overlays + base text
  exp1_e  visual_bbox_fake           — center with injected labeled overlays + base text
  exp1_f  crossmodal_bbox_real       — center with ID overlays + text ID:label map
  exp1_g  crossmodal_bbox_fake       — center with injected ID overlays + text map

Usage:
  # Single model
  python blog/anchoring_bias/src/run_exp1.py

  # Specific model
  python blog/anchoring_bias/src/run_exp1.py --model gemini-2.5-flash

  # Multiple models (runs all conditions on each)
  python blog/anchoring_bias/src/run_exp1.py --models gemini-2.5-pro,gemini-2.5-flash

  # Single condition (for testing)
  python blog/anchoring_bias/src/run_exp1.py --condition 0

Outputs per model go to: blog/anchoring_bias/data/exp1/{model_name}/
  exp1_{letter}_response.txt
  exp1_report.md
"""

import argparse
import json
import os
import time

from google import genai
from google.genai import types

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(ROOT, "blog/anchoring_bias/data/exp1")

# ── condition definitions ─────────────────────────────────────────────────────
CONDITIONS = {
    "0": {
        "name":        "baseline_vision_only",
        "center_img":  "exp1_0_center.jpg",       # clean
        "text_data":   "exp1_0_text_data.json",   # null
        "prompt_file": "exp1_0_prompt.txt",
    },
    "b": {
        "name":        "text_bbox_real",
        "center_img":  "exp1_0_center.jpg",       # clean
        "text_data":   "exp1_b_text_data.json",
        "prompt_file": "exp1_b_prompt.txt",
    },
    "c": {
        "name":        "text_bbox_fake",
        "center_img":  "exp1_0_center.jpg",       # clean
        "text_data":   "exp1_c_text_data.json",
        "prompt_file": "exp1_c_prompt.txt",
    },
    "d": {
        "name":        "visual_bbox_real",
        "center_img":  "exp1_d_center.jpg",       # real bbox overlays with labels
        "text_data":   "exp1_d_text_data.json",
        "prompt_file": "exp1_d_prompt.txt",
    },
    "e": {
        "name":        "visual_bbox_fake",
        "center_img":  "exp1_e_center.jpg",       # real + injected bbox overlays
        "text_data":   "exp1_e_text_data.json",
        "prompt_file": "exp1_e_prompt.txt",
    },
    "f": {
        "name":        "crossmodal_bbox_real",
        "center_img":  "exp1_f_center.jpg",       # real ID overlays
        "text_data":   "exp1_f_text_data.json",
        "prompt_file": "exp1_f_prompt.txt",
    },
    "g": {
        "name":        "crossmodal_bbox_fake",
        "center_img":  "exp1_g_center.jpg",       # real + injected ID overlays
        "text_data":   "exp1_g_text_data.json",
        "prompt_file": "exp1_g_prompt.txt",
    },
}

GRID_IMG = "exp1_shared_grid.jpg"


# ── Gemini caller ─────────────────────────────────────────────────────────────

def call_gemini(grid_path, center_path, prompt, model):
    """Call Gemini with two images (grid + center frame) and a text prompt."""
    with open(grid_path, "rb") as f:
        grid_bytes = f.read()
    with open(center_path, "rb") as f:
        center_bytes = f.read()

    grid_part   = types.Part.from_bytes(data=grid_bytes,   mime_type="image/jpeg")
    center_part = types.Part.from_bytes(data=center_bytes, mime_type="image/jpeg")

    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=[grid_part, center_part, prompt],
    )
    return response.text


# ── runner ────────────────────────────────────────────────────────────────────

def run_condition(letter, model, out_dir):
    """Run a single condition. Returns (letter, name, response_text)."""
    cond = CONDITIONS[letter]
    name = cond["name"]

    grid_path   = os.path.join(DATA_DIR, GRID_IMG)
    center_path = os.path.join(DATA_DIR, cond["center_img"])
    prompt_path = os.path.join(DATA_DIR, cond["prompt_file"])

    # Load prompt (user may have edited it)
    with open(prompt_path) as f:
        prompt = f.read()

    # Load token count for logging
    tc_path = os.path.join(DATA_DIR, "exp1_token_counts.json")
    token_counts = {}
    if os.path.exists(tc_path):
        with open(tc_path) as f:
            token_counts = json.load(f)
    tc_key = f"exp1_{letter}_{name}"
    total_tokens = token_counts.get(tc_key, {}).get("total_input_tokens", "?")

    print(f"\n  [exp1_{letter}] {name}")
    print(f"    center: {cond['center_img']}  |  ~{total_tokens} input tokens")
    print(f"    model:  {model}")
    print(f"    calling gemini...", end="", flush=True)

    t0 = time.time()
    response = call_gemini(grid_path, center_path, prompt, model)
    elapsed = time.time() - t0
    print(f" done ({elapsed:.1f}s)")

    # Save response
    resp_path = os.path.join(out_dir, f"exp1_{letter}_response.txt")
    with open(resp_path, "w") as f:
        f.write(response)
    print(f"    saved: {os.path.relpath(resp_path, ROOT)}")

    return letter, name, response


# ── report builder ────────────────────────────────────────────────────────────

def build_report(results, model, out_dir):
    """Generate exp1_report.md with all responses and a scoring template."""

    tc_path = os.path.join(DATA_DIR, "exp1_token_counts.json")
    with open(tc_path) as f:
        token_counts = json.load(f)

    lines = [
        f"# EXP1 Results Report",
        f"",
        f"**Model:** {model}  ",
        f"**Scene:** Shoplifting surveillance (85s–95s, sample1.mp4)  ",
        f"**Ground truth:** Woman in red top conceals garment in shoulder bag ~90s, walks away",
        f"",
        f"---",
        f"",
        f"## Input Token Summary",
        f"",
        f"| Condition | Name | Grid tokens | Center tokens | Text tokens | Total |",
        f"|-----------|------|------------|--------------|------------|-------|",
    ]

    for letter, cond in CONDITIONS.items():
        name = cond["name"]
        key  = f"exp1_{letter}_{name}"
        tc   = token_counts.get(key, {})
        lines.append(
            f"| exp1_{letter} | {name} "
            f"| {tc.get('grid_image_tokens','?')} "
            f"| {tc.get('center_image_tokens','?')} "
            f"| {tc.get('text_tokens','?')} "
            f"| {tc.get('total_input_tokens','?')} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## Scoring Template",
        f"",
        f"Score each response after reading. ",
        f"- **VD** Visual Detail: observations derivable only from pixels",
        f"- **DN** Data Narration: sentences referencing detection labels/boxes/IDs",
        f"- **Shoplifting**: detected (yes/no) + timing accuracy",
        f"- **Injected adoption** (C/E/G only): did VLM mention T89 / O412 / O87?",
        f"",
        f"| Condition | VD | DN | VD/(VD+DN) | Shoplifting | Injected adoption | Notes |",
        f"|-----------|----|----|-----------|------------|------------------|-------|",
    ]
    for letter, cond in CONDITIONS.items():
        inj = "C/E/G" if letter in ("c", "e", "g") else "—"
        lines.append(f"| exp1_{letter} | | | | | {inj} | |")

    lines += ["", "---", ""]

    # Embed each response
    for letter, name, response in results:
        cond = CONDITIONS[letter]
        lines += [
            f"## exp1_{letter} — {name}",
            f"",
            f"**Center image:** `{cond['center_img']}`  ",
            f"**Prompt:** `{cond['prompt_file']}`",
            f"",
            f"### Response",
            f"",
            response.strip(),
            f"",
            f"### Scores",
            f"",
            f"- VD: ",
            f"- DN: ",
            f"- VD/(VD+DN): ",
            f"- Shoplifting detected: ",
            f"- Timing: ",
        ]
        if letter in ("c", "e", "g"):
            lines += [
                f"- T89 (injected person) mentioned: ",
                f"- O412 (injected handbag) mentioned: ",
                f"- O87 (injected cell phone) mentioned: ",
            ]
        lines += ["", "---", ""]

    report = "\n".join(lines)
    report_path = os.path.join(out_dir, "exp1_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved: {os.path.relpath(report_path, ROOT)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run EXP1 anchoring bias experiments")
    parser.add_argument("--model",     default="gemini-2.5-pro",
                        help="Single model to use (default: gemini-2.5-pro)")
    parser.add_argument("--models",    default=None,
                        help="Comma-separated list of models, e.g. gemini-2.5-pro,gemini-2.5-flash")
    parser.add_argument("--condition", default=None,
                        help="Run a single condition only, e.g. --condition 0")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else [args.model]
    letters = [args.condition] if args.condition else list(CONDITIONS.keys())

    # Validate
    for letter in letters:
        if letter not in CONDITIONS:
            print(f"Unknown condition '{letter}'. Valid: {list(CONDITIONS.keys())}")
            return
    for path in [os.path.join(DATA_DIR, GRID_IMG)]:
        if not os.path.exists(path):
            print(f"Missing: {path}. Run prepare_exp1.py first.")
            return

    for model in models:
        model_slug = model.replace("/", "-")
        out_dir = os.path.join(DATA_DIR, model_slug)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"Output: blog/anchoring_bias/data/exp1/{model_slug}/")
        print(f"Conditions: {letters}")
        print(f"{'='*60}")

        results = []
        for letter in letters:
            try:
                result = run_condition(letter, model, out_dir)
                results.append(result)
            except Exception as e:
                print(f"\n  [exp1_{letter}] ERROR: {e}")

        if results:
            build_report(results, model, out_dir)

    print("\n✓ All done.")


if __name__ == "__main__":
    main()
