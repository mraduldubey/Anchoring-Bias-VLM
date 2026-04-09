"""Anchoring bias experiments for VLMs.

Experiments:
  E0 — "The Elephant in the Room": staged demo with user-provided image
  E1 — Contradictory data: planted wrong tracking data for specific tracks
  E2 — Partial data: tracking data for a subset of visible entities
  E3 — Anchoring gradient: same scene at 5 data density levels (G0–G4)
  E4 — Cross-model: replicate E1/E2 on GPT-4o / Claude

Reuses grid JPEGs and data structures from the main pipeline experiments.

Usage:
  python blog/anchoring_bias/src/experiment_anchoring.py \
      --experiment e1 \
      --model gemini-2.5-flash

  python blog/anchoring_bias/src/experiment_anchoring.py \
      --experiment e3 --gradient-level g2 \
      --model gemini-2.5-flash
"""

import argparse
import copy
import json
import os
import sys

from google import genai
from google.genai import types


# ── paths ────────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SAMPLE1_GRID = os.path.join(ROOT, "data/output/experiment_vlm/sample1/grid.jpg")
SAMPLE2_GRID = os.path.join(ROOT, "data/output/experiment_vlm/sample2/grid.jpg")
COLLISION_GRID = os.path.join(ROOT, "data/output/experiment_vlm/sample2_scan/chunk_10_20/grid.jpg")

SAMPLE1_MODE_B = os.path.join(ROOT, "data/output/experiment_vlm/sample1/event_log_mode_b.json")
SAMPLE2_MODE_B = os.path.join(ROOT, "data/output/experiment_vlm/sample2/event_log_mode_b.json")

SAMPLE1_SMOOTH_CSV = os.path.join(ROOT, "data/output/experiment_01_fullframe/layer1_6_smooth.csv")
SAMPLE2_SMOOTH_CSV = os.path.join(ROOT, "data/output/experiment_02/layer1_6_smooth.csv")

BLOG_DATA = os.path.join(ROOT, "blog/anchoring_bias/data")


# ── data manipulation functions ──────────────────────────────────────────────

def load_mode_b(path):
    with open(path) as f:
        return json.load(f)


def build_contradictory_data(base_data, overrides):
    """Return a copy of Mode B data with per-track field overrides.

    Args:
        base_data: Mode B JSON dict (has 'tracks' list)
        overrides: dict of track_id -> dict of field overrides.
            Example: {"T2": {"movement": "stationary", "carrying": [], "direction": "none"}}
            Special keys:
              - "movement": added to each sample as a new field
              - "direction": added to each sample as a new field
              - "carrying": replaces carrying list in each sample
              - "note": added as a top-level note on the track
    """
    data = copy.deepcopy(base_data)
    data["mode"] = "raw_numbers_modified"
    data["_experiment"] = "e1_contradictory"

    for track in data["tracks"]:
        tid = track["id"]
        if tid not in overrides:
            continue
        ovr = overrides[tid]

        if "note" in ovr:
            track["note"] = ovr["note"]

        for sample in track.get("samples", []):
            if "carrying" in ovr:
                sample["carrying"] = ovr["carrying"]
            if "movement" in ovr:
                sample["movement"] = ovr["movement"]
            if "direction" in ovr:
                sample["direction"] = ovr["direction"]
            # Override any numeric fields
            for key in ("norm_speed", "norm_area_delta"):
                if key in ovr:
                    sample[key] = ovr[key]

    return data


def build_partial_data(base_data, include_tracks):
    """Return Mode B data filtered to only the specified track IDs.

    Args:
        base_data: Mode B JSON dict
        include_tracks: set of track IDs to keep (e.g. {"T2"})
    """
    data = copy.deepcopy(base_data)
    data["mode"] = "raw_numbers_partial"
    data["_experiment"] = "e2_partial"
    data["tracks"] = [t for t in data["tracks"] if t["id"] in include_tracks]
    return data


def build_gradient_data(smooth_csv_path, fps, start_s, end_s, level):
    """Build tracking data at varying density levels for gradient experiment.

    Levels:
        g0: None (vision only)
        g1: Track IDs + time ranges + carrying items only (minimal)
        g2: 3-4 key state changes per track (summary)
        g3: Full Mode B (every 10 frames) — already exists
        g4: Dense (every 3 frames, ~3x Mode B density)

    Returns JSON-serializable dict, or None for g0.
    """
    import pandas as pd

    if level == "g0":
        return None

    df = pd.read_csv(smooth_csv_path)
    start_f = int(start_s * fps)
    end_f = int(end_s * fps)
    window = df[(df["frame_id"] >= start_f) & (df["frame_id"] <= end_f)].copy()

    tracks = []
    for ntid, grp in window.groupby("normalized_track_id"):
        grp = grp.sort_values("frame_id")
        tid = f"T{int(ntid)}"

        carrying = []
        if grp["has_handbag"].any():
            carrying.append("handbag")
        if grp["has_backpack"].any():
            carrying.append("backpack")
        if "has_suitcase" in grp.columns and grp["has_suitcase"].any():
            carrying.append("suitcase")

        first_t = grp.iloc[0]["timestamp"]
        last_t = grp.iloc[-1]["timestamp"]

        if level == "g1":
            # Minimal: just identity + time range + carrying
            tracks.append({
                "id": tid,
                "class": "person",
                "present": f"{first_t:.1f}s–{last_t:.1f}s",
                "carrying": carrying,
            })

        elif level == "g2":
            # Summary: first/last + 2-3 key state changes
            samples = []
            indices = [0]
            # Add midpoint and quartiles
            n = len(grp)
            for frac in [0.25, 0.5, 0.75]:
                idx = min(int(n * frac), n - 1)
                if idx not in indices:
                    indices.append(idx)
            indices.append(n - 1)
            indices = sorted(set(indices))

            for idx in indices:
                row = grp.iloc[idx]
                entry = {
                    "t": f"{row['timestamp']:.1f}s",
                    "cx": round(float(row["cx"]), 1),
                    "cy": round(float(row["cy"]), 1),
                    "bbox_area": round(float(row["bbox_area"]), 0),
                }
                if carrying:
                    entry["carrying"] = carrying
                samples.append(entry)

            tracks.append({
                "id": tid,
                "class": "person",
                "samples": samples,
            })

        elif level == "g4":
            # Dense: every 3 frames (~3x Mode B)
            sampled = grp.iloc[::3]
            samples = []
            for _, row in sampled.iterrows():
                carry = []
                if row.get("has_handbag"):
                    carry.append("handbag")
                if row.get("has_backpack"):
                    carry.append("backpack")
                entry = {
                    "t": f"{row['timestamp']:.1f}s",
                    "cx": round(float(row["cx"]), 1),
                    "cy": round(float(row["cy"]), 1),
                    "bbox_area": round(float(row["bbox_area"]), 0),
                    "norm_area_delta": round(float(row["norm_area_delta"]), 4) if pd.notna(row.get("norm_area_delta")) else 0,
                    "norm_speed": round(float(row["norm_speed"]), 4) if pd.notna(row.get("norm_speed")) else 0,
                }
                if carry:
                    entry["carrying"] = carry
                samples.append(entry)

            tracks.append({
                "id": tid,
                "class": "person",
                "samples": samples,
            })

    meta = {
        "source": "yolov8l + botsort",
    }
    if level in ("g2", "g4"):
        meta["cx_cy_meaning"] = "centroid pixel coordinates"
    if level == "g4":
        meta["norm_area_delta_meaning"] = "bbox area change per frame / bbox area (positive=approaching camera)"
        meta["norm_speed_meaning"] = "centroid speed / frame diagonal (scale-invariant lateral speed)"

    return {
        "mode": f"gradient_{level}",
        "meta": meta,
        "window": {"start": f"{start_s}s", "end": f"{end_s}s", "duration_s": end_s - start_s},
        "tracks": tracks,
    }


# ── prompts ──────────────────────────────────────────────────────────────────

PROMPT_VISION_ONLY = """\
You are analyzing a surveillance video.
The grid image shows {n} uniformly-sampled frames from a {duration}s window ({start}s–{end}s), \
left-to-right, top-to-bottom, each timestamped.

Describe what is happening in this window. Include timestamps.
Focus on: what people are doing, what objects are present, notable events, any suspicious or significant activity.\
"""

PROMPT_WITH_DATA = """\
You are analyzing a surveillance video.
The grid image shows {n} uniformly-sampled frames from a {duration}s window ({start}s–{end}s), \
left-to-right, top-to-bottom, each timestamped.
You are also given structured tracking data from a computer vision pipeline (YOLOv8 + BoT-SORT tracker).

{data_meta}

The data is ML-generated and may have minor noise. Use the visual frames as ground truth when signals seem ambiguous.

Tracking data:
{event_log}

Describe what is happening in this window. Include timestamps.
Focus on: what people are doing, what objects are present, notable events, any suspicious or significant activity.\
"""

PROMPT_E0_VISION = """\
Describe what you see in this image in detail. \
List every entity, object, person, and notable element you can identify. \
Be thorough — mention everything visible, even if it seems mundane.\
"""

PROMPT_E0_ANCHORED = """\
You are analyzing a scene. You are given structured detection data from a computer vision pipeline \
that has identified and tracked the following entities in the image.

Detection data:
{detection_data}

Using both the image and the detection data, describe what is happening in this scene. \
Be thorough — mention every entity you can identify.\
"""

DATA_META_RAW = """\
The tracking data contains raw normalized signals:
- norm_area_delta: bbox area change per frame / bbox area (positive = approaching camera)
- norm_speed: centroid speed / frame diagonal (scale-invariant lateral speed)
- cx, cy: centroid pixel coordinates"""

DATA_META_MINIMAL = """\
The tracking data contains person identities, time ranges, and items carried."""

DATA_META_SUMMARY = """\
The tracking data contains sampled positions for tracked persons at key moments."""


# ── VLM runner ───────────────────────────────────────────────────────────────

def call_gemini(image_path, prompt, model="gemini-2.5-flash"):
    """Call Gemini with an image and text prompt."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

    client = genai.Client()
    print(f"Calling {model}...")
    response = client.models.generate_content(
        model=model,
        contents=[image_part, prompt],
    )
    return response.text


def run_experiment(grid_path, data_json, prompt, model, output_dir, label):
    """Generic experiment runner. Saves prompt, data, and response."""
    os.makedirs(output_dir, exist_ok=True)

    # Save input data
    if data_json is not None:
        data_path = os.path.join(output_dir, f"{label}_input_data.json")
        with open(data_path, "w") as f:
            json.dump(data_json, f, indent=2)
        print(f"Input data saved → {data_path}")

    # Save prompt
    prompt_path = os.path.join(output_dir, f"{label}_prompt.txt")
    with open(prompt_path, "w") as f:
        f.write(prompt)

    # Call VLM
    response_text = call_gemini(grid_path, prompt, model)

    # Save response
    response_path = os.path.join(output_dir, f"{label}_response.txt")
    with open(response_path, "w") as f:
        f.write(response_text)
    print(f"Response saved → {response_path}")
    print(f"\n{'='*60}\n[{label.upper()}] RESPONSE:\n{'='*60}")
    print(response_text)

    return response_text


# ── experiment runners ───────────────────────────────────────────────────────

def run_e0(model, output_dir):
    """E0: The Elephant in the Room — staged demo."""
    output_dir = os.path.join(output_dir, "e0_elephant")

    image_path = os.path.join(output_dir, "image.jpg")
    if not os.path.exists(image_path):
        print(f"ERROR: Place the staged image at {image_path}")
        print("This experiment requires a user-provided image with an obvious")
        print("prominent element (elephant, gorilla suit, car on fire, etc.)")
        print("in a mundane scene.")
        sys.exit(1)

    detection_data_path = os.path.join(output_dir, "detection_data.json")
    if not os.path.exists(detection_data_path):
        print(f"ERROR: Place the detection data JSON at {detection_data_path}")
        print("This should be meticulous structured data about every mundane element")
        print("in the image, deliberately omitting the obvious prominent thing.")
        sys.exit(1)

    with open(detection_data_path) as f:
        detection_data = json.load(f)

    # Condition A: vision only
    print("\n--- E0 Condition A: Vision Only ---")
    run_experiment(image_path, None, PROMPT_E0_VISION, model, output_dir, "condition_a_vision")

    # Condition B: vision + anchoring data
    print("\n--- E0 Condition B: Vision + Anchoring Data ---")
    prompt_b = PROMPT_E0_ANCHORED.format(detection_data=json.dumps(detection_data, indent=2))
    run_experiment(image_path, detection_data, prompt_b, model, output_dir, "condition_b_anchored")


def _freeze_track_samples(track, anchor_cx, anchor_cy, anchor_area):
    """Replace all sample coordinates with a fixed position (simulates 'standing still')."""
    for s in track["samples"]:
        s["cx"] = anchor_cx
        s["cy"] = anchor_cy
        s["bbox_area"] = anchor_area
        s["norm_speed"] = 0.0
        s["norm_area_delta"] = 0.0
        s.pop("carrying", None)  # remove carrying field entirely


def run_e1(model, output_dir):
    """E1: Contradictory data — internally consistent fake tracking data on both samples.

    v2: All coordinate fields (cx, cy, bbox_area, norm_speed, norm_area_delta) are frozen
    to a plausible 'standing still' position, so the data cannot be detected as inconsistent
    from within the data itself. Only the visual frames reveal the lie.
    """
    output_dir = os.path.join(output_dir, "e1_contradictory")

    # --- Sample 1: shoplifting (85–95s) ---
    # Ground truth: T2 picks up garment, conceals in bag, walks away
    # Planted lie: T2 frozen at her starting position — stationary, no items
    print("\n--- E1/sample1: Contradictory Data v2 (T2 fully frozen) ---")
    base_s1 = load_mode_b(SAMPLE1_MODE_B)
    contra_s1 = copy.deepcopy(base_s1)
    contra_s1["mode"] = "raw_numbers_modified"
    contra_s1["_experiment"] = "e1_contradictory_v2"
    for track in contra_s1["tracks"]:
        if track["id"] == "T2":
            # Anchor at her real starting position (from Mode B data: cx≈307, cy≈253, area≈14000)
            _freeze_track_samples(track, anchor_cx=307.0, anchor_cy=253.0, anchor_area=14000)
            track["note"] = "Person standing still at clothing rack throughout. No items picked up or moved."
    prompt_s1 = PROMPT_WITH_DATA.format(
        n=10, duration=10.0, start=85.0, end=95.0,
        data_meta=DATA_META_RAW,
        event_log=json.dumps(contra_s1, indent=2),
    )
    run_experiment(SAMPLE1_GRID, contra_s1, prompt_s1, model,
                   os.path.join(output_dir, "sample1_v2"), "contradictory_t2_v2")

    # --- Sample 2: street scene (0–10s) ---
    # Ground truth: T2 walks diagonally across the street carrying a backpack
    # Planted lie: T2 frozen at a sidewalk position — stationary, no backpack
    print("\n--- E1/sample2: Contradictory Data v2 (T2 fully frozen) ---")
    base_s2 = load_mode_b(SAMPLE2_MODE_B)
    contra_s2 = copy.deepcopy(base_s2)
    contra_s2["mode"] = "raw_numbers_modified"
    contra_s2["_experiment"] = "e1_contradictory_v2"
    for track in contra_s2["tracks"]:
        if track["id"] == "T2":
            # Anchor at a left-sidewalk position (plausible starting spot)
            _freeze_track_samples(track, anchor_cx=120.0, anchor_cy=300.0, anchor_area=3500)
            track["note"] = "Person standing still on left sidewalk. Not moving, not carrying anything."
    prompt_s2 = PROMPT_WITH_DATA.format(
        n=10, duration=10.0, start=0.0, end=10.0,
        data_meta=DATA_META_RAW,
        event_log=json.dumps(contra_s2, indent=2),
    )
    run_experiment(SAMPLE2_GRID, contra_s2, prompt_s2, model,
                   os.path.join(output_dir, "sample2_v2"), "contradictory_t2_v2")


def run_e2(model, output_dir):
    """E2: Partial data — tracking only a subset of visible entities, on both samples."""
    output_dir = os.path.join(output_dir, "e2_partial")

    # --- Sample 2: street scene (0–10s) ---
    # Only track T2 (backpack pedestrian). Omit cow, vehicles, other people.
    print("\n--- E2/sample2: Partial Data (only T2, omit cow/vehicles) ---")
    base_s2 = load_mode_b(SAMPLE2_MODE_B)
    partial_s2 = build_partial_data(base_s2, include_tracks={"T2"})
    prompt_s2 = PROMPT_WITH_DATA.format(
        n=10, duration=10.0, start=0.0, end=10.0,
        data_meta=DATA_META_RAW,
        event_log=json.dumps(partial_s2, indent=2),
    )
    run_experiment(SAMPLE2_GRID, partial_s2, prompt_s2, model,
                   os.path.join(output_dir, "sample2"), "partial_t2_only")

    # --- Sample 1: shoplifting (85–95s) ---
    # Only track T1 (stationary background person). Omit T2 (the shoplifter) and others.
    # Tests: does the VLM still notice the shoplifting if the data only mentions a bystander?
    print("\n--- E2/sample1: Partial Data (only T1 bystander, omit shoplifter T2) ---")
    base_s1 = load_mode_b(SAMPLE1_MODE_B)
    partial_s1 = build_partial_data(base_s1, include_tracks={"T1"})
    prompt_s1 = PROMPT_WITH_DATA.format(
        n=10, duration=10.0, start=85.0, end=95.0,
        data_meta=DATA_META_RAW,
        event_log=json.dumps(partial_s1, indent=2),
    )
    run_experiment(SAMPLE1_GRID, partial_s1, prompt_s1, model,
                   os.path.join(output_dir, "sample1"), "partial_t1_only")


def _run_e3_on_sample(model, output_dir, sample_name, grid_path, smooth_csv, mode_b_path, fps, start, end, level=None):
    """Run gradient experiment on a single sample."""
    n_frames = round((end - start) * 1.0)  # 1 FPS sampling
    duration = end - start
    levels = [level] if level else ["g0", "g1", "g2", "g3", "g4"]

    for lvl in levels:
        if lvl == "g3":
            data = load_mode_b(mode_b_path)
            data["mode"] = "gradient_g3"
        else:
            data = build_gradient_data(smooth_csv, fps, start, end, lvl)

        if lvl == "g0":
            prompt = PROMPT_VISION_ONLY.format(
                n=n_frames, duration=duration, start=start, end=end,
            )
        else:
            if lvl == "g1":
                meta = DATA_META_MINIMAL
            elif lvl == "g2":
                meta = DATA_META_SUMMARY
            else:
                meta = DATA_META_RAW

            prompt = PROMPT_WITH_DATA.format(
                n=n_frames, duration=duration, start=start, end=end,
                data_meta=meta,
                event_log=json.dumps(data, indent=2),
            )

        lvl_label = {"g0": "none", "g1": "minimal", "g2": "summary", "g3": "full", "g4": "dense"}[lvl]
        sub_dir = os.path.join(output_dir, f"{lvl}_{lvl_label}")

        print(f"\n--- E3/{sample_name}: Gradient Level {lvl.upper()} ---")
        run_experiment(grid_path, data, prompt, model, sub_dir, f"gradient_{lvl}")


def run_e3(model, output_dir, level=None):
    """E3: Anchoring gradient — 5 density levels on both samples."""
    base_dir = os.path.join(output_dir, "e3_gradient")

    # Sample 1: shoplifting (85–95s)
    _run_e3_on_sample(
        model, os.path.join(base_dir, "sample1"),
        "sample1", SAMPLE1_GRID, SAMPLE1_SMOOTH_CSV, SAMPLE1_MODE_B,
        fps=30.0, start=85.0, end=95.0, level=level,
    )

    # Sample 2: street scene (0–10s)
    _run_e3_on_sample(
        model, os.path.join(base_dir, "sample2"),
        "sample2", SAMPLE2_GRID, SAMPLE2_SMOOTH_CSV, SAMPLE2_MODE_B,
        fps=30.0, start=0.0, end=10.0, level=level,
    )


def run_e4(model, output_dir):
    """E4: Cross-model — run E1 + E2 on a different model."""
    print(f"\n--- E4: Cross-model replication on {model} ---")
    print("Running E1 (contradictory) on", model)
    run_e1(model, os.path.join(output_dir, f"e4_cross_model/{model.replace('/', '_')}"))
    print("\nRunning E2 (partial) on", model)
    run_e2(model, os.path.join(output_dir, f"e4_cross_model/{model.replace('/', '_')}"))


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Anchoring bias experiments for VLMs")
    parser.add_argument("--experiment", required=True, choices=["e0", "e1", "e2", "e3", "e4"],
                        help="Which experiment to run")
    parser.add_argument("--model", default="gemini-2.5-flash",
                        help="Model to use (default: gemini-2.5-flash)")
    parser.add_argument("--gradient-level", choices=["g0", "g1", "g2", "g3", "g4"],
                        help="For E3: run a specific gradient level only")
    parser.add_argument("--output", default=BLOG_DATA,
                        help="Output directory (default: blog/anchoring_bias/data/)")
    args = parser.parse_args()

    if args.experiment == "e0":
        run_e0(args.model, args.output)
    elif args.experiment == "e1":
        run_e1(args.model, args.output)
    elif args.experiment == "e2":
        run_e2(args.model, args.output)
    elif args.experiment == "e3":
        run_e3(args.model, args.output, args.gradient_level)
    elif args.experiment == "e4":
        run_e4(args.model, args.output)


if __name__ == "__main__":
    main()
