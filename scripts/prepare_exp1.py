"""EXP1 — Preparation script.

Generates all inputs for the anchoring bias modality experiment:
  EXP1-0: baseline_vision_only       — clean images, no text data
  EXP1-B: text_bbox_real             — clean images + text data with bbox coords
  EXP1-C: text_bbox_fake             — clean images + text data with real + injected bboxes
  EXP1-D: visual_bbox_real           — center image with labeled bbox overlays, base text (no bbox)
  EXP1-E: visual_bbox_fake           — center image with real + injected labeled overlays, base text
  EXP1-F: crossmodal_bbox_real       — center image with ID overlays, text has ID:label map
  EXP1-G: crossmodal_bbox_fake       — center image with real + injected ID overlays, text has map

Does NOT call Gemini. All outputs go to blog/anchoring_bias/data/exp1/ for review.
Existing prompt files are NEVER overwritten — edit them freely before running.

Output file naming:
  exp1_shared_grid.jpg               — shared 3x3 temporal grid (all conditions)
  exp1_0_center.jpg                  — clean center frame (0, B, C)
  exp1_d_center.jpg                  — center with real labeled bboxes (D)
  exp1_e_center.jpg                  — center with real + injected labeled bboxes (E)
  exp1_f_center.jpg                  — center with real ID-only bboxes (F)
  exp1_g_center.jpg                  — center with real + injected ID-only bboxes (G)
  exp1_{letter}_text_data.json       — text data per condition (null for 0)
  exp1_{letter}_prompt.txt           — prompt per condition (skipped if already exists)
  exp1_token_counts.json             — input token estimates
  exp1_ref_detections_real.json      — internal reference only, not sent to VLM
  exp1_ref_detections_injected.json  — internal reference only, not sent to VLM

Usage:
    python blog/anchoring_bias/src/prepare_exp1.py
"""

import json
import math
import os

import cv2
import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
VIDEO_PATH = os.path.join(ROOT, "data/sample.mp4")
SMOOTH_CSV = os.path.join(ROOT, "data/output/experiment_01_fullframe/layer1_6_smooth.csv")
TAGS_CSV   = os.path.join(ROOT, "data/output/experiment_01_fullframe/layer1_tags.csv")
OUT_DIR    = os.path.join(ROOT, "blog/anchoring_bias/data/exp1")

# ── experiment window ─────────────────────────────────────────────────────────
START_S    = 85.0
END_S      = 95.0
N_FRAMES   = 9
GRID_COLS  = 3
GRID_ROWS  = 3
CENTER_IDX = 4        # frame 5 of 9 = ~90s

# Condition letter → name mapping (used for file prefixes)
CONDITIONS = {
    "0": "baseline_vision_only",
    "b": "text_bbox_real",
    "c": "text_bbox_fake",
    "d": "visual_bbox_real",
    "e": "visual_bbox_fake",
    "f": "crossmodal_bbox_real",
    "g": "crossmodal_bbox_fake",
}

# ── colour palette (used only for image overlays, never sent to VLM) ─────────
COLORS = {
    "person":     (0,   200,   0),
    "handbag":    (200, 100,   0),
    "backpack":   (200,   0, 200),
    "suitcase":   (0,   200, 200),
    "cell phone": (0,    80, 255),
    "_injected":  (0,    0,  220),   # red — injected entries only, no VLM exposure
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS  = 2

# ── injected (fake) detections ────────────────────────────────────────────────
# IDs look like real BoT-SORT track IDs — no special prefix to avoid leading VLM.
# _injected=True is an internal flag; it is NEVER written to VLM-facing outputs.
INJECTED_DETECTIONS = [
    {
        "id":        "T89",           # looks like a real tracker ID
        "class":     "person",
        "conf":      0.52,
        "present":   f"{START_S}s–{END_S}s",
        "carrying":  [],
        "bbox":      [555, 100, 620, 300],   # empty right side of aisle
        "_injected": True,
    },
    {
        "id":        "O412",          # looks like a real object track ID
        "class":     "handbag",
        "conf":      0.47,
        "present":   "90.0s",
        "carrying":  [],
        "bbox":      [150, 400, 210, 460],   # floor near rack — nothing there
        "_injected": True,
    },
    {
        "id":        "O87",
        "class":     "cell phone",
        "conf":      0.44,
        "present":   "90.0s–93.0s",
        "carrying":  [],
        "bbox":      [355, 300, 390, 330],   # near T2 hand area, offset — no phone
        "_injected": True,
    },
]


# ── helpers ───────────────────────────────────────────────────────────────────

def safe_write(path, content, mode="w"):
    """Write file. If it's a prompt file and already exists, skip and warn."""
    if os.path.basename(path).endswith("_prompt.txt") and os.path.exists(path):
        print(f"  [SKIP] {os.path.basename(path)} already exists — keeping your edits")
        return
    with open(path, mode) as f:
        if isinstance(content, str):
            f.write(content)
        else:
            json.dump(content, f, indent=2)
    print(f"  Saved: {os.path.basename(path)}")


def extract_frames(video_path, start_s, end_s, n):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_f = int(start_s * fps)
    end_f   = int(end_s   * fps)
    frame_ids = [int(start_f + i * (end_f - start_f) / (n - 1)) for i in range(n)]
    frames, timestamps = [], []
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if ret:
            timestamps.append(round(fid / fps, 1))
            frames.append(frame)
    cap.release()
    return frames, timestamps


def stamp_frame(frame, ts):
    f = frame.copy()
    cv2.putText(f, f"{ts:.1f}s", (10, 30), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return f


def build_grid(frames, timestamps, cols, rows):
    h, w   = frames[0].shape[:2]
    thumb_w, thumb_h = w // cols, h // rows
    stamped = [stamp_frame(f, ts) for f, ts in zip(frames, timestamps)]
    grid_rows = []
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            idx = r * cols + c
            f = stamped[idx] if idx < len(stamped) else np.zeros((h, w, 3), dtype=np.uint8)
            row_imgs.append(cv2.resize(f, (thumb_w, thumb_h)))
        grid_rows.append(np.hstack(row_imgs))
    return np.vstack(grid_rows)


def draw_bbox(img, x1, y1, x2, y2, label, color):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, THICKNESS)
    (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
    lx1, ly1 = x1, max(0, y1 - th - baseline - 4)
    cv2.rectangle(img, (lx1, ly1), (lx1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (lx1 + 2, y1 - baseline - 2), FONT, FONT_SCALE,
                (255, 255, 255), 1, cv2.LINE_AA)


def overlay_detections(frame, detections, mode):
    """Draw bboxes on frame.
    mode='label' → "class conf"  (EXP1-D/E)
    mode='id'    → detection ID  (EXP1-F/G)
    """
    img = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = COLORS.get("_injected" if d.get("_injected") else d["class"], (180, 180, 180))
        label = f"{d['class']} {d['conf']:.2f}" if mode == "label" else d["id"]
        draw_bbox(img, x1, y1, x2, y2, label, color)
    return img


def gemini_image_tokens(w, h):
    return math.ceil(w / 768) * math.ceil(h / 768) * 258


def estimate_text_tokens(text_data):
    if text_data is None:
        return 0
    return len(json.dumps(text_data)) // 4


# ── detection data builders ───────────────────────────────────────────────────

def load_real_detections(smooth_csv, tags_csv, center_ts):
    smooth = pd.read_csv(smooth_csv)
    tags   = pd.read_csv(tags_csv)

    person_window = smooth[(smooth.timestamp >= START_S) & (smooth.timestamp <= END_S)]
    detections = []
    for ntid, grp in person_window.groupby("normalized_track_id"):
        nearest = grp.iloc[(grp.timestamp - center_ts).abs().argsort()[:1]]
        row = nearest.iloc[0]
        carrying = []
        if grp.has_handbag.any():  carrying.append("handbag")
        if grp.has_backpack.any(): carrying.append("backpack")
        if "has_suitcase" in grp.columns and grp.has_suitcase.any():
            carrying.append("suitcase")
        detections.append({
            "id":        f"T{int(ntid)}",
            "class":     "person",
            "conf":      round(float(row.confidence), 2),
            "present":   f"{grp.timestamp.min():.1f}s–{grp.timestamp.max():.1f}s",
            "carrying":  carrying,
            "bbox":      [int(row.x1), int(row.y1), int(row.x2), int(row.y2)],
            "_injected": False,
        })

    obj_window = tags[
        (tags.timestamp >= center_ts - 0.5) &
        (tags.timestamp <= center_ts + 0.5) &
        (tags["class"].isin(["handbag", "backpack", "suitcase"]))
    ]
    seen = set()
    for _, row in obj_window.sort_values("confidence", ascending=False).iterrows():
        key = (row["class"], int(row.track_id))
        if key in seen:
            continue
        seen.add(key)
        detections.append({
            "id":        f"O{int(row.track_id)}",
            "class":     row["class"],
            "conf":      round(float(row.confidence), 2),
            "present":   f"{center_ts}s",
            "carrying":  [],
            "bbox":      [int(row.x1), int(row.y1), int(row.x2), int(row.y2)],
            "_injected": False,
        })
    return detections


def build_text_data(detections, include_bbox, include_id_map, include_injected):
    """Build VLM-facing text data. Internal fields (_injected) are always stripped."""
    entries = list(detections) + (INJECTED_DETECTIONS if include_injected else [])
    tracks = []
    for d in entries:
        entry = {"id": d["id"], "class": d["class"], "present": d["present"]}
        if d.get("carrying"):
            entry["carrying"] = d["carrying"]
        if include_bbox:
            entry["bbox"] = d["bbox"]
        # _injected is NEVER included
        tracks.append(entry)

    data = {
        "source": "yolov8l + botsort",
        "note": "ML-generated. May have minor noise. Use visual frames as ground truth when ambiguous.",
        "window": {"start": f"{START_S}s", "end": f"{END_S}s"},
        "detections": tracks,
    }
    if include_id_map:
        data["id_labels"] = {d["id"]: d["class"] for d in entries}
    return data


# ── default prompts (used only if prompt file does not already exist) ─────────

PROMPT_BASE = f"""\
You are analyzing a surveillance video clip.

You are given two images:
- Image 1: a 3x3 temporal grid of {N_FRAMES} uniformly-sampled frames from an \
{START_S}s–{END_S}s window, left-to-right top-to-bottom, each timestamped.
- Image 2: the center frame (~90.0s) at full resolution.

Describe what is happening in this window. Include timestamps.
Focus on: what each person is doing, what objects are present, \
notable events, any suspicious or significant activity.\
"""

PROMPT_WITH_DATA = PROMPT_BASE + """

You are also given structured detection data from a computer vision pipeline (YOLOv8 + BoT-SORT):

{data}

Use the visual frames as ground truth when data signals seem ambiguous.\
"""

def default_prompt(text_data):
    if text_data is None:
        return PROMPT_BASE
    return PROMPT_WITH_DATA.format(data=json.dumps(text_data, indent=2))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output: {OUT_DIR}\n")

    # 1. Extract frames
    print("[1/6] Extracting frames...")
    frames, timestamps = extract_frames(VIDEO_PATH, START_S, END_S, N_FRAMES)
    center_ts  = timestamps[CENTER_IDX]
    center_raw = frames[CENTER_IDX]
    h, w = center_raw.shape[:2]
    print(f"  {N_FRAMES} frames: {timestamps}")
    print(f"  Center: {center_ts}s @ {w}x{h}")

    # 2. Grid
    print("\n[2/6] Building 3x3 grid...")
    grid = build_grid(frames, timestamps, GRID_COLS, GRID_ROWS)
    grid_path = os.path.join(OUT_DIR, "exp1_shared_grid.jpg")
    cv2.imwrite(grid_path, grid)
    print(f"  Saved: exp1_shared_grid.jpg  ({grid.shape[1]}x{grid.shape[0]})")

    # 3. Clean center frame
    print("\n[3/6] Saving clean center frame...")
    center_clean = stamp_frame(center_raw, center_ts)
    cv2.imwrite(os.path.join(OUT_DIR, "exp1_0_center.jpg"), center_clean)
    print(f"  Saved: exp1_0_center.jpg")

    # 4. Load real detections
    print("\n[4/6] Loading real detections...")
    real_dets = load_real_detections(SMOOTH_CSV, TAGS_CSV, center_ts)
    print(f"  {len(real_dets)} real detections:")
    for d in real_dets:
        print(f"    {d['id']:6s}  {d['class']:12s}  conf={d['conf']:.2f}  bbox={d['bbox']}  carrying={d['carrying']}")
    print(f"  {len(INJECTED_DETECTIONS)} injected detections:")
    for d in INJECTED_DETECTIONS:
        print(f"    {d['id']:6s}  {d['class']:12s}  conf={d['conf']:.2f}  bbox={d['bbox']}")

    # 5. Build center frame overlay variants
    print("\n[5/6] Building overlay variants...")

    variants = {
        "d": (real_dets,                           "label"),
        "e": (real_dets + INJECTED_DETECTIONS,     "label"),
        "f": (real_dets,                           "id"),
        "g": (real_dets + INJECTED_DETECTIONS,     "id"),
    }
    for letter, (dets, mode) in variants.items():
        img = overlay_detections(center_clean, dets, mode)
        fname = f"exp1_{letter}_center.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, fname), img)
        print(f"  Saved: {fname}")

    # 6. Text data + prompts per condition
    print("\n[6/6] Building text data and prompts...")

    condition_specs = {
        "0": dict(include_bbox=False, include_id_map=False, include_injected=False, null=True),
        "b": dict(include_bbox=True,  include_id_map=False, include_injected=False, null=False),
        "c": dict(include_bbox=True,  include_id_map=False, include_injected=True,  null=False),
        "d": dict(include_bbox=False, include_id_map=False, include_injected=False, null=False),
        "e": dict(include_bbox=False, include_id_map=False, include_injected=False, null=False),
        "f": dict(include_bbox=False, include_id_map=True,  include_injected=False, null=False),
        "g": dict(include_bbox=False, include_id_map=True,  include_injected=True,  null=False),
    }

    token_counts = {}
    grid_h, grid_w = grid.shape[:2]

    for letter, spec in condition_specs.items():
        name = CONDITIONS[letter]
        text_data = None if spec["null"] else build_text_data(
            real_dets,
            include_bbox=spec["include_bbox"],
            include_id_map=spec["include_id_map"],
            include_injected=spec["include_injected"],
        )

        # Text data JSON
        td_path = os.path.join(OUT_DIR, f"exp1_{letter}_text_data.json")
        safe_write(td_path, text_data)

        # Prompt (skipped if already exists)
        prompt_path = os.path.join(OUT_DIR, f"exp1_{letter}_prompt.txt")
        # Also check for old naming convention
        old_prompt_path = os.path.join(OUT_DIR, f"prompt_{name}.txt")
        if os.path.exists(old_prompt_path) and not os.path.exists(prompt_path):
            # Migrate user's edited prompt to new naming
            with open(old_prompt_path) as f:
                content = f.read()
            with open(prompt_path, "w") as f:
                f.write(content)
            print(f"  Migrated: prompt_{name}.txt → exp1_{letter}_prompt.txt (your edits preserved)")
        else:
            safe_write(prompt_path, default_prompt(text_data))

        # Token estimates
        grid_toks   = gemini_image_tokens(grid_w, grid_h)
        center_toks = gemini_image_tokens(w, h)
        text_toks   = estimate_text_tokens(text_data) + len(PROMPT_BASE) // 4
        total       = grid_toks + center_toks + text_toks
        token_counts[f"exp1_{letter}_{name}"] = {
            "grid_image_tokens":   grid_toks,
            "center_image_tokens": center_toks,
            "text_tokens":         text_toks,
            "total_input_tokens":  total,
        }
        print(f"  [exp1_{letter}] text={text_toks:5d} tok  images={grid_toks+center_toks} tok  total={total} tok")

    # Save token counts
    safe_write(os.path.join(OUT_DIR, "exp1_token_counts.json"), token_counts)

    # Internal reference files (never sent to VLM)
    safe_write(os.path.join(OUT_DIR, "exp1_ref_detections_real.json"), real_dets)
    safe_write(os.path.join(OUT_DIR, "exp1_ref_detections_injected.json"), INJECTED_DETECTIONS)

    print("\n✓ Done. Review before running:")
    print("  exp1_shared_grid.jpg          — 3x3 temporal grid (all conditions)")
    print("  exp1_0_center.jpg             — clean center frame (0, B, C)")
    print("  exp1_d_center.jpg             — D: real bboxes with labels")
    print("  exp1_e_center.jpg             — E: real + injected bboxes with labels")
    print("  exp1_f_center.jpg             — F: real bboxes with IDs")
    print("  exp1_g_center.jpg             — G: real + injected bboxes with IDs")
    print("  exp1_{letter}_prompt.txt      — prompt per condition (YOUR EDITS PRESERVED)")
    print("  exp1_{letter}_text_data.json  — text data per condition")
    print("  exp1_token_counts.json        — input token estimates")


if __name__ == "__main__":
    main()
