# Colab demo: baseline vs augmented checkpoint on one image

import os
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# --- Paths ---
REPO = Path("/content/gtsrb_repo")
MODELS = REPO / "models"
CONFIG = REPO / "config.yaml"

BASELINE_CKPT = MODELS / "best_baseline.pt"
AUGMENTED_CKPT = MODELS / "best_augmented.pt"

assert BASELINE_CKPT.exists(), f"Missing {BASELINE_CKPT}"
assert AUGMENTED_CKPT.exists(), f"Missing {AUGMENTED_CKPT}"

# --- Import project model builder ---
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from model import build_model  # repo code
import yaml

cfg = yaml.safe_load(open(CONFIG, "r"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Labels ---
CLASS_NAMES = {
    0: "speed_limit_20", 1: "speed_limit_30", 2: "speed_limit_50", 3: "speed_limit_60",
    4: "speed_limit_70", 5: "speed_limit_80", 6: "end_speed_limit_80", 7: "speed_limit_100",
    8: "speed_limit_120", 9: "no_passing", 10: "no_passing_trucks", 11: "priority_at_next_intersection",
    12: "priority_road", 13: "yield", 14: "stop", 15: "no_vehicles",
    16: "trucks_prohibited", 17: "no_entry", 18: "general_caution", 19: "dangerous_curve_left",
    20: "dangerous_curve_right", 21: "double_curve", 22: "bumpy_road", 23: "slippery_road",
    24: "road_narrows_right", 25: "road_work", 26: "traffic_signals", 27: "pedestrians",
    28: "children_crossing", 29: "bicycles_crossing", 30: "beware_ice_snow", 31: "wild_animals_crossing",
    32: "end_speed_passing_limits", 33: "turn_right_ahead", 34: "turn_left_ahead", 35: "ahead_only",
    36: "go_straight_or_right", 37: "go_straight_or_left", 38: "keep_right", 39: "keep_left",
    40: "roundabout_mandatory", 41: "end_no_passing", 42: "end_no_passing_trucks"
}

# --- Preprocess ---
img_size = int(cfg.get("image_size", cfg.get("imagesize", 128)))
tfm = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_model(ckpt_path):
    model = build_model(cfg["modelname"], cfg["numclasses"])
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model

baseline_model = load_model(BASELINE_CKPT)
augmented_model = load_model(AUGMENTED_CKPT)

@torch.no_grad()
def predict(model, image_path):
    image = Image.open(image_path).convert("RGB")
    x = tfm(image).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    top3 = np.argsort(probs)[::-1][:3]
    return image, pred, conf, [(int(i), float(probs[i])) for i in top3]

def render_demo(image_path, true_label=None):
    image, b_pred, b_conf, b_top3 = predict(baseline_model, image_path)
    _, a_pred, a_conf, a_top3 = predict(augmented_model, image_path)

    fixed = (true_label == 22 and b_pred != 22 and a_pred == 22) if true_label is not None else False

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image)
    ax0.axis("off")
    title = f"Input image\n{Path(image_path).name}"
    if true_label is not None:
        title += f"\nTrue label: {true_label} ({CLASS_NAMES.get(true_label, 'unknown')})"
    ax0.set_title(title, fontsize=12)

    def panel(ax, model_name, pred, conf, top3, color):
        ax.axis("off")
        lines = [
            model_name,
            "",
            f"Top-1: {pred} ({CLASS_NAMES.get(pred, 'unknown')})",
            f"Confidence: {conf:.3f}",
            "",
            "Top-3:"
        ]
        for k, (cls, p) in enumerate(top3, start=1):
            lines.append(f"{k}. {cls} - {CLASS_NAMES.get(cls, 'unknown')} ({p:.3f})")
        ax.text(
            0.02, 0.98, "\n".join(lines),
            va="top", ha="left", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.6", facecolor=color, alpha=0.15, edgecolor=color, linewidth=2)
        )

    ax1 = fig.add_subplot(gs[0, 1])
    panel(ax1, "Baseline model", b_pred, b_conf, b_top3, "#d62728")

    ax2 = fig.add_subplot(gs[0, 2])
    panel(ax2, "Augmented model", a_pred, a_conf, a_top3, "#2ca02c")

    if fixed:
        fig.suptitle("Class 22 recovered by synthetic augmentation", fontsize=18, color="#2ca02c", weight="bold")
    elif true_label == 22:
        fig.suptitle("Class 22 comparison", fontsize=18)
    else:
        fig.suptitle("Baseline vs augmented comparison", fontsize=18)

    plt.tight_layout()
    plt.show()

# --- Optional helper: find candidate class-22 images where the models disagree ---
def scan_class22_examples(limit=30):
    class22_dir = REPO / "data" / "GTSRB" / "Final_Training" / "Images" / "00022"
    files = sorted([p for p in class22_dir.iterdir() if p.suffix.lower() in [".ppm", ".png", ".jpg", ".jpeg"]])[:limit]
    rows = []
    for p in files:
        _, b_pred, b_conf, _ = predict(baseline_model, p)
        _, a_pred, a_conf, _ = predict(augmented_model, p)
        rows.append({
            "file": p.name,
            "baseline_pred": b_pred,
            "baseline_conf": round(b_conf, 3),
            "augmented_pred": a_pred,
            "augmented_conf": round(a_conf, 3),
            "fixed_22": (b_pred != 22 and a_pred == 22),
        })
    return rows

print("Demo loaded.")
print("Use render_demo('/path/to/image.ppm', true_label=22)")
print("Or scan candidates with: scan_class22_examples(limit=50)")
