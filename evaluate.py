from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import GTSRBDataset, load_test_samples, prepare_gtsrb_official_layout
from model import build_model


def get_output_paths(cfg: dict, tag: str) -> tuple[Path, Path, Path]:
    results_dir = Path(cfg["results_dir"])
    predictions_dir = Path(cfg.get("predictions_dir", results_dir / "predictions"))
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(
        cfg.get(f"{tag}_metrics_json", results_dir / f"metrics_{tag}.json")
    )
    confusion_png = Path(
        cfg.get(f"{tag}_confusion_png", results_dir / f"confusion_matrix_{tag}.png")
    )
    predictions_csv = Path(
        cfg.get(f"{tag}_predictions_csv", predictions_dir / f"predictions_{tag}.csv")
    )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_png.parent.mkdir(parents=True, exist_ok=True)
    predictions_csv.parent.mkdir(parents=True, exist_ok=True)

    return metrics_path, confusion_png, predictions_csv


def save_confusion_matrix_png(cm, num_classes: int, out_path: Path, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(range(num_classes)),
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=90)
    ax.set_title(f"GTSRB Confusion Matrix ({tag})")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_predictions_csv(
    rows: list[dict],
    out_path: Path,
) -> None:
    fieldnames = ["image_path", "true_label", "pred_label", "correct", "confidence"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_errors(cm):
    off_diag = cm.copy()
    for i in range(len(off_diag)):
        off_diag[i][i] = 0

    row_errors = off_diag.sum(axis=1)
    top_misclassified_true_class = int(row_errors.argmax())
    top_misclassified_true_class_count = int(row_errors[top_misclassified_true_class])

    top_pair_idx = off_diag.argmax()
    n = off_diag.shape[0]
    true_cls = int(top_pair_idx // n)
    pred_cls = int(top_pair_idx % n)
    top_pair_count = int(off_diag[true_cls, pred_cls])

    return {
        "top_misclassified_true_class": top_misclassified_true_class,
        "top_misclassified_true_class_count": top_misclassified_true_class_count,
        "top_confusion_pair": {
            "true_label": true_cls,
            "pred_label": pred_cls,
            "count": top_pair_count,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tag", type=str, default="baseline")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    metrics_path, confusion_png, predictions_csv = get_output_paths(cfg, args.tag)

    paths = prepare_gtsrb_official_layout(
        cfg["train_zip"],
        cfg["test_zip"],
        cfg["test_gt_zip"],
        cfg["data_dir"],
    )
    test_samples = load_test_samples(paths["test_dir"], paths["test_csv"])

    tfms = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = GTSRBDataset(test_samples, transform=tfms)
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model_name"], cfg["num_classes"]).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_true, y_pred = [], []
    prediction_rows = []

    sample_offset = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

            preds_list = preds.cpu().numpy().tolist()
            confs_list = confs.cpu().numpy().tolist()
            labels_list = labels.cpu().numpy().tolist()

            y_pred.extend(preds_list)
            y_true.extend(labels_list)

            batch_size = len(labels_list)
            batch_samples = dataset.samples[sample_offset: sample_offset + batch_size]

            for (img_path, true_label), pred_label, conf in zip(
                batch_samples, preds_list, confs_list
            ):
                prediction_rows.append({
                    "image_path": img_path,
                    "true_label": int(true_label),
                    "pred_label": int(pred_label),
                    "correct": int(true_label == pred_label),
                    "confidence": float(conf),
                })

            sample_offset += batch_size

    cm = confusion_matrix(y_true, y_pred)
    error_summary = summarize_errors(cm)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": cm.tolist(),
        "tag": args.tag,
        "checkpoint": args.checkpoint,
        "num_test_samples": len(y_true),
        "outputs": {
            "metrics_json": str(metrics_path),
            "confusion_png": str(confusion_png),
            "predictions_csv": str(predictions_csv),
        },
        "error_summary": error_summary,
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix_png(cm, cfg["num_classes"], confusion_png, args.tag)
    save_predictions_csv(prediction_rows, predictions_csv)

    print(json.dumps({
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "metrics_json": str(metrics_path),
        "confusion_png": str(confusion_png),
        "predictions_csv": str(predictions_csv),
        "top_misclassified_true_class": error_summary["top_misclassified_true_class"],
        "top_misclassified_true_class_count": error_summary["top_misclassified_true_class_count"],
        "top_confusion_pair": error_summary["top_confusion_pair"],
    }, indent=2))


if __name__ == "__main__":
    main()
