from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import GTSRBDataset, load_test_samples, prepare_gtsrb_official_layout
from model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tag", type=str, default="baseline")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    paths = prepare_gtsrb_official_layout(cfg["train_zip"], cfg["test_zip"], cfg["test_gt_zip"], cfg["data_dir"])
    test_samples = load_test_samples(paths["test_dir"], paths["test_csv"])

    tfms = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(GTSRBDataset(test_samples, transform=tfms), batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model_name"], cfg["num_classes"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.numpy().tolist())

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "tag": args.tag,
    }
    with (Path(cfg["results_dir"]) / f"metrics_{args.tag}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
