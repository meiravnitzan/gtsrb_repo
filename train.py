from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import (
    GTSRBDataset,
    load_train_samples,
    prepare_gtsrb_official_layout,
    split_train_val,
)
from model import build_model


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def read_manifest_samples(manifest_csv: str):
    manifest_path = Path(manifest_csv)
    if not manifest_path.exists():
        print(f"Augmentation manifest not found: {manifest_path}")
        return []

    samples = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_path", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest CSV is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            image_path = row["image_path"]
            label = int(row["label"])
            if Path(image_path).exists():
                samples.append((image_path, label))
            else:
                print(f"Warning: skipping missing synthetic image: {image_path}")

    return samples


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    return total_loss / len(loader.dataset), f1_score(y_true, y_pred, average="macro")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    return total_loss / len(loader.dataset), f1_score(y_true, y_pred, average="macro")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--mode",
        choices=["baseline", "augmented", "gen22"],
        default="baseline",
        help="Training mode. 'gen22' is kept as a backward-compatible alias for 'augmented'.",
    )
    args = parser.parse_args()

    effective_mode = "augmented" if args.mode == "gen22" else args.mode

    cfg = yaml.safe_load(open(args.config, "r"))
    set_seed(cfg["seed"])

    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["models_dir"]).mkdir(parents=True, exist_ok=True)

    paths = prepare_gtsrb_official_layout(
        cfg["train_zip"],
        cfg["test_zip"],
        cfg["test_gt_zip"],
        cfg["data_dir"],
    )

    train_samples = load_train_samples(paths["train_dir"])
    train_samples, val_samples = split_train_val(
        train_samples,
        val_ratio=cfg.get("val_ratio", 0.2),
        seed=cfg["seed"],
    )

    if effective_mode == "augmented":
        synth_samples = read_manifest_samples(cfg["augmented_manifest"])
        print(f"Loaded synthetic samples from manifest: {len(synth_samples)}")
        train_samples = train_samples + synth_samples

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    train_tfms = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.RandomApply(
            [transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)],
            p=0.5,
        ),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    g = torch.Generator()
    g.manual_seed(cfg["seed"])

    train_loader = DataLoader(
        GTSRBDataset(train_samples, transform=train_tfms),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        GTSRBDataset(val_samples, transform=eval_tfms),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        worker_init_fn=seed_worker,
        generator=g,
    )

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model_name"], cfg["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg.get("label_smoothing", 0.0)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    best_f1 = -1.0
    history = []

    checkpoint_name = (
        "best_baseline.pt" if effective_mode == "baseline" else "best_augmented.pt"
    )
    history_name = (
        "history_baseline.json"
        if effective_mode == "baseline"
        else "history_augmented.json"
    )

    checkpoint_path = Path(cfg["models_dir"]) / checkpoint_name
    history_path = Path(cfg["results_dir"]) / history_name

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "mode": effective_mode,
        }
        history.append(record)
        print(record)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": val_f1,
                    "config": cfg,
                    "mode": effective_mode,
                },
                checkpoint_path,
            )
            print(f"Best checkpoint saved to: {checkpoint_path}")

    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
