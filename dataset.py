from __future__ import annotations

import csv
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset


def prepare_gtsrb_official_layout(train_zip: str, test_zip: str, test_gt_zip: str, data_dir: str) -> Dict[str, str]:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    train_dir = root / "GTSRB" / "Final_Training" / "Images"
    test_dir = root / "GTSRB" / "Final_Test" / "Images"
    gt_csv = root / "GT-final_test.csv"

    if not train_dir.exists():
        with zipfile.ZipFile(train_zip, "r") as zf:
            zf.extractall(root)
    if not test_dir.exists():
        with zipfile.ZipFile(test_zip, "r") as zf:
            zf.extractall(root)
    if not gt_csv.exists():
        with zipfile.ZipFile(test_gt_zip, "r") as zf:
            zf.extractall(root)

    return {"train_dir": str(train_dir), "test_dir": str(test_dir), "test_csv": str(gt_csv)}


def load_train_samples(train_dir: str) -> List[Tuple[str, int]]:
    train_root = Path(train_dir)
    samples: List[Tuple[str, int]] = []
    for class_dir in sorted(train_root.iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            class_id = int(class_dir.name)
        except ValueError:
            continue
        for img_path in sorted(class_dir.glob("*.ppm")):
            samples.append((str(img_path), class_id))
    return samples


def load_test_samples(test_dir: str, test_csv: str) -> List[Tuple[str, int]]:
    test_root = Path(test_dir)
    samples: List[Tuple[str, int]] = []
    with Path(test_csv).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            samples.append((str(test_root / row["Filename"]), int(row["ClassId"])))
    return samples


def split_train_val(samples: Sequence[Tuple[str, int]], val_ratio: float = 0.2, seed: int = 42):
    by_class: Dict[int, List[Tuple[str, int]]] = {}
    for path, label in samples:
        by_class.setdefault(label, []).append((path, label))

    rng = random.Random(seed)
    train_split: List[Tuple[str, int]] = []
    val_split: List[Tuple[str, int]] = []

    for _, items in by_class.items():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val_split.extend(items[:n_val])
        train_split.extend(items[n_val:])
    return train_split, val_split


class GTSRBDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[str, int]], transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
