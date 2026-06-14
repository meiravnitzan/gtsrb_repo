from __future__ import annotations

import argparse

import torch
import yaml
from PIL import Image
from torchvision import transforms

from model import build_model


CLASS_NAMES = {22: "Bumpy road"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model_name"], cfg["num_classes"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(args.image).convert("RGB")
    x = tfms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        pred = int(torch.argmax(probs).item())
        conf = float(probs[pred].item())
    print({"predicted_class": pred, "label": CLASS_NAMES.get(pred, str(pred)), "confidence": conf})


if __name__ == "__main__":
    main()
