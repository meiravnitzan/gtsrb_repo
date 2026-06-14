from pathlib import Path
from urllib.request import urlretrieve
import zipfile

DATA_DIR = Path("data")

FILES = [
    {
        "url": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip",
        "path": DATA_DIR / "GTSRB_Final_Training_Images.zip",
    },
    {
        "url": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip",
        "path": DATA_DIR / "GTSRB_Final_Test_Images.zip",
    },
    {
        "url": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip",
        "path": DATA_DIR / "GTSRB_Final_Test_GT.zip",
    },
]


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] already downloaded: {dest}")
        return
    print(f"[download] {url} -> {dest}")
    urlretrieve(url, dest)


def extract(zip_path: Path, out_dir: Path) -> None:
    print(f"[extract] {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for item in FILES:
        download(item["url"], item["path"])

    extract(DATA_DIR / "GTSRB_Final_Training_Images.zip", DATA_DIR)
    extract(DATA_DIR / "GTSRB_Final_Test_Images.zip", DATA_DIR)
    extract(DATA_DIR / "GTSRB_Final_Test_GT.zip", DATA_DIR)

    print("\nDone.")
    print(f"Training images: {DATA_DIR / 'GTSRB' / 'Final_Training' / 'Images'}")
    print(f"Test images:     {DATA_DIR / 'GTSRB' / 'Final_Test' / 'Images'}")
    print(f"Test CSV:        {DATA_DIR / 'GT-final_test.csv'}")


if __name__ == "__main__":
    main()
