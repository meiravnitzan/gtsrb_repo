from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import os

ROOT = Path('/content/gtsrb_project')
RAW = ROOT / 'data' / 'raw'
EXTRACT = ROOT / 'data'
RAW.mkdir(parents=True, exist_ok=True)
EXTRACT.mkdir(parents=True, exist_ok=True)

FILES = [
    {
        "url": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip",
        "path": "data/GTSRB_Final_Training_Images.zip",
    },
    {
        "url": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip",
        "path": "data/GTSRB_Final_Test_Images.zip",
    },
    {
        "url": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip",
        "path": "data/GTSRB_Final_Test_GT.zip",
    },
]

def download(url: str, dest: Path):
    if dest.exists():
        print(f'Skipping download, exists: {dest}')
        return
    print(f'Downloading {url} -> {dest}')
    urlretrieve(url, dest)


def unzip(src: Path, dest: Path):
    print(f'Extracting {src} -> {dest}')
    with zipfile.ZipFile(src, 'r') as zf:
        zf.extractall(dest)


def main():
    for item in FILES.values():
        download(item['url'], item['path'])

    unzip(FILES['train_zip']['path'], EXTRACT)
    unzip(FILES['test_zip']['path'], EXTRACT)
    unzip(FILES['test_gt_zip']['path'], EXTRACT)

    print('\nDone. Files available at:')
    for key, item in FILES.items():
        print(f'- {key}: {item["path"]}')

    print('\nSuggested config.yaml paths:')
    print(f'train_zip: {FILES["train_zip"]["path"]}')
    print(f'test_zip: {FILES["test_zip"]["path"]}')
    print(f'test_gt_zip: {FILES["test_gt_zip"]["path"]}')
    print(f'data_dir: {EXTRACT}')


if __name__ == '__main__':
    main()