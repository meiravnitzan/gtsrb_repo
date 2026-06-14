from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import os

ROOT = Path('/content/gtsrb_project')
RAW = ROOT / 'data' / 'raw'
EXTRACT = ROOT / 'data'
RAW.mkdir(parents=True, exist_ok=True)
EXTRACT.mkdir(parents=True, exist_ok=True)

FILES = {
    'train_zip': {
        'url': 'https://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip',
        'path': RAW / 'GTSRB_Final_Training_Images.zip'
    },
    'test_zip': {
        'url': 'https://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip',
        'path': RAW / 'GTSRB_Final_Test_Images.zip'
    },
    'test_gt_zip': {
        'url': 'https://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip',
        'path': RAW / 'GTSRB_Final_Test_GT.zip'
    }
}


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