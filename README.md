# GTSRB Traffic Sign Classification Project

This project trains and evaluates a traffic sign classifier on the German Traffic Sign Recognition Benchmark (GTSRB). The repository includes scripts for downloading the dataset, training a baseline model, evaluating the trained checkpoint, running inference, analyzing failure cases, and launching a lightweight demo workflow. The attached full Colab notebook documents an end-to-end run on a Tesla T4 GPU and shows the exact command sequence used for the baseline experiment [file:922].

## Project goal

The goal is to classify German road signs into the benchmark label set used by GTSRB. The full run in the notebook focuses on a baseline training/evaluation pipeline and then inspects the weakest classes through targeted error analysis images [file:922].

## Repository structure

The full notebook lists the main repository files after cloning the public repo, including `train.py`, `evaluate.py`, `inference.py`, `demo.py`, `download_data.py`, `dataset.py`, `model.py`, `config.yaml`, `requirements.txt`, `requirements-gen.txt`, the `results/` and `models/` folders, and the notebook copy under `notebooks/` [file:922].

```text
.
├── README.md
├── README_demo.md
├── config.yaml
├── requirements.txt
├── requirements-gen.txt
├── download_data.py
├── train.py
├── evaluate.py
├── inference.py
├── demo.py
├── dataset.py
├── model.py
├── analyze_class_errors.py
├── generate_img2img.py
├── generate_class.py
├── generate_class22.py
├── generate_22_27.py
├── generation_plan.json
├── generation_plan_22_27.json
├── data/
├── models/
├── results/
└── notebooks/
```

## Environment

The uploaded notebook was run in Colab with GPU acceleration enabled, and it explicitly reports `CUDA available: True` and `GPU: Tesla T4` before training begins [file:922]. The notebook installs project dependencies from `requirements-gen.txt` during setup [file:922].

## Full project run

### 1. Clone the repository

The full notebook starts by cloning the public GitHub repository into `/content/gtsrb_repo` and then listing the files to confirm the download [file:922].

```bash
git clone https://github.com/meiravnitzan/gtsrb_repo.git
cd gtsrb_repo
```

### 2. Install dependencies

The setup cell installs the environment with the requirements file used in the notebook run [file:922].

```bash
pip install -q -r requirements-gen.txt
```

If you prefer to manage dependencies locally, create and activate a Python environment first, then run the same install command.

### 3. Download the dataset

The notebook downloads the official GTSRB training images, test images, and ground-truth CSV through `download_data.py`, then extracts them under `data/` [file:922]. After extraction, the notebook reports these key paths: `data/GTSRB/Final_Training/Images`, `data/GTSRB/Final_Test/Images`, and `data/GT-final_test.csv` [file:922].

```bash
python download_data.py
```

### 4. Train the baseline model

The baseline experiment is launched with `train.py --config config.yaml --mode baseline` [file:922]. During that run, the notebook reports `Train samples: 33337` and `Val samples: 5872` [file:922].

```bash
python train.py --config config.yaml --mode baseline
```

The training log in the notebook shows 8 epochs for the baseline mode and saves the best checkpoint to `models/best_baseline.pt` [file:922]. The notebook also writes training history to `results/history_baseline.json` [file:922].

### 5. Evaluate the trained checkpoint

The notebook evaluates the saved checkpoint with `evaluate.py` using the baseline tag [file:922].

```bash
python evaluate.py --config config.yaml --checkpoint models/best_baseline.pt --tag baseline
```

From the notebook run, the baseline evaluation produced:

- Accuracy: `0.9898653998416469` [file:922]
- Macro F1: `0.984752082504987` [file:922]
- Metrics JSON: `results/metrics_baseline.json` [file:922]
- Confusion matrix image: `results/confusion_matrix_baseline.png` [file:922]
- Predictions CSV: `results/predictions/predictions_baseline.csv` [file:922]

### 6. Inspect error cases

The full notebook includes an optional analysis step for the most error-prone class. In the reported baseline run, the most misclassified true class was class `22`, with `30` misclassified samples, and the top confusion pair was true label `22` predicted as `25` with count `22` [file:922].

The analysis command used in the notebook is:

```bash
python analyze_class_errors.py \
  --csv-path results/predictions/predictions_baseline.csv \
  --data-dir data/GTSRB/Final_Training/Images \
  --class-idx 22 \
  --output-dir outputs/class22_report
```

That step saved multiple visual summaries, including `outputs/class22_report/class22_as_25_25.png`, which the notebook then displayed inline [file:922].

## Main artifacts

After a full run, the notebook confirms the following core artifacts are produced:

- `models/best_baseline.pt` for the best saved checkpoint [file:922]
- `results/history_baseline.json` for epoch-by-epoch training history [file:922]
- `results/metrics_baseline.json` for evaluation metrics and error summary [file:922]
- `results/confusion_matrix_baseline.png` for the confusion matrix visualization [file:922]
- `results/predictions/predictions_baseline.csv` for per-sample predictions [file:922]
- `outputs/class22_report/*.png` for class-specific error analysis images [file:922]

## Quick reproduction

To reproduce the same baseline workflow end to end, run:

```bash
git clone https://github.com/meiravnitzan/gtsrb_repo.git
cd gtsrb_repo
pip install -q -r requirements-gen.txt
python download_data.py
python train.py --config config.yaml --mode baseline
python evaluate.py --config config.yaml --checkpoint models/best_baseline.pt --tag baseline
python analyze_class_errors.py \
  --csv-path results/predictions/predictions_baseline.csv \
  --data-dir data/GTSRB/Final_Training/Images \
  --class-idx 22 \
  --output-dir outputs/class22_report
```

## Notes

The project kit asks for a README that explains how to set up the environment, how to run the code, and how to understand the key outputs of the project deliverables [file:923]. This README is written to match that requirement while reflecting the actual commands and outputs captured in the uploaded full-run notebook [file:923][file:922].
