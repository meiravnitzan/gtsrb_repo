# GTSRB Traffic Sign Classification with Optional Class 22 Synthetic Augmentation

## Problem
This project builds and evaluates a deep learning classifier for the German Traffic Sign Recognition Benchmark (GTSRB). The primary goal is accurate multiclass traffic sign recognition, with a secondary focus on improving performance for an underrepresented class using synthetic augmentation.

The final project submission is centered on a standard GTSRB image classification pipeline implemented in PyTorch. The main experiment compares a baseline training setup against a version augmented with additional synthetic data for class 22.

## Dataset
The core dataset is the German Traffic Sign Recognition Benchmark (GTSRB), a multiclass traffic sign dataset commonly used for image classification research. The training pipeline assumes that the dataset has already been organized locally into the expected class-labeled folder structure.

This repository supports two training settings:
- **Baseline**: train only on the original GTSRB training set.
- **gen22**: train on the original GTSRB data plus additional synthetic images for class 22.

The project submission treats the GTSRB classifier as the main deliverable. The synthetic generation workflow is included only as an optional extension and is **not required** for grading the main submission.

## Repository Structure

```text
PerplexitySuggestion-GTRSB/
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── notebooks/
│   └── .gitkeep
├── results/
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── inference.py
├── demo.py
├── generate_class22.py
├── config.yaml
├── requirements.txt
├── requirements-gen.txt
└── README.md
```

### Folder usage
- `data/` — optional local data placeholder, metadata files, or small helper assets. Generated images are written by default to `data/generated/class22/`.
- `models/` — saved trained checkpoints such as `best_baseline.pt` and `best_gen22.pt`.
- `notebooks/` — optional exploratory or generation notebooks (e.g., `class22_synthetic_augmentation.ipynb`).
- `results/` — evaluation outputs such as confusion matrices, logs, or JSON summaries.

### File usage
- `dataset.py` — dataset loading and preprocessing utilities.
- `model.py` — model architecture definition.
- `train.py` — training entry point.
- `evaluate.py` — evaluation script for trained checkpoints.
- `inference.py` — single-image or batch inference utilities.
- `demo.py` — lightweight demo / prediction script.
- `generate_class22.py` — optional script to generate synthetic images for class 22 using a diffusion model.
- `config.yaml` — central experiment configuration.
- `requirements.txt` — instructor-safe runtime dependencies only.
- `requirements-gen.txt` — full dependency set including optional synthetic generation tools.

## Installation
Two dependency files are provided so the repository is easy to run for grading while still preserving the optional synthetic-data workflow.

### Option 1: Instructor-safe install
Use this file for normal project reproduction, grading, classifier training, evaluation, and inference:

```bash
pip install -r requirements.txt
```

This is the recommended path for:
- instructors,
- TAs,
- anyone reproducing the main classifier results,
- anyone who does **not** need to regenerate synthetic images.

`requirements.txt` contains only the packages needed for the classifier pipeline itself.

### Option 2: Full install with generation dependencies
Use this file only if the optional synthetic-image generation workflow needs to be rerun:

```bash
pip install -r requirements-gen.txt
```

This file includes the full runtime stack plus additional generative-AI dependencies such as Hugging Face / diffusion tooling.

## Requirements files

### `requirements.txt`
Instructor-safe, minimal runtime dependencies for:
- training the classifier,
- evaluating saved checkpoints,
- running inference,
- running the demo script.

This file is sufficient for the main course submission.

### `requirements-gen.txt`
Extended dependency set for:
- synthetic image generation experiments,
- regenerating class 22 synthetic augmentation data,
- reproducing the optional generative augmentation workflow.

This file is **not required** for grading the core project.

## Configuration
The main experiment settings are controlled through `config.yaml`. Typical settings include:
- dataset paths,
- image size,
- batch size,
- learning rate,
- number of epochs,
- checkpoint output path,
- baseline vs. `gen22` run selection.

Before training, update the dataset and output paths in `config.yaml` to match the local environment.

## Training
Train the classifier using the main training script:

```bash
python train.py --config config.yaml
```

Depending on how `config.yaml` is set, this can be used for either:
- a **baseline** run using original GTSRB only, or
- a **gen22** run using original data plus synthetic class 22 samples.

Saved checkpoints should be written to the `models/` folder.

## Evaluation
Evaluate a trained model using:

```bash
python evaluate.py --config config.yaml --checkpoint models/best_gen22.pt
```

Example checkpoints may include:
- `models/best_baseline.pt`
- `models/best_gen22.pt`

Evaluation outputs such as summary metrics, plots, or confusion matrices can be stored under `results/`.

## Inference
Run inference on one image or a small batch using:

```bash
python inference.py --config config.yaml --checkpoint models/best_gen22.pt --image path/to/image.png
```

Use `demo.py` for a simpler end-user prediction flow if desired.

## Results
The main project comparison is:
- **Baseline** model trained on original GTSRB data.
- **gen22** model trained with additional class 22 synthetic samples.

The expected outcome is improved balance across classes, especially for class 22, while maintaining strong overall classifier performance.

Store final artifacts such as:
- best checkpoints in `models/`,
- metrics files in `results/`,
- optional plots in `results/`.

## Demo
The demo script can be used to quickly test a trained checkpoint on one or more images.

Example:

```bash
python demo.py --checkpoint models/best_gen22.pt --image path/to/image.png
```

## [Optional] Class 22 Synthetic Augmentation
This repository also documents an optional extension: synthetic augmentation for **class 22**. This experiment was used to explore whether targeted synthetic data could improve classifier performance for an underrepresented class.

This step is **optional** and is not required to:
- train the baseline classifier,
- evaluate saved checkpoints,
- run inference,
- grade the core project submission.

### When to use this optional workflow
Use the optional generation workflow only if the goal is to:
- recreate the synthetic class 22 images,
- extend the augmentation experiment,
- repeat the full research pipeline from generation through retraining.

### Install dependencies
For this optional workflow, install the extended dependency set:

```bash
pip install -r requirements-gen.txt
```

### Generate synthetic class 22 images (script)
To generate synthetic images with the provided script:

```bash
python generate_class22.py --output_dir data/generated/class22 --num_images 200
```

Add `--show_examples` to visualize a small grid of three generated images at the end:

```bash
python generate_class22.py --output_dir data/generated/class22 --num_images 32 --show_examples
```

Generated images will be saved under `data/generated/class22/` by default.

### Generate synthetic class 22 images (Colab notebook)
Alternatively, you can use a Colab notebook version of the same pipeline, for example:

```text
notebooks/class22_synthetic_augmentation.ipynb
```

The notebook:
- installs the optional generation dependencies,
- loads the same Stable Diffusion pipeline,
- generates class 22 images into `data/generated/class22/`,
- and displays a 3-image sample grid.

### Suggested next steps after generation
After generating synthetic images:
1. Review the images manually.
2. Filter out low-quality or off-distribution samples.
3. Move accepted images into the class 22 training directory used by `dataset.py`.
4. Retrain the classifier using `train.py` with a configuration that includes the augmented data.
5. Compare the updated model against the baseline (macro-F1 and per-class metrics).

### Notes
- The instructor does **not** need to run this step.
- The main project can be trained, evaluated, and graded using `requirements.txt` only.
- This script and notebook are included to document the optional augmentation experiment described in the project.

## Notes for Instructors
For grading and reproducibility of the main submission:
- use `requirements.txt`,
- ignore the optional generation workflow,
- run the classifier pipeline only,
- evaluate the saved checkpoints or retrain from the provided dataset.

The Hugging Face / diffusion stack is included only for the optional synthetic augmentation extension and should not be necessary for standard grading.

## Reproducibility Notes
To improve reproducibility:
- keep dataset paths explicit in `config.yaml`,
- save final checkpoints under `models/`,
- save metrics and figures under `results/`,
- document whether a run is `baseline` or `gen22`.


Versions used:
torch: 2.11.0+cu128
torchvision: 0.26.0+cu128
diffusers: 0.38.0
transformers: 5.10.2
matplotlib: 3.10.0
scikit-learn: 1.6.1
pyyaml: 6.0.3
Pillow: 11.3.0
tqdm: 4.67.3
diffusers: 0.38.0
transformers: 5.10.2
huggingface_hub: 1.18.0
accelerate: 1.13.0
safetensors: 0.8.0