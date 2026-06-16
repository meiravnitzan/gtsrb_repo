# GTSRB Demo Run

This README describes the lightweight Colab demo workflow for the GTSRB traffic sign classification project. It is meant for presenting the trained baseline and augmented models without rerunning the full training pipeline.

## Purpose

Use this demo notebook to show how synthetic augmentation for class 22 improves traffic sign classification on selected examples, while also displaying generated synthetic images and baseline error cases.

## Files involved

The main demo files are:

- `x401_GTSRB_project_demo.ipynb` — the Colab notebook that runs the demo flow.
- `demo.py` — helper functions loaded into the notebook with `%run`.
- `README_demo.md` — this guide.

## Demo setup

### 1. Clone the repository

Run the following in Colab:

```python
%cd /content
!rm -rf gtsrb_repo
!git clone https://github.com/meiravnitzan/gtsrb_repo.git
%cd /content/gtsrb_repo
```

The notebook starts by cloning the public repository into `/content/gtsrb_repo`.

### 2. Install dependencies

Install the demo dependencies:

```python
!pip install -q -r requirements-gen.txt
```

The notebook uses `requirements-gen.txt` during setup.

### 3. Download the dataset assets

Because the dataset is not stored in the GitHub repository, download and extract it with:

```python
!python download_data.py
```

This step downloads the official GTSRB training images, test images, and test labels into the `data/` directory.

### 4. Load the demo helpers

Run:

```python
%run /content/gtsrb_repo/demo.py
```

This loads the helper functions and trained-model utilities into the active Colab session so they can be called from later notebook cells.

## Suggested demo flow

The demo focuses on class 22 (`bumpy_road`), because the augmentation work targeted that class. The same helper functions can also be used with other classes when relevant.

### 1. Find baseline mistakes for class 22

List misclassified baseline examples for class 22:

```python
bad_22 = list_misclassified_images_for_class(22)
len(bad_22), bad_22[:5]
```

In the notebook, this returns misclassified test images including `/content/gtsrb_repo/data/GTSRB/Final_Test/Images/00339.ppm`, with several examples predicted as class 25.

### 2. Preview misclassified examples

Display a few class-22 images that the baseline model predicted as class 25:

```python
show_misclassified_examples(22, predicted_as=25, max_show=3)
```

This is useful for introducing the failure mode before showing the model comparison.

### 3. Show baseline prediction

Pick one misclassified image and show the baseline model output:

```python
img = "/content/gtsrb_repo/data/GTSRB/Final_Test/Images/00339.ppm"
show_one_model(baseline_model, img, model_name="Baseline only", true_label=22)
```

The notebook uses this exact image to demonstrate a baseline error on class 22.

### 4. Show generated augmented images

At this stage of the project, generated synthetic images are available for class 22. Display selected ranges with:

```python
show_generated_images(22, 20, 27)
show_generated_images(22, 70, 77)
```

The notebook uses both of these calls to display samples from the generated class-22 folder.

### 5. Show augmented-model prediction

Run the same test image through the augmented model:

```python
img = "/content/gtsrb_repo/data/GTSRB/Final_Test/Images/00339.ppm"
show_one_model(augmented_model, img, model_name="Augmented only", true_label=22)
```

This allows a direct before-and-after comparison using the same input image.

## Notes

- The demo notebook is `x401_GTSRB_project_demo.ipynb`.
- The flow is intended for presentation and inspection, not for retraining the models.
