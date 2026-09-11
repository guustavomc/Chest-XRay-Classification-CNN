# Chest-XRay-Classification-CNN

> Developed for the Deep Learning course (Redes Neurais Profundas) — Specialization in Applied Artificial Intelligence, UNISINOS.

A convolutional neural network (CNN) built with PyTorch to classify chest X-rays as **NORMAL** or **PNEUMONIA**, trained on the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) dataset.

Built as a practical exercise for the Deep Learning course (Redes Neurais Profundas) of the AI specialization at UNISINOS. It covers the full computer vision project lifecycle: dataset loading, preprocessing, model design, training/validation, and evaluation.

## Pipeline

`main.py` runs the whole workflow end to end and is idempotent — re-running it skips any step whose output already exists.

1. **Dataset loading & organization** — downloads the competition data via the Kaggle CLI, reads `stage_2_train_labels.csv`, splits patients 70/15/15 (train/val/test) with stratification on the target label, and converts each DICOM image to a resized grayscale PNG organized into `NORMAL` / `PNEUMONIA` folders.
2. **Exploratory analysis** — prints and plots the class distribution overall and per split (the dataset is imbalanced: ~77% NORMAL vs ~23% PNEUMONIA).
3. **Preprocessing** — resize to 128x128, grayscale, normalization, with random horizontal flip augmentation on the training set only.
4. **Model** — a CNN with 4 conv blocks (Conv2d + BatchNorm + ReLU + MaxPool), Kaiming weight initialization, dropout, and a fully connected classifier head.
5. **Training & validation** — Adam optimizer with weight decay, `ReduceLROnPlateau` scheduler, class-weighted cross-entropy loss (to counter class imbalance), and early stopping on validation loss. The best checkpoint is saved automatically.
6. **Evaluation** — accuracy, precision/recall/F1-score, full classification report, and a confusion matrix on the held-out test set.

## Results

Training stopped early at epoch 6/20 (no validation loss improvement for 5 epochs).

| Metric | Value |
|---|---|
| Test accuracy | 0.771 |
| NORMAL — precision / recall / F1 | 0.88 / 0.82 / 0.85 |
| PNEUMONIA — precision / recall / F1 | 0.49 / 0.62 / 0.55 |

The lower scores on PNEUMONIA reflect the class imbalance and the simplicity of the baseline architecture — see [outputs/](outputs/) for the generated plots (class distribution, training curves, confusion matrix) once you run the pipeline.

## Requirements

- Python 3.12
- A Kaggle account with API access:
  1. Generate an API token at [kaggle.com/settings/api](https://www.kaggle.com/settings/api).
  2. Accept the rules of the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules).
  3. Save the token to `%USERPROFILE%\.kaggle\access_token`.

Install dependencies:

```bash
pip install torch torchvision pydicom pandas numpy pillow scikit-learn seaborn matplotlib tqdm kaggle
```

(Use the [PyTorch install selector](https://pytorch.org/get-started/locally/) if you need a specific CUDA build.)

## Usage

```bash
python main.py
```

First run downloads (~4GB) and converts the dataset, then trains and evaluates the model. Subsequent runs reuse the already-downloaded/converted data and just retrain.

## Project structure

```
main.py              # full pipeline (data, model, training, evaluation)
data/                # downloaded + organized dataset (gitignored)
outputs/              # trained model checkpoint and generated plots (gitignored)
Arquivos Base/        # course-provided assignment brief and example notebooks
```
