"""
Exercicio 01 - Redes Neurais Profundas (UNISINOS)
Classificacao de radiografias toracicas (Normal x Pneumonia) com CNN,
usando o dataset RSNA Pneumonia Detection Challenge (ChestX-ray RSNA).

Etapas:
  1. Download / organizacao do dataset (DICOM -> PNG em pastas train/val/test)
  2. Analise exploratoria + pre-processamento (resize, normalizacao)
  3. Construcao da CNN
  4. Treinamento e validacao
  5. Avaliacao (matriz de confusao, precision, recall, F1-score)

Dependencias: torch, torchvision, pydicom, pandas, numpy, pillow,
              scikit-learn, seaborn, matplotlib, tqdm, kaggle
"""

import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------
SEED = 42
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
ORGANIZED_DIR = BASE_DIR / "data" / "organized"
OUTPUT_DIR = BASE_DIR / "outputs"

COMPETITION = "rsna-pneumonia-detection-challenge"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # indice 0 / 1 == Target 0 / 1 do RSNA

IMG_SIZE = 128
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# 1. Download e organizacao do dataset
# ---------------------------------------------------------------------------
def download_dataset():
    """Baixa e extrai o dataset da competicao RSNA via Kaggle CLI, se necessario."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    labels_csv = RAW_DIR / "stage_2_train_labels.csv"
    images_dir = RAW_DIR / "stage_2_train_images"

    if labels_csv.exists() and images_dir.exists():
        print("Dataset bruto ja presente em", RAW_DIR)
        return

    zip_path = RAW_DIR / f"{COMPETITION}.zip"
    if not zip_path.exists():
        print("Baixando dataset do Kaggle (isso pode demorar, ~4GB)...")
        result = subprocess.run(
            [sys.executable, "-m", "kaggle", "competitions", "download",
             COMPETITION, "-p", str(RAW_DIR)],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(
                "Falha ao baixar dataset do Kaggle. Verifique se:\n"
                "  1) o token esta salvo em %USERPROFILE%\\.kaggle\\access_token\n"
                "  2) voce aceitou as regras da competicao em "
                f"https://www.kaggle.com/c/{COMPETITION}/rules"
            )

    print("Extraindo arquivos...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    print("Download/extracao concluidos.")


def build_labels_dataframe():
    """Le o CSV de labels e retorna um paciente/target unico (sem duplicatas de bbox)."""
    labels_csv = RAW_DIR / "stage_2_train_labels.csv"
    df = pd.read_csv(labels_csv)
    df = df[["patientId", "Target"]].drop_duplicates(subset="patientId").reset_index(drop=True)
    return df


def split_dataset(df):
    """Divide os pacientes em train/val/test de forma estratificada pelo Target."""
    train_df, temp_df = train_test_split(
        df, train_size=TRAIN_FRAC, stratify=df["Target"], random_state=SEED
    )
    rel_val = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    val_df, test_df = train_test_split(
        temp_df, train_size=rel_val, stratify=temp_df["Target"], random_state=SEED
    )
    return {"train": train_df, "val": val_df, "test": test_df}


def dicom_to_png(dcm_path: Path, out_path: Path):
    """Converte um arquivo DICOM em PNG em escala de cinza, redimensionado."""
    dcm = pydicom.dcmread(dcm_path)
    pixels = dcm.pixel_array.astype(np.uint8)
    if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = 255 - pixels
    img = Image.fromarray(pixels, mode="L").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img.save(out_path)


def organize_images(splits: dict):
    """Organiza as imagens DICOM em pastas train/val/test/{NORMAL,PNEUMONIA} como PNG."""
    images_dir = RAW_DIR / "stage_2_train_images"

    for split_name, split_df in splits.items():
        for _, class_name in enumerate(CLASS_NAMES):
            (ORGANIZED_DIR / split_name / class_name).mkdir(parents=True, exist_ok=True)

        pending = []
        for _, row in split_df.iterrows():
            class_name = CLASS_NAMES[row["Target"]]
            out_path = ORGANIZED_DIR / split_name / class_name / f"{row['patientId']}.png"
            if not out_path.exists():
                pending.append((row["patientId"], out_path))

        if not pending:
            print(f"[{split_name}] ja organizado ({len(split_df)} imagens).")
            continue

        print(f"[{split_name}] convertendo {len(pending)} imagens DICOM -> PNG...")
        for patient_id, out_path in tqdm(pending, desc=split_name):
            dcm_path = images_dir / f"{patient_id}.dcm"
            dicom_to_png(dcm_path, out_path)


# ---------------------------------------------------------------------------
# 2. Analise exploratoria
# ---------------------------------------------------------------------------
def exploratory_analysis(df: pd.DataFrame, splits: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nDistribuicao geral de classes:")
    counts = df["Target"].value_counts().rename({0: "NORMAL", 1: "PNEUMONIA"})
    print(counts)

    print("\nDistribuicao por split:")
    for split_name, split_df in splits.items():
        split_counts = split_df["Target"].value_counts().rename({0: "NORMAL", 1: "PNEUMONIA"})
        print(f"  {split_name}: {split_counts.to_dict()} (total={len(split_df)})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    counts.plot(kind="bar", ax=axes[0], color=["#4C72B0", "#C44E52"])
    axes[0].set_title("Distribuicao geral de classes")
    axes[0].set_ylabel("Numero de imagens")

    split_summary = pd.DataFrame({
        name: sdf["Target"].value_counts().rename({0: "NORMAL", 1: "PNEUMONIA"})
        for name, sdf in splits.items()
    }).T
    split_summary.plot(kind="bar", stacked=True, ax=axes[1], color=["#4C72B0", "#C44E52"])
    axes[1].set_title("Distribuicao por split (train/val/test)")
    axes[1].set_ylabel("Numero de imagens")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)
    print(f"\nGrafico salvo em {OUTPUT_DIR / 'class_distribution.png'}")


# ---------------------------------------------------------------------------
# 3. DataLoaders
# ---------------------------------------------------------------------------
def build_dataloaders():
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_data = datasets.ImageFolder(ORGANIZED_DIR / "train", transform=train_transform)
    val_data = datasets.ImageFolder(ORGANIZED_DIR / "val", transform=eval_transform)
    test_data = datasets.ImageFolder(ORGANIZED_DIR / "test", transform=eval_transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_data


# ---------------------------------------------------------------------------
# 4. Arquitetura da CNN
# ---------------------------------------------------------------------------
class ChestXRayCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# 5. Treinamento e validacao
# ---------------------------------------------------------------------------
def compute_class_weights(train_data):
    targets = np.array(train_data.targets)
    class_counts = np.bincount(targets)
    weights = class_counts.sum() / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, class_weights):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = OUTPUT_DIR / "best_model.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoca {epoch}/{NUM_EPOCHS} - "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} - "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping na epoca {epoch} (sem melhora por "
                      f"{EARLY_STOPPING_PATIENCE} epocas).")
                break

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    return model, history


def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_loss"], label="Treino")
    axes[0].plot(history["val_loss"], label="Validacao")
    axes[0].set_title("Loss por epoca")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Treino")
    axes[1].plot(history["val_acc"], label="Validacao")
    axes[1].set_title("Acuracia por epoca")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Acuracia")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"Curvas de treino salvas em {OUTPUT_DIR / 'training_curves.png'}")


# ---------------------------------------------------------------------------
# 6. Avaliacao
# ---------------------------------------------------------------------------
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\nAcuracia no teste: {acc:.4f}")
    print(f"Precision (PNEUMONIA): {precision:.4f}")
    print(f"Recall (PNEUMONIA): {recall:.4f}")
    print(f"F1-score (PNEUMONIA): {f1:.4f}")
    print("\nRelatorio de classificacao:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusao - Conjunto de Teste")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Matriz de confusao salva em {OUTPUT_DIR / 'confusion_matrix.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Dispositivo em uso: {DEVICE}")

    print("\n=== 1. Download e organizacao do dataset ===")
    download_dataset()
    labels_df = build_labels_dataframe()
    splits = split_dataset(labels_df)
    organize_images(splits)

    print("\n=== 2. Analise exploratoria ===")
    exploratory_analysis(labels_df, splits)

    print("\n=== 3. Preparando DataLoaders ===")
    train_loader, val_loader, test_loader, train_data = build_dataloaders()
    print(f"Classes (ImageFolder): {train_data.classes}")

    print("\n=== 4. Construindo a CNN ===")
    model = ChestXRayCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    class_weights = compute_class_weights(train_data)
    print(f"Pesos de classe (para lidar com desbalanceamento): {class_weights.tolist()}")

    print("\n=== 5. Treinamento e validacao ===")
    model, history = train_model(model, train_loader, val_loader, class_weights)
    plot_training_curves(history)

    print("\n=== 6. Avaliacao no conjunto de teste ===")
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()
