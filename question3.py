


import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    average_precision_score
)

# =========================
# CONFIG
# =========================

DATASET_PATH = r"C:\Users\bagheri\Downloads\Cataract_dataset_classification.v1i.multiclass"

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 1e-4
N_SPLITS = 3
THRESHOLD = 0.5
NUM_WORKERS = 0



OUTPUT_DIR = Path("question3_kfold_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================
# Dataset
# =========================
class CataractMultiLabelDataset(Dataset):
    def __init__(self, dataframe, image_root, label_cols, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.label_cols = label_cols
        self.transform = transform
        self.image_col = self.df.columns[0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.image_root / row[self.image_col]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        labels = row[self.label_cols].values.astype("float32")
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels


# =========================
# Read all data
# =========================
def load_split(split_name):
    split_dir = Path(DATASET_PATH) / split_name
    csv_path = split_dir / "_classes.csv"

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df[df.columns[0]] = df[df.columns[0]].astype(str).str.strip()

    df["split_dir"] = str(split_dir)

    return df


train_df = load_split("train")
valid_df = load_split("valid")
test_df = load_split("test")

# برای K-Fold، همه داده‌ها را با هم ادغام می‌کنیم
all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

image_col = all_df.columns[0]
label_cols = list(train_df.columns[1:-1])  # حذف filename و split_dir

print("Labels:", label_cols)
print("Total images:", len(all_df))
print("Number of labels:", len(label_cols))

X = all_df[image_col].values
Y = all_df[label_cols].values.astype(int)


# =========================
# Transform
# =========================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

valid_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# Model
# =========================
def create_model(num_labels):
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_labels)

    return model


# =========================
# Accuracy
# =========================
def multilabel_accuracy(outputs, labels, threshold=0.5):
    probs = torch.sigmoid(outputs)
    preds = (probs >= threshold).float()
    correct = (preds == labels).float()
    return correct.mean().item()


# =========================
# Train one epoch
# =========================
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_acc = 0
    batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        acc = multilabel_accuracy(outputs, labels, THRESHOLD)

        total_loss += loss.item()
        total_acc += acc
        batches += 1

    return total_loss / batches, total_acc / batches


# =========================
# Evaluation
# =========================
def evaluate_model(model, loader):
    model.eval()

    all_true = []
    all_prob = []
    all_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= THRESHOLD).astype(int)

            all_prob.append(probs)
            all_pred.append(preds)
            all_true.append(labels.numpy())

    y_true = np.vstack(all_true)
    y_prob = np.vstack(all_prob)
    y_pred = np.vstack(all_pred)

    metrics = {
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred)
    }

    try:
        metrics["mAP_macro"] = average_precision_score(y_true, y_prob, average="macro")
        metrics["mAP_micro"] = average_precision_score(y_true, y_prob, average="micro")
    except:
        metrics["mAP_macro"] = None
        metrics["mAP_micro"] = None

    return metrics


# =========================
# Multi-label Stratified K-Fold
# =========================
mskf = MultilabelStratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=42
)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(mskf.split(X, Y), start=1):
    print("\n" + "=" * 60)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("=" * 60)

    fold_train_df = all_df.iloc[train_idx].copy()
    fold_val_df = all_df.iloc[val_idx].copy()

    # چون تصاویر از train/valid/test آمده‌اند، باید مسیر هر تصویر را جداگانه نگه داریم.
    # برای ساده‌سازی، فایل‌های هر split در همان پوشه خودش خوانده می‌شوند.
    class MultiRootDataset(Dataset):
        def __init__(self, dataframe, label_cols, transform=None):
            self.df = dataframe.reset_index(drop=True)
            self.label_cols = label_cols
            self.transform = transform
            self.image_col = self.df.columns[0]

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image_path = Path(row["split_dir"]) / row[self.image_col]

            image = Image.open(image_path).convert("RGB")

            labels = row[self.label_cols].values.astype("float32")
            labels = torch.tensor(labels, dtype=torch.float32)

            if self.transform:
                image = self.transform(image)

            return image, labels

    train_dataset = MultiRootDataset(
        fold_train_df,
        label_cols,
        transform=train_transform
    )

    val_dataset = MultiRootDataset(
        fold_val_df,
        label_cols,
        transform=valid_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # محاسبه pos_weight فقط با train همان fold
    train_labels = fold_train_df[label_cols].values.astype("float32")
    pos_counts = train_labels.sum(axis=0)
    neg_counts = len(train_labels) - pos_counts
    pos_weight = neg_counts / np.maximum(pos_counts, 1)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

    model = create_model(len(label_cols)).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_weights = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)

    metrics = evaluate_model(model, val_loader)
    metrics["fold"] = fold

    fold_results.append(metrics)

    # ذخیره نمودارهای هر fold
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title(f"Fold {fold} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"fold_{fold}_loss.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.title(f"Fold {fold} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"fold_{fold}_accuracy.png", dpi=300)
    plt.close()

    torch.save(
        model.state_dict(),
        OUTPUT_DIR / f"efficientnet_b0_fold_{fold}.pth"
    )


# =========================
# Final results
# =========================
results_df = pd.DataFrame(fold_results)
results_df = results_df[
    ["fold"] + [c for c in results_df.columns if c != "fold"]
]

results_df.to_excel(OUTPUT_DIR / "kfold_results.xlsx", index=False)

print("\n" + "=" * 60)
print("K-FOLD FINAL RESULTS")
print("=" * 60)

print(results_df)

print("\nMean Results:")
print(results_df.drop(columns=["fold"]).mean(numeric_only=True))

print("\nStd Results:")
print(results_df.drop(columns=["fold"]).std(numeric_only=True))

summary_df = pd.DataFrame({
    "metric": results_df.drop(columns=["fold"]).columns,
    "mean": results_df.drop(columns=["fold"]).mean(numeric_only=True).values,
    "std": results_df.drop(columns=["fold"]).std(numeric_only=True).values
})

summary_df.to_excel(OUTPUT_DIR / "kfold_summary.xlsx", index=False)

print("\nFiles saved in:", OUTPUT_DIR.resolve())