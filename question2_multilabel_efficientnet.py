import os
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

from sklearn.metrics import (
    precision_score, recall_score, f1_score, hamming_loss,
    average_precision_score, classification_report,
    multilabel_confusion_matrix
)

# =========================
# CONFIG
# =========================
DATASET_PATH = r"C:\Users\bagheri\Downloads\Cataract_dataset_classification.v1i.multiclass"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
THRESHOLD = 0.5
NUM_WORKERS = 0

OUTPUT_DIR = Path("question2_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


class CataractMultiLabelDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.transform = transform

        csv_path = self.folder_path / "_classes.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"_classes.csv not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()

        self.image_col = self.df.columns[0]
        self.label_cols = list(self.df.columns[1:])
        self.df[self.image_col] = self.df[self.image_col].astype(str).str.strip()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.folder_path / row[self.image_col]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        labels = row[self.label_cols].values.astype("float32")
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

valid_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


train_dir = Path(DATASET_PATH) / "train"
valid_dir = Path(DATASET_PATH) / "valid"
test_dir = Path(DATASET_PATH) / "test"

train_dataset = CataractMultiLabelDataset(train_dir, train_transform)
valid_dataset = CataractMultiLabelDataset(valid_dir, valid_test_transform)
test_dataset = CataractMultiLabelDataset(test_dir, valid_test_transform)

label_names = train_dataset.label_cols
num_labels = len(label_names)

print("Labels:", label_names)
print("Number of labels:", num_labels)
print("Train images:", len(train_dataset))
print("Valid images:", len(valid_dataset))
print("Test images:", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# Class imbalance handling with pos_weight
train_labels_matrix = train_dataset.df[label_names].values.astype("float32")
positive_counts = train_labels_matrix.sum(axis=0)
negative_counts = len(train_labels_matrix) - positive_counts
pos_weight = negative_counts / np.maximum(positive_counts, 1)
pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

pd.DataFrame({
    "label": label_names,
    "positive_count": positive_counts.astype(int),
    "negative_count": negative_counts.astype(int),
    "pos_weight": pos_weight.numpy()
}).to_excel(OUTPUT_DIR / "label_pos_weights.xlsx", index=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_labels)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def multilabel_accuracy(outputs, labels, threshold=0.5):
    probs = torch.sigmoid(outputs)
    preds = (probs >= threshold).float()
    return (preds == labels).float().mean().item()


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    running_acc = 0.0
    total_batches = 0

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
        running_loss += loss.item()
        running_acc += acc
        total_batches += 1

    return running_loss / total_batches, running_acc / total_batches


def train_model(model, epochs):
    best_weights = copy.deepcopy(model.state_dict())
    best_valid_loss = float("inf")

    history = {"train_loss": [], "valid_loss": [], "train_acc": [], "valid_acc": []}

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        valid_loss, valid_acc = run_epoch(model, valid_loader, criterion, optimizer=None)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), OUTPUT_DIR / "best_efficientnet_b0_multilabel.pth")

    model.load_state_dict(best_weights)
    return model, history


model, history = train_model(model, EPOCHS)

# Curves
plt.figure(figsize=(8, 5))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["valid_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history["train_acc"], label="Train Accuracy")
plt.plot(history["valid_acc"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Multi-label Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=300)
plt.show()

pd.DataFrame(history).to_excel(OUTPUT_DIR / "training_history.xlsx", index=False)


def predict(model, loader):
    model.eval()
    all_true, all_prob, all_pred = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= THRESHOLD).astype(int)

            all_prob.append(probs)
            all_pred.append(preds)
            all_true.append(labels.numpy())

    return np.vstack(all_true), np.vstack(all_prob), np.vstack(all_pred)


y_true, y_prob, y_pred = predict(model, test_loader)

metrics = {
    "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
    "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
    "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
    "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
    "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
    "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    "hamming_loss": hamming_loss(y_true, y_pred),
}

try:
    metrics["mAP_macro"] = average_precision_score(y_true, y_prob, average="macro")
    metrics["mAP_micro"] = average_precision_score(y_true, y_prob, average="micro")
except Exception as e:
    metrics["mAP_macro"] = None
    metrics["mAP_micro"] = None
    print("mAP warning:", e)

print("\n========== TEST METRICS ==========")
for k, v in metrics.items():
    print(f"{k}: {v}")

pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()]).to_excel(
    OUTPUT_DIR / "test_metrics.xlsx", index=False
)

report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
print("\n========== CLASSIFICATION REPORT ==========")
print(report)

with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

mcm = multilabel_confusion_matrix(y_true, y_pred)
rows = []
for i, label in enumerate(label_names):
    tn, fp, fn, tp = mcm[i].ravel()
    rows.append({"label": label, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})

confusion_df = pd.DataFrame(rows)
confusion_df.to_excel(OUTPUT_DIR / "multilabel_confusion_matrix.xlsx", index=False)

plt.figure(figsize=(12, 6))
plt.bar(confusion_df["label"], confusion_df["TP"], label="TP")
plt.bar(confusion_df["label"], confusion_df["FP"], bottom=confusion_df["TP"], label="FP")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Count")
plt.title("TP and FP per Label")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tp_fp_per_label.png", dpi=300)
plt.show()

torch.save(model.state_dict(), OUTPUT_DIR / "final_efficientnet_b0_multilabel.pth")
print("\nAll outputs saved in:", OUTPUT_DIR.resolve())
