import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from dataset import SEED, CLASSES, set_seed, get_dataloaders
from model import SimpleCNN, validate_shapes

EPOCHS = 15          # prueba primero con 2 y sube después
BATCH_SIZE = 128
LR = 1e-3
OUT_DIR = "outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def plot_curves(h, path):
    ep = range(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ep, h["train_loss"], label="Train")
    ax[0].plot(ep, h["val_loss"], label="Validacion")
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoca"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep, h["train_acc"], label="Train")
    ax[1].plot(ep, h["val_acc"], label="Validacion")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoca"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    validate_shapes()
    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE, SEED)

    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    for epoch in range(EPOCHS):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer)
        vl, va = run_epoch(model, val_loader, criterion)
        for k, v in zip(hist, [tl, ta, vl, va]):
            hist[k].append(v)
        print(f"Epoca {epoch+1:02d}/{EPOCHS} | train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}")

    plot_curves(hist, f"{OUT_DIR}/loss_accuracy.png")

    # Evaluacion final en el conjunto de prueba
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x.to(DEVICE)).argmax(1).cpu())
            labels.append(y)
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    test_acc = (preds == labels).mean()

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Prediccion"); plt.ylabel("Real")
    plt.title(f"Matriz de confusion (test acc {test_acc:.4f})")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    print(f"\nTEST ACCURACY: {test_acc:.4f}")
    print(classification_report(labels, preds, target_names=CLASSES))
    print(f"Brecha final train-val (acc): {hist['train_acc'][-1] - hist['val_acc'][-1]:.4f}")

    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump({"seed": SEED, "test_acc": float(test_acc), **hist}, f, indent=2)


if __name__ == "__main__":
    main()