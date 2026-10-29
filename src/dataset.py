import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

SEED = 44

CLASSES = ["avion", "automovil", "pajaro", "gato", "ciervo",
           "perro", "rana", "caballo", "barco", "camion"]
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def get_datasets(root="./data", seed=SEED, val_ratio=0.1):
    set_seed(seed)
    train_tf, eval_tf = get_transforms()

    full_train_aug = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    full_train_eval = datasets.CIFAR10(root, train=True, download=True, transform=eval_tf)
    test_set = datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)

    indices = np.arange(len(full_train_aug))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_ratio,
        stratify=full_train_aug.targets, random_state=seed)

    train_set = Subset(full_train_aug, train_idx)
    val_set = Subset(full_train_eval, val_idx)

    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    print(f"Shape de una imagen: {tuple(train_set[0][0].shape)}")
    return train_set, val_set, test_set


def get_dataloaders(batch_size=128, seed=SEED, root="./data"):
    train_set, val_set, test_set = get_datasets(root, seed)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    xb, yb = next(iter(train_loader))
    print(f"Batch 4D: {tuple(xb.shape)} (Batch, Channels, Height, Width)")
    return train_loader, val_loader, test_loader


def show_samples(n=16, root="./data", save_path=None):
    raw = datasets.CIFAR10(root, train=True, download=True)
    rng = np.random.RandomState(SEED)
    idxs = rng.choice(len(raw), n, replace=False)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8, 2 * rows))
    for ax, i in zip(axes.ravel(), idxs):
        img, label = raw[i]
        ax.imshow(img)
        ax.set_title(CLASSES[label], fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    show_samples(16, save_path="outputs/samples.png")
    get_dataloaders()