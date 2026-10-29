import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Bloque 1: (B,3,32,32) -> (B,32,16,16)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2))
        # Bloque 2: (B,32,16,16) -> (B,64,8,8)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2))
        # Bloque 3: (B,64,8,8) -> (B,128,4,4)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2))
        self.flatten = nn.Flatten()          # (B,128,4,4) -> (B,2048)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes))     # (B,256) -> (B,10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        return self.classifier(x)

    def forward_verbose(self, x):
        print(f"Entrada          : {tuple(x.shape)}")
        for name in ["block1", "block2", "block3", "flatten"]:
            x = getattr(self, name)(x)
            print(f"Tras {name:<11}: {tuple(x.shape)}")
        x = self.classifier(x)
        print(f"Salida (logits)  : {tuple(x.shape)}")
        return x


def validate_shapes(batch_size=8):
    model = SimpleCNN()
    dummy = torch.randn(batch_size, 3, 32, 32)   # batch simulado 4D
    out = model.forward_verbose(dummy)
    assert out.shape == (batch_size, 10), f"Shape inesperado: {out.shape}"
    print("OK: el modelo procesa tensores 4D sin Shape Mismatch.")
    print(f"Parametros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


if __name__ == "__main__":
    validate_shapes()