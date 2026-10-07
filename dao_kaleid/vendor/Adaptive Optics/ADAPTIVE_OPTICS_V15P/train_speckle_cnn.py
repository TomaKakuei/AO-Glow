import os
import glob
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# -------------------------
# 1. Multi-Scale Front-End (Paine 2018)
# -------------------------
class MultiScaleStem(nn.Module):
    """
    Extracts features at multiple scales to handle both high-frequency speckle 
    and large low-frequency aberrations simultaneously.
    """
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        # Branch 1: Fine details (3x3)
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        # Branch 2: Medium details (7x7)
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        # Branch 3: Coarse structure (11x11)
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        # Branch 4: Max Pooling
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
        self.pool = nn.MaxPool2d(2, 2) # Downsample to 64x64

    def forward(self, x):
        f1 = self.b1(x)
        f2 = self.b2(x)
        f3 = self.b3(x)
        f4 = self.b4(x)
        out = torch.cat([f1, f2, f3, f4], dim=1) # out_channels total
        return self.pool(out)

# -------------------------
# 2. Depthwise Separable Block (Hu 2019)
# -------------------------
class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                   padding=1, stride=stride, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.skip(x)
        out = self.act(self.bn1(self.depthwise(x)))
        out = self.bn2(self.pointwise(out))
        out += residual
        return self.act(out)

# -------------------------
# 3. ResMLP Backend (DAO Original)
# -------------------------
class ResBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 4)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(width * 4, width)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.ln1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return res + x

class MonolithicResMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int = 2048, blocks: int = 8, dropout: float = 0.1):
        super().__init__()
        layers = [
            nn.Linear(input_dim, width),
            nn.SiLU(),
        ]
        for _ in range(blocks):
            layers.append(ResBlock(width=width, dropout=dropout))
        layers.extend([
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, output_dim)
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# -------------------------
# 4. Full Sensorless Network
# -------------------------
class SpeckleCNN(nn.Module):
    def __init__(self, output_dim=20):
        super().__init__()
        self.stem = MultiScaleStem(in_channels=9, out_channels=64)
        
        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2
        self.stage1 = DepthwiseSeparableBlock(64, 128, stride=2)
        self.stage2 = DepthwiseSeparableBlock(128, 256, stride=2)
        self.stage3 = DepthwiseSeparableBlock(256, 512, stride=2)
        self.stage4 = DepthwiseSeparableBlock(512, 1024, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 20 DOF output (4 anchors * 5 DOFs)
        self.mlp = MonolithicResMLP(input_dim=1024, output_dim=output_dim, width=2048, blocks=8, dropout=0.1)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.mlp(x)

# -------------------------
# 5. Dataset & Training Loop
# -------------------------
class SpeckleDataset(Dataset):
    def __init__(self, data_dirs: list[str], mode: str = "speckle"):
        self.files = []
        for d in data_dirs:
            self.files.extend(sorted(glob.glob(os.path.join(d, "*.npz"))))
        self.mode = mode
        
        print(f"Loading {len(self.files)} dataset files into memory for mode {mode}...")
        all_imgs = []
        all_mechs = []
        for f in self.files:
            with np.load(f) as data:
                if mode == "speckle":
                    all_imgs.append(data['speckles'])
                else:
                    all_imgs.append(data['psfs'])
                all_mechs.append(data['mechs'])
                
        self.imgs = np.concatenate(all_imgs, axis=0)
        self.mechs = np.concatenate(all_mechs, axis=0)
        
        # Data-driven gain: Z-score normalization for labels
        self.mech_mean = np.mean(self.mechs, axis=0)
        self.mech_std = np.std(self.mechs, axis=0)
        # Avoid division by zero
        self.mech_std[self.mech_std < 1e-8] = 1e-8
        
        self.total_size = self.imgs.shape[0]
        print(f"Loaded {self.total_size} samples.")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        img = self.imgs[idx]
        target = self.mechs[idx]
        
        # Apply data-driven normalization
        target = (target - self.mech_mean) / self.mech_std
        
        # Normalize image to [0, 1] per channel (FOV)
        img = img.astype(np.float32)
        channel_max = np.max(img, axis=(1, 2), keepdims=True)
        img = img / (channel_max + 1e-6)
        
        return torch.tensor(img), torch.tensor(target, dtype=torch.float32)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, choices=['front', 'rear'], required=True, help="Lens group to train (front or rear).")
    parser.add_argument("--mode", type=str, choices=["speckle", "psf"], default="speckle")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--resume', action='store_true', help="Resume training from existing checkpoint if available.")
    args = parser.parse_args()
    
    # Use both original (small range) and new (large range) datasets
    args.data_dirs = [f"artifacts/{args.group}_speckle_dataset", f"artifacts/speckle_dataset_{args.group}"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = SpeckleDataset(args.data_dirs, mode=args.mode)
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return
        
    # Simple split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SpeckleCNN(output_dim=20).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    out_path = f"artifacts/{args.mode}_cnn_{args.group}.pth"
    best_val_loss = float('inf')
    
    # Optional resume
    if args.resume and os.path.exists(out_path):
        print(f"Resuming from checkpoint {out_path}...")
        checkpoint = torch.load(out_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        # best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    print(f"Starting training {args.mode} model...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = F.huber_loss(preds, targets, delta=1.0)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            
        train_loss /= len(train_dataset)
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs)
                loss = F.huber_loss(preds, targets, delta=1.0)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_dataset)
        
        print(f"Epoch {epoch+1:03d} | Train L1: {train_loss:.6f} | Val L1: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'state_dict': model.state_dict(),
                'mech_mean': dataset.mech_mean,
                'mech_std': dataset.mech_std
            }, out_path)
            print(f"  -> Best model saved to {out_path} (Val L1: {val_loss:.6f})")

if __name__ == "__main__":
    train()
