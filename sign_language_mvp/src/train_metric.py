import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from losses import ContrastiveLoss

# ======================
# CONFIG
# ======================
EMBEDDING_DIM = 128
BATCH_SIZE = 16
EPOCHS = 40
LR = 0.001

# ======================
# PATHS
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "fixed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ======================
# LOAD DATA
# ======================
data = {}
for gesture in os.listdir(DATA_DIR):
    samples = []
    for f in os.listdir(os.path.join(DATA_DIR, gesture)):
        samples.append(np.load(os.path.join(DATA_DIR, gesture, f)))
    data[gesture] = samples

gestures = list(data.keys())

# ======================
# PAIR DATASET
# ======================
class GesturePairDataset(Dataset):
    def __init__(self, data, gestures):
        self.data = data
        self.gestures = gestures

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        same = random.choice([0, 1])

        if same:
            g = random.choice(self.gestures)
            a, b = random.sample(self.data[g], 2)
            y = 1
        else:
            g1, g2 = random.sample(self.gestures, 2)
            a = random.choice(self.data[g1])
            b = random.choice(self.data[g2])
            y = 0

        a = torch.tensor(a.reshape(-1), dtype=torch.float32)
        b = torch.tensor(b.reshape(-1), dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return a, b, y

loader = DataLoader(GesturePairDataset(data, gestures),
                    batch_size=BATCH_SIZE, shuffle=True)

# ======================
# MODEL
# ======================
class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30 * 42 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, EMBEDDING_DIM)
        )

    def forward(self, x):
        return self.net(x)

model = EmbeddingNet()
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ======================
# TRAIN
# ======================
for epoch in range(EPOCHS):
    total_loss = 0
    for x1, x2, y in loader:
        optimizer.zero_grad()
        e1 = model(x1)
        e2 = model(x2)
        loss = criterion(e1, e2, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f}")

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "embedding_model.pth"))
print("Embedding model saved.")
