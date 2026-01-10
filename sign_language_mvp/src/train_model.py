import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ======================
# PATHS
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "fixed")

# ======================
# LOAD DATA
# ======================
X = []
y = []

gestures = os.listdir(DATA_DIR)

for gesture in gestures:
    gesture_dir = os.path.join(DATA_DIR, gesture)
    for file in os.listdir(gesture_dir):
        if file.endswith(".npy"):
            path = os.path.join(gesture_dir, file)
            X.append(np.load(path))
            y.append(gesture)

X = np.array(X)                # (N, 30, 21, 2)
X = X.reshape(X.shape[0], -1)  # flatten

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

print("Classes:", le.classes_)
print("Total samples:", len(X))

# ======================
# TRAIN / TEST SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# DATASET
# ======================
class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = GestureDataset(X_train, y_train)
test_ds = GestureDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=8)

# ======================
# MODEL
# ======================
class GestureNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = GestureNet(X.shape[1], len(le.classes_))

# ======================
# TRAINING SETUP
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ======================
# TRAIN LOOP
# ======================
for epoch in range(30):
    model.train()
    correct = total = 0
    loss_sum = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1:02d} | Loss: {loss_sum:.3f} | Train Acc: {acc:.2f}%")

# ======================
# TEST
# ======================
model.eval()
correct = total = 0

with torch.no_grad():
    for xb, yb in test_loader:
        out = model(xb)
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

print(f"\nTest Accuracy: {100 * correct / total:.2f}%")

# ======================
# SAVE MODEL
# ======================
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "gesture_model.pth"))
np.save(os.path.join(MODEL_DIR, "labels.npy"), le.classes_)

print("Model saved successfully.")
