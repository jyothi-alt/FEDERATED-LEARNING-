import flwr as fl
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


if len(sys.argv) != 2:
    print("Usage: python client.py <client_id>")
    sys.exit(1)

try:
    client_id = int(sys.argv[1])
except ValueError:
    print("Client ID must be an integer.")
    sys.exit(1)

data_path = os.path.join("data", "splits", f"client_{client_id}.csv")
if not os.path.exists(data_path):
    print(f"Dataset not found: {data_path}")
    sys.exit(1)


df = pd.read_csv(data_path)
if "RiskLevel" not in df.columns:
    print("Error: 'RiskLevel' column not found.")
    sys.exit(1)


le = LabelEncoder()
df["RiskLevel"] = le.fit_transform(df["RiskLevel"].astype(str))

X = df.drop(columns=["RiskLevel"]).astype(float).values
y = df["RiskLevel"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y if len(np.unique(y)) > 1 else None
)

print(f"Client {client_id} loaded {len(df)} samples "
      f"({len(X_train)} train / {len(X_test)} test)")


X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32)

device = "cuda" if torch.cuda.is_available() else "cpu"


class_counts = np.bincount(y_train_t.numpy())
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
class_weights = class_weights.to(device)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=None):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

n_features = X_train.shape[1]
n_classes = len(np.unique(y_train))
model = MLP(n_features, hidden_dim=128, output_dim=n_classes).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
LOCAL_EPOCHS = 10


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = model.state_dict()
        for i, key in enumerate(state_dict.keys()):
            state_dict[key] = torch.tensor(parameters[i])
        model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.train()
        for epoch in range(LOCAL_EPOCHS):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                preds = torch.argmax(model(X_train_t.to(device)), dim=1)
                train_acc = (preds == y_train_t.to(device)).float().mean().item()
            print(f"Client {client_id} - Epoch {epoch+1}/{LOCAL_EPOCHS} - Train Acc: {train_acc:.4f}")
        return self.get_parameters(), len(train_loader.dataset), {"train_accuracy": train_acc}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                total_loss += loss.item() * xb.size(0)
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == yb).sum().item()
                total_samples += yb.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        print(f"Client {client_id} Eval - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
        print(classification_report(all_labels, all_preds, zero_division=0))
        return float(avg_loss), total_samples, {"accuracy": accuracy}

if __name__ == "__main__":
    print(f"Client {client_id} connecting to server at 127.0.0.1:8080 ...")
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client()
    )
