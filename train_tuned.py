import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from models.hybrid_model import HybridModel

# -------------------------------
# STEP 1: LOAD DATA
# -------------------------------
df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1).values

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

y = df["Class"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
import numpy as np

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

from sklearn.utils import resample
import pandas as pd

# Convert to dataframe
data = pd.DataFrame(X_train.numpy())
data['Class'] = y_train.numpy()

# Separate fraud and normal
fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]

# Downsample normal (IMPORTANT)
normal_downsampled = resample(
    normal,
    replace=False,
    n_samples=len(fraud) * 2,
    random_state=42
)

# Combine
balanced = pd.concat([fraud, normal_downsampled])

# Convert back to tensors
X_train = torch.tensor(balanced.drop("Class", axis=1).values, dtype=torch.float32)
y_train = torch.tensor(balanced["Class"].values, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# -------------------------------
# STEP 2: HYPERPARAMETERS (REDUCED FOR STABILITY)
# -------------------------------
learning_rates = [0.001]
batch_sizes = [32]
hidden_dims = [64]

# -------------------------------
# STEP 3: CLASS IMBALANCE FIX
# -------------------------------
fraud_count = sum(y_train)
normal_count = len(y_train) - fraud_count

fraud_weight = normal_count / fraud_count
fraud_weight = min(fraud_weight,50)  # cap to avoid instability

criterion = torch.nn.BCEWithLogitsLoss()

# Track best model using F1-score
best_f1 = 0
best_config = None

# -------------------------------
# STEP 4: TUNING LOOP
# -------------------------------
for lr in learning_rates:
    for batch_size in batch_sizes:
        for hidden_dim in hidden_dims:

            print("\n==============================")
            print(f"Trying: lr={lr}, batch={batch_size}, hidden={hidden_dim}")

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            model = HybridModel(input_dim=X_train.shape[1], hidden_dim=hidden_dim)
            epoch_losses = []
            epoch_accuracies = []
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # -------------------------------
            # TRAINING
            # -------------------------------
            for epoch in range(12):
                model.train()
                total_loss = 0

                all_preds_epoch = []
                all_labels_epoch = []

                for inputs, labels in train_loader:
                  outputs = model(inputs)

                  loss = criterion(outputs, labels)

                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()

                  total_loss += loss.item()

                  preds = (torch.sigmoid(outputs) > 0.5).float()

                  all_preds_epoch.extend(preds.detach().cpu().numpy())
                  all_labels_epoch.extend(labels.detach().cpu().numpy())

                avg_loss = total_loss / len(train_loader)

                acc = accuracy_score(all_labels_epoch, all_preds_epoch)

                epoch_losses.append(avg_loss)
                epoch_accuracies.append(acc)

                print(f"Epoch {epoch} Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")

            # -------------------------------
            # EVALUATION
            # -------------------------------
            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)

                    # adjusted threshold (IMPORTANT)
                    preds = torch.sigmoid(outputs) > 0.65

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)

            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            # -------------------------------
            # SAVE BEST MODEL
            # -------------------------------
            if f1 > best_f1:
               best_f1 = f1
               best_config = (lr, batch_size, hidden_dim)

            torch.save(model.state_dict(), "best_model.pt")

            # Save graph data
            import numpy as np

            np.save("losses.npy", epoch_losses)
            np.save("accuracies.npy", epoch_accuracies)

            # -------- GRAPHS --------
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(epoch_losses)
            plt.title("Loss Curve")

            plt.subplot(1, 2, 2)
            plt.plot(epoch_accuracies)
            plt.title("Accuracy Curve")

            plt.tight_layout()
            plt.show()

# -------------------------------
# FINAL RESULT
# -------------------------------
print("\n==============================")
print("BEST CONFIG:", best_config)
print("BEST F1 SCORE:", best_f1)