import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import torch

print("Step 1: Loading dataset...")

# Load dataset
df = pd.read_csv("data/creditcard.csv")

print("Dataset loaded!")
print(df.head())

print("\nStep 2: Checking fraud vs normal...")
print(df['Class'].value_counts())

print("\nStep 3: Balancing data...")

# Separate classes
df_normal = df[df.Class == 0]
df_fraud = df[df.Class == 1]

# Downsample normal
df_normal_down = resample(df_normal,
                         replace=False,
                         n_samples=len(df_fraud),
                         random_state=42)

# Combine
df_balanced = pd.concat([df_normal_down, df_fraud])

import matplotlib.pyplot as plt

print("\nCreating Fraud Visualization...")

normal = df_balanced[df_balanced['Class'] == 0]
fraud = df_balanced[df_balanced['Class'] == 1]

plt.figure()
plt.scatter(normal['Time'], normal['Amount'], color='blue', alpha=0.5)
plt.scatter(fraud['Time'], fraud['Amount'], color='red', alpha=0.9, marker='x')

plt.xlabel("Time")
plt.ylabel("Amount")
plt.title("Fraud vs Normal Transactions")
plt.legend(["Normal", "Fraud"])
plt.grid(True)

plt.savefig("fraud_visualization.png")

print("Balanced data:")
print(df_balanced['Class'].value_counts())

print("\nStep 4: Normalizing data...")

scaler = StandardScaler()
df_balanced['Amount'] = scaler.fit_transform(df_balanced[['Amount']])
df_balanced['Time'] = scaler.fit_transform(df_balanced[['Time']])

print("Normalization done!")

print("\nLoading SAME test data used in training...")

X_test = np.load("X_test.npy", allow_pickle=True)
y_test = np.load("y_test.npy", allow_pickle=True)

# Balance test data (same logic)
test_data = pd.DataFrame(X_test)
test_data['Class'] = y_test

fraud = test_data[test_data['Class'] == 1]
normal = test_data[test_data['Class'] == 0]

n_samples = min(len(normal), len(fraud) * 2)

normal_downsampled = resample(
    normal,
    replace=False,
    n_samples=n_samples,
    random_state=42
)

balanced_test = pd.concat([fraud, normal_downsampled])

X_test = balanced_test.drop("Class", axis=1)
y_test = balanced_test["Class"]

print("\nStep 5: Test data ready!")

# ------------------ MODEL ------------------

from models.hybrid_model import HybridModel

print("\nRunning Hybrid Model...")

input_dim = X_test.shape[1]
hidden_dim = 64   

hybrid = HybridModel(input_dim, hidden_dim)

# Load trained model
hybrid.load_state_dict(torch.load("best_model.pt"))
hybrid.eval()

# Convert to tensor
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

# Forward pass
output = hybrid(X_test_tensor)

y_true = y_test.values

print("Final Output Shape:", output.shape)

# ------------------ GRAPHS ------------------

import matplotlib.pyplot as plt

losses = np.load("losses.npy")
accuracies = np.load("accuracies.npy")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Training Loss")

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.title("Training Accuracy")

plt.tight_layout()
plt.savefig("combined_graphs.png")

# ------------------ METRICS ------------------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

print("\nEvaluating Model...")

preds = (torch.sigmoid(output) > 0.65).float().detach().numpy()

accuracy = accuracy_score(y_true, preds)
precision = precision_score(y_true, preds)
recall = recall_score(y_true, preds)
f1 = f1_score(y_true, preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ------------------ CONFUSION MATRIX ------------------

cm = confusion_matrix(y_true, preds)

tn, fp, fn, tp = cm.ravel()

print("\n--- Confusion Matrix Analysis ---")
print(f"True Positives (Fraud detected correctly): {tp}")
print(f"False Negatives (Fraud missed ❌): {fn}")
print(f"False Positives (False alarm ⚠️): {fp}")
print(f"True Negatives (Correct normal): {tn}")

print("\n--- Model Insights ---")

if fn > 0:
    print("⚠️ Model is missing some fraud cases (needs improvement)")
else:
    print("✅ No fraud cases missed")

if fp > 0:
    print("⚠️ Some normal transactions flagged as fraud")
else:
    print("✅ No false alarms")

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()