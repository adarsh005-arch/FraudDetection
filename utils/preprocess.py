import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

plt.figure()

# Separate fraud and normal
normal = df_balanced[df_balanced['Class'] == 0]
fraud = df_balanced[df_balanced['Class'] == 1]

# Plot normal transactions
plt.scatter(normal['Time'], normal['Amount'], 
            color='blue', label='Normal', alpha=0.5)

# Plot fraud transactions (highlighted)
plt.scatter(fraud['Time'], fraud['Amount'], 
            color='red', label='Fraud', alpha=0.9, marker='x')

plt.xlabel("Time")
plt.ylabel("Amount")
plt.title("Fraud vs Normal Transactions")

plt.legend()     # shows labels
plt.grid(True)   # adds grid for clarity

plt.savefig("fraud_visualization.png")
#plt.show()

print("Balanced data:")
print(df_balanced['Class'].value_counts())

print("\nStep 4: Normalizing data...")

scaler = StandardScaler()
df_balanced['Amount'] = scaler.fit_transform(df_balanced[['Amount']])
df_balanced['Time'] = scaler.fit_transform(df_balanced[['Time']])

print("Normalization done!")

print("\nStep 5: Splitting data...")

X = df_balanced.drop('Class', axis=1)
y = df_balanced['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Split done!")

print("\nStep 6: Sorting for Transformer...")
X_train = X_train.sort_values(by='Time')

print("Sorting done!")

print("\nStep 7: Converting to tensor...")
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)

print("Tensor shape:", X_train_tensor.shape)

print("\n✅ PREPROCESSING COMPLETED SUCCESSFULLY!")

from utils.graph_builder import build_graph

graph_data = build_graph(X_train_tensor)

print(graph_data)

print("Running GNN now...")
from models.gnn_model import GNNModel

# Initialize model
input_dim = graph_data.x.shape[1]
gnn = GNNModel(input_dim)

# Run model
gnn_output = gnn(graph_data)

print("GNN Output Shape:", gnn_output.shape)

#transformer
from models.transformer_model import TransformerModel

print("Running Transformer now...")

transformer = TransformerModel(input_dim)
transformer_output = transformer(graph_data.x)

print("Transformer Output Shape:", transformer_output.shape)

#Hybrid Model
from models.hybrid_model import HybridModel

print("Running Hybrid Model...")

hybrid = HybridModel()

output = hybrid(gnn_output, transformer_output)

print("Final Output Shape:", output.shape)

#training
import torch.nn as nn
import torch.optim as optim

# Labels
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

# Model
hybrid = HybridModel()

# Loss function
class_weights = torch.tensor([1.0, 3.0])  # fraud more important
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(hybrid.parameters(), lr=0.0005)
epochs = 30
losses = []
accuracies = []
for epoch in range(epochs):
    optimizer.zero_grad()

    gnn_out = gnn(graph_data)
    transformer_out = transformer(graph_data.x)

    output = hybrid(gnn_out, transformer_out)

    loss = criterion(output, y_train_tensor)

    loss.backward()
    optimizer.step()
    preds = torch.argmax(output, dim=1)
    acc = (preds == y_train_tensor).float().mean().item()
    accuracies.append(acc)
    losses.append(loss.item())

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

# Loss graph
plt.subplot(2, 2, 1)
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Accuracy graph
plt.subplot(2, 2, 2)
plt.plot(accuracies)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()

plt.savefig("combined_graphs.png")
#plt.show()

#accuracy
from sklearn.metrics import accuracy_score

print("\nEvaluating Model...")

# Predictions
preds = torch.argmax(output, dim=1)

accuracy = accuracy_score(y_train_tensor.numpy(), preds.detach().numpy())

print("Accuracy:", accuracy)

#confusion Matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("\nEvaluating Model...")

# Predictions
preds = torch.argmax(output, dim=1).detach().numpy()
y_true = y_train_tensor.numpy()

# Metrics
accuracy = accuracy_score(y_true, preds)
precision = precision_score(y_true, preds)
recall = recall_score(y_true, preds)
f1 = f1_score(y_true, preds)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion Matrix
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

plt.savefig("confusion_matrix.png")   # save for GitHub
plt.show()