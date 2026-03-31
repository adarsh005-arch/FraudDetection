from utils.preprocess import y_train_tensor
import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
import random

from utils.graph_builder import build_graph
from models.gnn_model import GNNModel
from models.transformer_model import TransformerModel
from models.hybrid_model import HybridModel

st.title(" Fraud Detection using GNN + Transformer")
# Import required data from preprocess
from utils.preprocess import X_train_tensor, y_train_tensor
from utils.graph_builder import build_graph

# Build graph data
graph_data = build_graph(X_train_tensor)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/creditcard.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.write(df.head())

# Visualization
st.subheader("📊 Fraud vs Normal Visualization")

normal = df[df['Class'] == 0]
fraud = df[df['Class'] == 1]

fig, ax = plt.subplots()
ax.scatter(normal['Time'], normal['Amount'], color='blue', label='Normal', alpha=0.3)
ax.scatter(fraud['Time'], fraud['Amount'], color='red', label='Fraud', alpha=0.8, marker='x')

ax.set_xlabel("Time")
ax.set_ylabel("Amount")
ax.set_title("Fraud vs Normal Transactions")
ax.legend()

st.pyplot(fig)

# Load processed graph (you already built)
st.subheader("🧠 Model Prediction")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# Build model outputs
input_dim = graph_data.x.shape[1]

gnn = GNNModel(input_dim)
transformer = TransformerModel(input_dim)
hybrid = HybridModel()

gnn_out = gnn(graph_data)
transformer_out = transformer(graph_data.x)

output = hybrid(gnn_out, transformer_out)

st.subheader("📊 Model Metrics")

# Predictions
preds = torch.argmax(output, dim=1).detach().numpy()
from utils.preprocess import y_train_tensor

y_true = y_train_tensor.numpy()

accuracy = accuracy_score(y_true, preds)
precision = precision_score(y_true, preds)
recall = recall_score(y_true, preds)
f1 = f1_score(y_true, preds)

st.write(f"Accuracy: {accuracy:.4f}")
st.write(f"Precision: {precision:.4f}")
st.write(f"Recall: {recall:.4f}")
st.write(f"F1 Score: {f1:.4f}")

# Dummy graph creation (reuse your pipeline)
from utils.preprocess import graph_data

input_dim = graph_data.x.shape[1]

gnn = GNNModel(input_dim)
transformer = TransformerModel(input_dim)
hybrid = HybridModel()

gnn_out = gnn(graph_data)
trans_out = transformer(graph_data.x)
output = hybrid(gnn_out, trans_out)

# Real-time prediction
st.subheader("⚡ Real-Time Prediction")

index = st.number_input("Enter transaction index", min_value=0, max_value=len(graph_data.x)-1, value=10)

if st.button("Predict"):
    prediction = torch.argmax(output[index]).item()

    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.error("🚨 Fraud Transaction Detected")
    else:
        st.success("✅ Normal Transaction")

    st.write(f"Transaction Index: {index}")

st.subheader("🧠 Confusion Matrix Analysis")

cm = confusion_matrix(y_true, preds)
tn, fp, fn, tp = cm.ravel()

st.write(f"✅ True Positives (Fraud detected): {tp}")
st.write(f"❌ False Negatives (Missed fraud): {fn}")
st.write(f"⚠️ False Positives (False alarms): {fp}")
st.write(f"✔ True Negatives (Correct normal): {tn}")

st.subheader("📌 Insights")

if fn > 0:
    st.warning("⚠️ Model is missing some fraud cases")
else:
    st.success("✅ No fraud cases missed")

if fp > 0:
    st.warning("⚠️ Some normal transactions flagged as fraud")
else:
    st.success("✅ No false alarms")

# Show graphs
st.subheader("📈 Model Performance")

try:
    st.image("combined_graphs.png", caption="Loss & Accuracy")
except:
    st.warning("Run training to generate graphs")

try:
    st.image("confusion_matrix.png", caption="Confusion Matrix")
except:
    st.warning("Run evaluation to generate confusion matrix")