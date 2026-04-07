import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models.hybrid_model import HybridModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.title(" Fraud Detection using Hybrid Model")

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/creditcard.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.write(df.head())

# ---------------- VISUALIZATION ----------------

st.subheader(" Fraud vs Normal Visualization")

normal = df[df['Class'] == 0]
fraud = df[df['Class'] == 1]

fig, ax = plt.subplots()
ax.scatter(normal['Time'], normal['Amount'], color='blue', alpha=0.3)
ax.scatter(fraud['Time'], fraud['Amount'], color='red', alpha=0.8, marker='x')

ax.set_xlabel("Time")
ax.set_ylabel("Amount")
ax.set_title("Fraud vs Normal Transactions")

st.pyplot(fig)

# ---------------- LOAD TRAINED DATA ----------------

st.subheader(" Model Evaluation")

X_test = np.load("X_test.npy", allow_pickle=True)
y_test = np.load("y_test.npy", allow_pickle=True)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# ---------------- LOAD MODEL ----------------

input_dim = X_test.shape[1]
hidden_dim = 64   # MUST match training

model = HybridModel(input_dim, hidden_dim)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# ---------------- PREDICTIONS ----------------

output = model(X_test_tensor)
preds = (torch.sigmoid(output) > 0.65).float().detach().numpy()

y_true = y_test

# ---------------- METRICS ----------------

accuracy = accuracy_score(y_true, preds)
precision = precision_score(y_true, preds)
recall = recall_score(y_true, preds)
f1 = f1_score(y_true, preds)

st.subheader(" Model Metrics")

st.write(f"Accuracy: {accuracy:.4f}")
st.write(f"Precision: {precision:.4f}")
st.write(f"Recall: {recall:.4f}")
st.write(f"F1 Score: {f1:.4f}")

# ---------------- CONFUSION MATRIX ----------------

st.subheader(" Confusion Matrix Analysis")

cm = confusion_matrix(y_true, preds)
tn, fp, fn, tp = cm.ravel()

st.write(f" True Positives: {tp}")
st.write(f" False Negatives: {fn}")
st.write(f" False Positives: {fp}")
st.write(f" True Negatives: {tn}")

# ---------------- INSIGHTS ----------------

st.subheader(" Insights")

if fn > 0:
    st.warning("Model is missing some fraud cases")
else:
    st.success("No fraud cases missed")

if fp > 0:
    st.warning("Some normal transactions flagged as fraud")
else:
    st.success("No false alarms")

# ---------------- REAL-TIME PREDICTION ----------------

st.subheader(" Real-Time Prediction")

index = st.number_input("Enter transaction index", min_value=0, max_value=len(X_test)-1, value=10)

if st.button("Predict"):
    pred = preds[index][0]

    st.subheader("Prediction Result")

    if pred == 1:
        st.error(" Fraud Transaction Detected")
    else:
        st.success(" Normal Transaction")

    st.write(f"Transaction Index: {index}")

# ---------------- SHOW GRAPHS ----------------

st.subheader(" Model Performance")

try:
    st.image("combined_graphs.png", caption="Training Graphs")
except:
    st.warning("Run training to generate graphs")

try:
    st.image("confusion_matrix.png", caption="Confusion Matrix")
except:
    st.warning("Run evaluation to generate confusion matrix")