# Fraud Detection using Hybrid of GNN and Transformer Model

## 📌 Overview

This project focuses on detecting fraudulent financial transactions using a hybrid deep learning approach. The system combines structured feature learning with a neural network architecture to handle highly imbalanced data effectively. The model is trained, tuned, and deployed with a complete pipeline, including preprocessing, evaluation, and a Streamlit-based user interface for real-time predictions.

---

## 🚀 Key Results

* **Accuracy:** ~0.93
* **Precision:** ~0.97
* **Recall:** ~0.90
* **F1 Score:** ~0.93

The model achieves a strong balance between detecting fraud (high recall) and minimizing false alarms (high precision).

---

## 🧠 Model Architecture

* Hybrid Neural Network (inspired by GNN + Transformer concepts)
* Fully connected layers with feature fusion
* Dropout for regularization
* Sigmoid activation with threshold tuning (0.65)

---

## ⚙️ Features

* Data preprocessing and normalization pipeline
* Handling of imbalanced datasets using resampling
* Hyperparameter tuning for optimal performance
* Confusion matrix analysis with insights
* Training loss and accuracy visualization
* Real-time fraud prediction using Streamlit

---

## 📊 Visual Outputs

* Fraud vs Normal transaction scatter plot
* Training loss and accuracy graphs
* Confusion matrix heatmap

---

## 💻 Tech Stack

* **Programming Language:** Python
* **Libraries:** PyTorch, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
* **Frontend:** Streamlit
* **Version Control:** Git & GitHub

---

## 📁 Project Structure

```
FraudDetection/
│
├── models/                # Model architectures
├── utils/                 # Preprocessing & helper scripts
├── data/                  # Dataset (not included)
├── app.py                 # Streamlit application
├── train_tuned.py         # Model training & tuning
├── best_model.pt          # Trained model
├── losses.npy             # Training loss data
├── accuracies.npy         # Training accuracy data
├── combined_graphs.png    # Performance graphs
├── confusion_matrix.png   # Confusion matrix
├── README.md              # Project documentation
```

---

## 📥 Dataset

The dataset is not included due to GitHub size limitations.

Download it from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place it in:

```
data/creditcard.csv
```

---

## ▶️ How to Run the Project

### 1. Install Dependencies

```
pip install -r requirements.txt
```

### 2. Train the Model

```
python train_tuned.py
```

### 3. Run Preprocessing & Evaluation

```
python -m utils.preprocess
```

### 4. Launch Streamlit App

```
streamlit run app.py
```

---

## ⚡ Real-Time Prediction

The Streamlit interface allows users to:

* View dataset insights
* Analyze model performance
* Input a transaction index
* Get instant fraud prediction results

---

## 📈 Model Insights

* The model prioritizes **fraud detection (high recall)** while maintaining **low false positives**
* Threshold tuning plays a key role in balancing precision and recall
* The system is robust against class imbalance

---

## 🎯 Key Learnings

* Handling imbalanced datasets effectively
* Importance of evaluation metrics like F1-score over accuracy
* Model tuning and threshold optimization
* Building end-to-end ML pipelines with deployment

---

## 📌 Future Improvements

* Integration with real-time transaction APIs
* Advanced anomaly detection techniques
* Model explainability (SHAP/LIME)
* Deployment on cloud platforms

---

## 👤 Author

**Adarsh M**
BTech Computer Science (AI/ML)

---

## 📜 License

This project is for educational and research purposes.
