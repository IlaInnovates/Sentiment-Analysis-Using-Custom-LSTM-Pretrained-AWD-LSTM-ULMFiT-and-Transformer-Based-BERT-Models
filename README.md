# Sentiment-Analysis-Using-Custom-LSTM-Pretrained-AWD-LSTM-ULMFiT-and-Transformer-Based-BERT-Models
## 🎬 Sentiment Analysis using Custom LSTM, ULMFiT, and BERT

## 📖 Project Overview

This project implements and compares three deep learning architectures for Sentiment Analysis on the IMDb Movie Reviews dataset:

1. Custom LSTM (trained from scratch)
2. Pretrained AWD-LSTM (ULMFiT)
3. Transformer-based BERT

The objective is to evaluate how transfer learning and transformer architectures improve classification performance compared to traditional recurrent neural networks.

---

## 🎯 Problem Statement

Design and implement a sentiment analysis system that:

- Classifies movie reviews as Positive or Negative
- Compares performance of:
  - Custom LSTM
  - ULMFiT (AWD-LSTM)
  - BERT
- Evaluates models using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Convergence speed

---

## 📂 Dataset

**Dataset:** IMDb Movie Reviews  
**Source:** Stanford AI Lab / HuggingFace  
**Size:** 50,000 labeled reviews  
- 25,000 Training samples  
- 25,000 Testing samples  

Each review contains:
- Text data
- Binary label (0 = Negative, 1 = Positive)

---

## 🛠️ Technologies Used

- Python
- PyTorch
- FastAI
- HuggingFace Transformers
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

---

## 🏗️ Project Architecture

### 1️⃣ Custom LSTM
- Embedding Layer
- LSTM Layer
- Fully Connected Layer
- Trained from scratch

### 2️⃣ ULMFiT (AWD-LSTM)
- Pretrained Language Model
- Fine-tuned using FastAI
- Transfer learning based

### 3️⃣ BERT
- Transformer-based model
- Bidirectional Encoder
- Fine-tuned for sequence classification

---

## 🔄 Project Workflow

1. Load IMDb dataset
2. Preprocess text data
3. Tokenization & Padding
4. Train Custom LSTM
5. Fine-tune ULMFiT
6. Fine-tune BERT
7. Evaluate all models
8. Compare performance
9. Select best model

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Training loss
- Validation loss
- Epochs to convergence

---

## 📈 Final Results

| Model          | Accuracy | F1 Score |
|---------------|----------|----------|
| Custom LSTM  | ~85%     | ~0.85    |
| ULMFiT       | ~92%     | ~0.92    |
| BERT         | **~96%** | **~0.96**|

🏆 **Best Model: BERT**

---

## 🔍 Key Insights

- Custom LSTM learns task-specific features but converges slowly.
- ULMFiT significantly improves performance through transfer learning.
- BERT achieves highest accuracy due to:
  - Bidirectional context understanding
  - Self-attention mechanism
  - Large-scale pretraining

However, BERT has higher computational cost compared to LSTM-based models.

---

## 🚀 How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/sentiment-analysis-lstm-bert.git
cd sentiment-analysis-lstm-bert
