# Sentiment Analysis Using Custom LSTM, Pretrained AWD-LSTM (ULMFiT), and BERT

## 📘 Project Overview
This project presents a comprehensive sentiment analysis framework using deep learning techniques on the IMDb movie reviews dataset. It explores and compares three major approaches:

1. A **Custom LSTM model built from scratch using PyTorch**
2. A **Pretrained AWD-LSTM model fine-tuned using the ULMFiT methodology**
3. A **Transformer-based BERT model fine-tuned for sentiment classification**

The goal is to analyze how transfer learning and modern transformer architectures improve performance, convergence speed, and generalization compared to a traditional LSTM model.

---

## 🎯 Objectives
- Load and preprocess the IMDb movie reviews dataset
- Perform tokenization and padding of text data
- Build and train a custom LSTM sentiment classifier
- Fine-tune a pretrained AWD-LSTM using ULMFiT
- Fine-tune a pretrained BERT-base-uncased model
- Evaluate all models using identical metrics
- Perform a comparative analysis of model performance

---

## 🧠 Models Implemented

### 🔹 Project 1: LSTM-Based Models
- **Custom LSTM (PyTorch)**  
  - Learns task-specific representations  
  - Requires more epochs to converge  

- **Pretrained AWD-LSTM (ULMFiT)**  
  - Uses transfer learning  
  - Faster convergence  
  - Better generalization  

### 🔹 Project 2: Transformer-Based Model
- **BERT (bert-base-uncased)**  
  - WordPiece tokenization  
  - Bidirectional contextual embeddings  
  - State-of-the-art performance  

---

## 📊 Evaluation Metrics
All models are evaluated using the same metrics for controlled comparison:
- Accuracy
- Precision
- Recall
- F1-score
- Training loss
- Validation loss
- Epochs to convergence

---

## 📈 Key Findings
- The **Custom LSTM** learns meaningful patterns but converges slowly.
- The **Pretrained AWD-LSTM** achieves higher accuracy with fewer epochs.
- **Transfer learning significantly improves performance** on limited labeled data.
- **BERT outperforms LSTM-based models** in robustness and generalization but has higher computational cost.

---

## 🗂 Project Structure
