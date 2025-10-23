# 🧠 NLP Assignment 1 — Sexism Detection in Tweets

**Course:** Natural Language Processing (A.Y. 2025–2026)  
**University:** University of Bologna  
**Authors:** [Your Names Here]  
**Prof:** Paolo Torroni  
**TAs:** Federico Ruggeri, Eleonora Mancini  

---

## 📌 Overview
This project addresses the **EXIST 2023 Task 2** on sexism detection in tweets.  
The goal is to classify each tweet according to the *intention of the author* into one of the following categories:

- `DIRECT` — explicitly sexist messages or those that incite sexism  
- `REPORTED` — reports of sexist situations experienced by women  
- `JUDGEMENTAL` — tweets judging or condemning sexist behavior  
- `-` — non-sexist messages  

The task is formulated as a **multi-class classification problem** on noisy, informal, multilingual social media data.

---

## 🧩 Tasks Summary

### **Task 1 — Corpus**
- Load the dataset (train, validation, test) from JSON files.  
- Apply **majority voting** on Task 2 labels to determine the final label.  
- Filter for English tweets only and keep the following columns:
  - `id_EXIST`, `lang`, `tweet`, `label`
- Encode labels as:
  {
'-': 0,
'DIRECT': 1,
'JUDGEMENTAL': 2,
'REPORTED': 3
} 

---

### **Task 2 — Data Cleaning**
Preprocess tweets to remove:
- Emojis  
- Hashtags (`#example`)  
- Mentions (`@user`)  
- URLs  
- Special and quote characters  

Then perform **lemmatization** to normalize words.

---

### **Task 3 — Text Encoding**
- Use **GloVe embeddings** to represent words.  
- Build a vocabulary from training tokens.  
- Handle **OOV tokens**:
- If OOV in train → add to vocabulary with random/custom embedding  
- If OOV in val/test → assign `<UNK>` token with static embedding  

---

### **Task 4 — Model Definition**
Implement two RNN-based classifiers:
1. **Baseline model:** Bidirectional LSTM + Dense layer  
2. **Stacked model:** Two Bidirectional LSTM layers + Dense layer  

Input can be handled either as precomputed embeddings or using a **trainable Embedding layer** initialized with the embedding matrix.

---

### **Task 5 — Training & Evaluation**
- Train each model using at least **3 random seeds**.  
- Evaluate on the validation set using:
- **Macro F1-score**
- **Precision**
- **Recall**
- Report **average ± standard deviation** across seeds.  
- Select the best model based on F1-score.

---

### **Task 6 — Transformer Model**
Use **Twitter-roBERTa-base for Hate Speech Detection** from Hugging Face.  
Steps:
1. Load tokenizer and model  
2. Tokenize dataset  
3. Fine-tune using the `Trainer` API  
4. Evaluate with the same metrics as LSTM models  

Model: [cardiffnlp/twitter-roberta-base-hate](https://huggingface.co/cardiffnlp/twitter-roberta-base-hate)

---

### **Task 7 — Error Analysis**
Perform a short error analysis including:
- Confusion matrix  
- Common misclassifications  
- Observed issues (e.g., OOV words, data imbalance)  
- Suggestions for improvement  

---

### **Task 8 — Report**
Summarize your pipeline and results in a short report (max 2 pages) following the NLP course template:
- Describe methods, results, and insights
- Include tables for metrics and figures for learning curves
- Avoid raw code or screenshots

---

## 🧪 Expected Results
Typical F1-score range: **30–40** (given task complexity)  
Leaderboard reference: **40–50** using hierarchical methods.

---

## 🛠️ Technologies
- **Python**
- **Pandas**, **NumPy**, **scikit-learn**
- **TensorFlow / PyTorch**
- **Hugging Face Transformers**
- **NLTK / spaCy** for preprocessing and lemmatization

---

## 📚 References
- [EXIST 2023 Shared Task](https://clef2023.clef-initiative.eu/index.php?page=Pages/labs.html#EXIST)
- [Twitter-roBERTa-base-hate](https://huggingface.co/cardiffnlp/twitter-roberta-base-hate)
- [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)

---

## 📄 License
This repository is intended for **educational use** as part of the University of Bologna NLP course.
