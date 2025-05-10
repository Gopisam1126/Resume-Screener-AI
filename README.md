## Resume Classification Pipeline

This repository contains a complete pipeline for automated resume classification using TF-IDF vectorization and an SVM model. You can train on your own `resume_data.csv`, evaluate multiple classifiers via grid search, and then export a production-ready SVM model for inference.

---

### Features

- **Data balancing** by oversampling minority classes  
- **Text cleaning** (URL, punctuation, non-ASCII removal, stop-word filtering)  
- **TF-IDF** feature extraction  
- **Model selection** via 5-fold `GridSearchCV` over Random Forest, SVM, and Logistic Regression  
- **Final export** of an SVM pipeline (`re_sc_pipeline.pkl`) containing:
  - `model` (trained `SVC`)
  - `vectorizer` (`TfidfVectorizer`)
  - `label_encoder` (`LabelEncoder`)

---
