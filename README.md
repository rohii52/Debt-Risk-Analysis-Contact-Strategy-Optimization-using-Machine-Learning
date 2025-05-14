# Debt Risk Analysis & Contact Strategy Optimization using Machine Learning

This project is a complete end-to-end case study on optimizing debt recovery using predictive analytics, risk segmentation, and machine learning. It was developed as part of a data analyst case challenge and aims to simulate how a data-driven debt collection strategy can improve contact efficiency and reduce operational costs.

---

##  Project Summary

- **Clients Analyzed**: 1000+
- **Goal**: Predict repayment risk and improve client contact strategies
- **Data**: Synthetic client data with arrears, contact info, and bureau scores

---

##  Key Steps

### 1. Data Cleaning & Preprocessing
- Removed duplicates, handled missing values
- Validated residency using `dp1` and `dp3` scores
- Created custom `residency_zone` and mobile/email confirmation flags

### 2. Segmentation
- Created **Segment Risk Scores** using weighted averages of:
  - Arrears Balance
  - Data Partner Scores (dp1, dp3)
  - Contact availability
- Categorized clients into 3 Action Tiers (Low, Medium, High Risk)

### 3. Machine Learning Models
- **XGBoost Classifier** to predict Risk Tier (98.2% Accuracy)
- **Logistic Regression vs DNN** to predict mobile availability
- **Isolation Forest** for anomaly detection
- **AutoEncoder** to generate client embeddings for unsupervised clustering

### 4. Unsupervised Insights
- Used **KMeans** to cluster similar clients
- Compared clustering groups to manually created tiers

### 5. Strategy Simulation
- Designed a rule-based simulator to test contact strategies
- Used ML outputs and tier changes to simulate contact outcomes

---

## 🤖 Project Yellow (LLM Assistant)

Built a Flask + LangChain-based assistant that:
- Understands company and project context
- Answers queries using custom memory, prompt injection, and vector DB
- Responds to internal team members or external users differently
- Please email me for the whole pipeline and source code of this AI Assistant

---

## Tech Stack

- Python, Pandas, Scikit-learn, XGBoost, Keras
- LangChain + Ollama for LLM
- Flask for local assistant
- Matplotlib / Seaborn for insights

---

## 📁 Folders

```
📦 Python Notebooks
 ┣ 📄 test_1.ipynb               ← Cleaning + Scoring + Zones
 ┣ 📄 unique_segment_scores.ipynb← Segment Score Logic
 ┣ 📄 risk_prediction_xgboost.ipynb
 ┣ 📄 Client_Embeddings.ipynb    ← AutoEncoder Embeddings
 ┣ 📄 RL_Simulator.ipynb         ← Strategy simulation
 ┣ 📄 Clustering_kmeans.ipynb
 ┣ 📄 Anomaly_Detection.ipynb
 ┗ 📄 deep_mobile_prediction_model.ipynb

📦 Project Yellow
 ┣ 📄 Project Yellow.mp4         ← Demo Video

```

---

## Insights Delivered

- Clients with lowest arrears and highest mobile % are in Tier 1
- KMeans clustering revealed hidden segments not captured by static logic
- ML-based tiering matched 98% of manual logic but uncovered 2% edge cases
- Contact strategy improvements simulated +15-20% boost in efficiency

---

## Future Work

- Add cost data for ROI modeling.
- Incorporate time series for repayment predictions.
- Extend "Project Yellow" to proceed with data analysis & build strategies dynamically based on data / CSV uploads.

---

## Acknowledgement

This project was developed for a real-world case study simulation to demonstrate hybrid capabilities in data analysis, machine learning, and business logic deployment.

📧 Contact: rohithgofficial@gmail.com  
🔗 GitHub: [github.com/rohii52](https://github.com/rohii52)  
💼 LinkedIn: [linkedin.com/in/rohii52](https://linkedin.com/in/rohii52)  

