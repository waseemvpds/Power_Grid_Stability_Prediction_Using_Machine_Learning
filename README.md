# ⚡ Power Grid Stability Prediction

A Machine Learning-based web application that predicts whether an electrical power grid is **stable or unstable** based on system parameters.

---

## 📌 Overview

Power grid stability is critical for ensuring reliable electricity distribution.  
This project uses a trained **XGBoost regression model** to analyze system parameters and determine stability.

The model predicts a **stability score**:
- Negative → Stable ✅  
- Positive → Unstable ⚠️  

---

## 🎯 Features

- ⚡ Predict grid stability in real-time
- 🎚 Interactive sliders for input parameters
- 📊 Clear result display (Stable / Unstable)
- 🔢 Shows stability score
- 🖼 Clean and intuitive UI with visual representation
- 🌐 Deployable using Streamlit Cloud

---

## 🧠 Input Parameters

### ⏱️ Time Constants (τ1–τ4)
- Represents how quickly each node responds to changes

### ⚡ Power Injection (p1–p4)
- Positive → Power generation  
- Negative → Power consumption  

### 🎛️ Control Gain (g1–g4)
- Determines system response strength to imbalance  

---

## 🧪 Model Details

- Algorithm: **XGBoost Regressor**
- Dataset: Electrical Grid Stability (UCI)
- Evaluation Metrics:
  - R² Score
  - MAE
  - RMSE

---

## 🖥️ Tech Stack

- Python 🐍  
- Streamlit 🌐  
- Scikit-learn 🤖  
- XGBoost ⚡  
- NumPy  

---
## 🚀 How to Run Locally


1. Clone the repository:

git clone https://github.com/waseemvpds/Power_Grid_Stability_Prediction_Using_Machine_Learning.git
cd Power_Grid_Stability_Prediction_Using_Machine_Learning


## Install dependencies:

pip install -r requirements.txt

---

## Run the app:

streamlit run Streamli.py

---