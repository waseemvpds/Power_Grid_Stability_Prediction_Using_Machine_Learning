# ⚡ GridSense AI – Power Grid Stability Risk Prediction using Machine Learning

GridSense AI is a machine learning-powered web application that predicts the **stability score** of an electrical power grid using real-time system parameters. The application leverages an optimized **XGBoost Regressor** trained on the **UCI Electrical Grid Stability Simulated Dataset** to identify potentially unstable operating conditions and provide AI-driven recommendations for proactive grid monitoring.

---

## 📖 Project Overview

Maintaining the stability of an electrical power grid is essential for reliable electricity distribution. A disturbance in the balance between power generation and consumption can lead to instability, resulting in equipment failures or large-scale blackouts.

This project predicts the continuous **stability score (`stab`)** of the grid using machine learning. Based on the predicted score, the application classifies the system into different risk levels and provides actionable recommendations through an interactive Streamlit dashboard.

---

## 🎯 Project Objective

- Predict the electrical grid stability score using machine learning.
- Detect unstable operating conditions before failures occur.
- Compare multiple regression algorithms to identify the best-performing model.
- Optimize model performance using hyperparameter tuning.
- Deploy the trained model through an interactive Streamlit web application.

---

## 📊 Dataset

**Source:** UCI Machine Learning Repository

**Dataset:** Electrical Grid Stability Simulated Data

Each record represents a simulated operating condition of an electrical power grid.

### Features

#### ⏱ Time Constants (Reaction Time)

- tau1
- tau2
- tau3
- tau4

#### ⚡ Power Values

- p1
- p2
- p3
- p4

#### 🎛 Control Parameters

- g1
- g2
- g3
- g4

### Target Variable

**stab**

Continuous stability score

- Negative → Stable
- Positive → Unstable

---

## 🧠 Machine Learning Workflow

- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Correlation Analysis
- Model Comparison using 5-Fold Cross Validation
- Hyperparameter Tuning using GridSearchCV
- Final Model Training
- Model Evaluation
- Streamlit Deployment

---

## 🤖 Regression Models Evaluated

- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

The models were compared using **5-Fold Cross Validation**, and **XGBoost** achieved the best overall performance.

---

## ⚙ Hyperparameter Tuning

The XGBoost model was optimized using **GridSearchCV**.

### Best Parameters

- Learning Rate: **0.1**
- Max Depth: **7**
- Number of Estimators: **200**
- Subsample: **0.8**

---

## 📈 Model Performance

| Metric | Value |
|---------|-------|
| R² Score | **0.9485** |
| MAE | **0.0061** |
| RMSE | **0.0084** |

---

## 🌐 Streamlit Application Features

- ⚡ Real-time stability prediction
- 🎛 Interactive parameter sliders
- 🤖 AI-powered stability recommendations
- 📊 Enterprise analytics dashboard
- 📈 Feature importance visualization
- 🔥 Correlation heatmap
- 📉 Prediction vs Actual analysis
- 📜 Session prediction history
- 📄 Downloadable PDF prediction report
- 🌙 Dark & Light themes
- ⚙ Predefined operational scenarios
- 📌 Grid health and risk indicators

---

## 🛠 Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- XGBoost

### Data Analysis

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn
- Plotly

### Web Application

- Streamlit

### Report Generation

- ReportLab

---

## 📂 Project Structure

```
Power_Grid_Stability_Prediction/
│
├── app.py
├── Model.ipynb
├── model.pkl
├── requirements.txt
├── README.md
│
├── feature_importance.csv
├── correlation_matrix.csv
├── prediction_results.csv
├── dataset_summary.csv
├── model_metrics.json
│
└── Data_for_UCI_named.csv
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/waseemvpds/Power_Grid_Stability_Prediction_Using_Machine_Learning.git
```

Move into the project directory

```bash
cd Power_Grid_Stability_Prediction_Using_Machine_Learning
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ Run the Application

```bash
streamlit run app.py
```

---

## 📸 Screenshots

> Add screenshots of the following sections after uploading them to GitHub.

- Home Dashboard
- Prediction Interface
- Analytics Dashboard
- Feature Importance
- Correlation Heatmap
- PDF Report

---

## 💡 Future Improvements

- SHAP Explainability
- Model Monitoring Dashboard
- REST API Integration
- Cloud Deployment
- Live Grid Data Integration
- Time-Series Stability Forecasting

---

## 📚 References

- UCI Machine Learning Repository
- Scikit-learn Documentation
- XGBoost Documentation
- Streamlit Documentation

---

## 👨‍💻 Author

**Waseem VP**

- LinkedIn: https://www.linkedin.com/in/waseemvpds
- GitHub: https://github.com/waseemvpds

---