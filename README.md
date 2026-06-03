# 🚀 Auto Feature Engineering Agent

An AI-powered automated feature engineering platform built with **Python**, **Streamlit**, **Scikit-Learn**, **Pandas**, and **Plotly**.

The application automatically analyzes datasets, generates meaningful engineered features, evaluates their impact on machine learning performance, and provides explainable insights through interactive visualizations.

---

## 📌 Overview

Feature engineering is one of the most important steps in building high-performing machine learning models. This project automates the process by:

* Detecting column data types automatically
* Generating candidate engineered features
* Evaluating feature usefulness using machine learning
* Ranking features based on performance improvement
* Providing human-readable explanations for generated features

The system follows an agent-based architecture consisting of:

* Data Analysis Agent
* Feature Generation Agent
* Evaluation Agent
* Explainability Engine

---

## ✨ Features

### 🔍 Automatic Data Type Detection

Automatically identifies:

* Numerical Columns
* Categorical Columns
* DateTime Columns
* Text Columns

### 🧠 Automated Feature Engineering

#### Numerical Features

* Squared Features
* Cubed Features
* Log Transformations
* Feature Interactions
* Ratios
* Differences

#### Categorical Features

* Frequency Encoding

#### DateTime Features

* Year
* Month
* Weekday Extraction

#### Text Features

* Word Count Features
* Keyword Detection Features

---

## 🤖 Machine Learning Evaluation

Each generated feature is evaluated using:

* Random Forest Classifier
* Cross Validation
* Accuracy-Based Scoring

The system compares the baseline model performance against the model enhanced with each engineered feature and ranks them accordingly.

---

## 📊 Interactive Visualizations

The application provides:

* Feature Improvement Bar Charts
* Model Accuracy Gauge Charts
* Correlation Heatmaps
* Explainable Feature Insights

---

## 💡 Explainable AI Insights

For every selected feature, the platform generates:

* Feature Type
* Mathematical Formula
* Performance Improvement
* Human-Readable Explanation

This helps users understand why a feature contributes to model performance.

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* Plotly

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Karthik-0027/Auto_Feature_Engineering_Agent.git

cd Auto_Feature_Engineering_Agent
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run Auto_Feature_Engineering.py
```

---

## 📥 Supported Input Formats

* CSV
* XLSX
* XLS

Datasets can be uploaded directly or loaded from URLs.

---

## 📤 Outputs

### Engineered Dataset

```text
autofe_premium_features.csv
```

### Executive Report

```text
autofe_executive_report.md
```

The generated report includes:

* Baseline Accuracy
* Enhanced Accuracy
* Top Engineered Features
* Feature Explanations
* Performance Improvements

---

## 🎯 Use Cases

* Automated Machine Learning (AutoML)
* Feature Engineering Research
* Data Science Projects
* Predictive Analytics
* Academic Research
* Machine Learning Prototyping

---

## 🚀 Future Enhancements

* Regression Model Support
* SHAP Explainability
* Deep Learning Evaluation
* Multi-Model Benchmarking
* Agentic AI Integration

---

## 👨‍💻 Author

**Karthik Gollapudi**

Final Year B.Tech – Data Science

Areas of Interest:

* Machine Learning
* Agentic AI
* Explainable AI
* Data Science
* Automated Feature Engineering

---

⭐ If you found this project useful, consider giving it a star.
