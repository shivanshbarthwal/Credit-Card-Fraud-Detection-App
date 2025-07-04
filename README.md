# 💳 Credit Card Fraud Detection App

A machine learning-powered Streamlit web application that predicts whether a credit card transaction is **legitimate** or **fraudulent**. Built using `Logistic Regression` and a balanced dataset, it provides probability-based results with a visual pie chart.

---

## 🧠 Features

- ✅ Input transaction features manually through the web interface
- 🔍 Logistic Regression model for binary classification
- 📉 Data balanced using under-sampling for improved fairness
- 📊 Pie chart visualization of prediction probabilities
- 🧪 Error handling for invalid inputs

---

## 🛠️ Technology Stack

- **Python 3**
- **Streamlit** – for building the web interface
- **scikit-learn** – for machine learning model
- **NumPy & Pandas** – for data handling
- **Matplotlib** – for pie chart visualization

---

## 📦 Installation

Install the required packages using pip:

```bash
pip install streamlit scikit-learn numpy pandas matplotlib
```

---

## 🚀 How to Run

Make sure you have the `creditcard.csv` dataset in the same directory.

Then run:

```bash
streamlit run app.py
```

> `app.py` should contain the Streamlit code provided.

---

## 📁 File Structure

```
fraud-detection-app/
├── app.py              # Main Streamlit application
├── creditcard.csv      # Dataset used for training
├── README.md           # Project documentation
```

---

## 🧪 Example Input

```
-1.3598071, -0.0727812, 2.5363467, ..., -0.1083005, 1.1195934, 0.3453457
```

---

## ⚠️ Notes

- Input must be **comma-separated numerical values** only (30 features expected)
- You must use the same number and order of features as in the original dataset
- The dataset is **balanced** using under-sampling for effective binary classification
