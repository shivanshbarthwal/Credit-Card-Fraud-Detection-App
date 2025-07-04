import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt

data = pd.read_csv('creditcard.csv')
legit = data[data.Class == 0]
fraud = data[data.Class == 1]
legit_sample = legit.sample(n=len(fraud), random_state=2)
balanced_data = pd.concat([legit_sample, fraud], axis=0)

X = balanced_data.drop(columns="Class", axis=1)
y = balanced_data["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=500, random_state=42)
model.fit(X_train, y_train)


st.title("Credit Card Fraud Detection")
st.write("This app predicts whether a transaction is legitimate or fraudulent based on input features.")

input_features = st.text_input(
    "Enter transaction features (comma-separated values):",
    placeholder="e.g., -1.23, 0.45, 2.34, ..."
)

if st.button("Predict"):
    try:
        features = np.array(input_features.split(','), dtype=np.float64).reshape(1, -1)

        prediction = model.predict(features)
        probabilities = model.predict_proba(features).flatten()

        if prediction[0] == 0:
            st.success("This transaction is Legitimate.")
        else:
            st.error("This transaction is Fraudulent.")

        st.subheader("Prediction Probabilities")
        
        fig, ax = plt.subplots()
        labels = ["Legitimate", "Fraudulent"]
        ax.pie(probabilities, labels=labels, autopct="%1.1f%%", startangle=90, colors=["green", "red"])
        ax.axis("equal")  # Equal aspect ratio ensures the pie is circular
        st.pyplot(fig)

    except ValueError:
        st.error("Invalid input. Please enter numeric values separated by commas.")
    except Exception as e:
        st.error(f"An error occurred: {e}")