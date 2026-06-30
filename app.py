
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib


st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

st.sidebar.title("📋 Menu")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset",
        "📈 Visualization",
        "🤖 Prediction",
        "ℹ️ About"
    ]
)

# Home Page
if page == "🏠 Home":

    st.title("🚢 Titanic Survival Prediction")

    st.success("👋 Welcome to the Titanic Survival Prediction App!")

    st.metric("Model Accuracy", "83.80%")

    st.subheader("📖 Project Description")

    st.write("""
This application predicts whether a passenger would survive the Titanic disaster
using a Machine Learning model.
""")
    
    # ================= DATASET PAGE =================
elif page == "📊 Dataset":

    st.title("📊 Titanic Dataset")

    df = pd.read_csv("titanic.csv")

    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.subheader("Dataset Shape")
    st.write(df.shape)

# ================= VISUALIZATION PAGE =================

elif page == "📈 Visualization":

    st.title("📈 Titanic Data Visualization")

    df = pd.read_csv("titanic.csv")

    st.subheader("Survival Count")

    survival = df["Survived"].value_counts()

    st.bar_chart(survival)

    st.subheader("Gender Count")

    gender = df["Sex"].value_counts()

    st.bar_chart(gender)

    st.subheader("Passenger Class")

    pclass = df["Pclass"].value_counts()

    st.bar_chart(pclass)
  

    

# ================= PREDICTION PAGE =================
elif page == "🤖 Prediction":

    st.title("🤖 Titanic Survival Prediction")
    model = joblib.load("model.pkl")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input("Age",1,100)

        gender = st.selectbox(
            "Gender",
            ["Male","Female"]
        )

        pclass = st.selectbox(
            "Passenger Class",
            [1,2,3]
        )

    with col2:

        fare = st.number_input(
            "Fare",
            0.0
        )

        embarked = st.selectbox(
            "Embarked",
            ["S","C","Q"]
        )

    predict = st.button("Predict")

    if predict:

        st.success("Prediction Done")

