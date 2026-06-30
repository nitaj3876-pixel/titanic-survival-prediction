
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
st.markdown("""
<style>

.stApp{
    background-color:#F5F7FA;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

section[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#2563EB,#7C3AED);
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.columns([1,2,1])

col1, col2, col3 = st.sidebar.columns([1,2,1])

with col2:
    st.image("assets/logo.png", width=140)

st.sidebar.title("🚢 Titanic ML App")
st.divider()

st.sidebar.markdown("---")


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

# ================= HOME PAGE =================

if page == "🏠 Home":

    st.title("🚢 Titanic Survival Prediction System")
    st.divider()

    st.caption("Machine Learning Dashboard using Streamlit & Scikit-Learn")

    st.success("👋 Welcome to the Titanic Survival Prediction System")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📊 Dataset", "891")

    with col2:
        st.metric("🎯 Accuracy", "83.8%")

    with col3:
        st.metric("🧠 Algorithm", "Random Forest")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("🚀 Prediction Speed", "< 1 sec")

    with col2:
        st.metric("💻 Deployment", "Streamlit Cloud")
    

    st.markdown("---")

    left, right = st.columns([2,1])

    with left:

        st.subheader("📖 About Project")

        st.write("""
This project predicts whether a passenger survived the Titanic disaster.

The prediction is made using a Machine Learning model trained on the Titanic dataset.

The application is built using:

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
""")

    with right:

        st.info("💡 Quick Information")

        st.write("✔ Real-Time Prediction")

        st.write("✔ Interactive Dashboard")

        st.write("✔ Machine Learning")

        st.write("✔ Easy to Use")

    st.markdown("---")

    st.subheader("✨ Project Features")

    c1, c2 = st.columns(2)

    with c1:

        st.success("✅ Predict Survival")

        st.success("✅ Interactive Charts")

        st.success("✅ Dataset Dashboard")

    with c2:

        st.success("✅ Fast Prediction")

        st.success("✅ Clean UI")

        st.success("✅ Streamlit Deployment")
    st.markdown("## 🚀 Why Use This Application?")

    col1, col2 = st.columns(2)

    with col1:
     st.info("✔ Predict Titanic Passenger Survival")
    st.info("✔ Machine Learning Based")
    st.info("✔ Interactive Dashboard")

    with col2:
     st.success("✔ Fast Predictions")
    st.success("✔ Clean User Interface")
    st.success("✔ Live Streamlit Deployment")


    st.markdown("---")

    st.caption("Developed by Nita Jadhav ❤️")



    # ================= DATASET PAGE =================
elif page == "📊 Dataset":

    st.title("📊 Titanic Dataset")

    df = pd.read_csv("titanic.csv")

    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.subheader("Dataset Shape")
    rows, cols = df.shape

    col1, col2 = st.columns(2)

    with col1:
     st.metric("Rows", rows)

    with col2:
     st.metric("Columns", cols)

     st.subheader("Missing Values")

    st.dataframe(df.isnull().sum())
    st.subheader("Dataset Information")

    rows, cols = df.shape

    c1, c2 = st.columns(2)

    with c1:
     st.metric("Rows", rows)

    with c2:
     st.metric("Columns", cols)

    st.dataframe(df.head())
    csv = df.to_csv(index=False)

    st.download_button(
    label="📥 Download Dataset",
    data=csv,
    file_name="titanic_dataset.csv",
    mime="text/csv"
    )
    search = st.text_input("🔍 Search Passenger Name")

    if search:
     result = df[df["Name"].str.contains(search, case=False)]
    st.dataframe(result)



# ================= VISUALIZATION PAGE =================

elif page == "📈 Visualization":

    st.title("📈 Titanic Data Visualization")

    df = pd.read_csv("titanic.csv")

    # Survival Count
    st.subheader("🚢 Survival Count")
    survival = df["Survived"].value_counts()
    st.bar_chart(survival)

    # Survival Pie Chart
    st.subheader("🥧 Survival Percentage")

    fig, ax = plt.subplots()

    ax.pie(
        survival,
        labels=["Not Survived", "Survived"],
        autopct="%1.1f%%",
        startangle=90
    )

    ax.axis("equal")

    st.pyplot(fig)

    # Gender Count
    st.subheader("👨 Gender Distribution")
    gender = df["Sex"].value_counts()
    st.bar_chart(gender)

    # Passenger Class
    st.subheader("🎫 Passenger Class")
    pclass = df["Pclass"].value_counts()
    st.bar_chart(pclass)

    # Fare Distribution
    st.subheader("💰 Fare Distribution")
    st.line_chart(df["Fare"])

    # Age Area Chart
    st.subheader("👶 Age Area Chart")
    st.area_chart(df["Age"])

    # Age Histogram
    st.subheader("📊 Age Distribution")

    fig, ax = plt.subplots()

    ax.hist(df["Age"].dropna(), bins=20)

    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Passengers")

    ax.grid(alpha=0.3)

    st.pyplot(fig)

    

# ================= PREDICTION PAGE =================

elif page == "🤖 Prediction":

    st.title("🤖 Titanic Survival Prediction")

    st.info("Fill in the passenger details below and click Predict.")

    model = joblib.load("model.pkl")

    col1, col2 = st.columns(2)

    with col1:

        pclass = st.selectbox(
            "Passenger Class",
            [1, 2, 3]
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        age = st.slider(
            "Age",
            1,
            80,
            25
        )

    with col2:

        fare = st.number_input(
            "Fare",
            min_value=0.0,
            value=32.0
        )

        embarked = st.selectbox(
            "Embarked",
            ["S", "C", "Q"]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        predict = st.button(
            "🚀 Predict Survival",
            use_container_width=True,
            type="primary"
        )

    # Prediction will happen only after button click
    if predict:

        # Convert Gender
        sex = 1 if gender == "Male" else 0

        # Convert Embarked
        if embarked == "S":
            emb = 2
        elif embarked == "C":
            emb = 0
        else:
            emb = 1

        # Prepare Input
        input_data = [[
            pclass,
            sex,
            age,
            0,
            0,
            fare,
            emb
        ]]

        # Prediction will happen only after button click
    if predict:

        # Convert Gender
        sex = 1 if gender == "Male" else 0

        # Convert Embarked
        if embarked == "S":
            emb = 2
        elif embarked == "C":
            emb = 0
        else:
            emb = 1

        # Prepare Input
        input_data = [[
            pclass,
            sex,
            age,
            0,
            0,
            fare,
            emb
        ]]

        # Predict
        prediction = model.predict(input_data)

        # Probability
        probability = model.predict_proba(input_data)

        # Confidence
        if prediction[0] == 1:
            prob = probability[0][1]
        else:
            prob = probability[0][0]

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        st.progress(int(prob * 100))

        st.metric(
            "🎯 Prediction Confidence",
            f"{prob * 100:.2f}%"
        )

        if prediction[0] == 1:
            st.success("🎉 Passenger is likely to Survive")
            st.balloons()
        else:
            st.error("❌ Passenger is not likely to Survive")

        st.markdown("### 📋 Passenger Details")

        st.write(f"**Passenger Class:** {pclass}")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Age:** {age}")
        st.write(f"**Fare:** ₹{fare}")
        st.write(f"**Embarked:** {embarked}")
        report = f"""
        Titanic Survival Prediction Report

        Passenger Class : {pclass}
        Gender : {gender}
        Age : {age}
        Fare : {fare}
        Embarked : {embarked}

        Prediction : {"Survived" if prediction[0] == 1 else "Not Survived"}

        Confidence : {prob*100:.2f}%
        """

        st.download_button(
        label="📄 Download Report",
        data=report,
        file_name="prediction_report.txt",
        mime="text/plain"
    )


 #----------About Page----------------
elif page == "ℹ️ About":

 st.title("👩‍💻 About Developer")

 st.markdown("""
 ### Nita Jadhav

 🎓 Diploma in Information Engineering

 💻 Python | Machine Learning | Streamlit | FastAPI

 📧 nita3876@gmail.com

 🌐 GitHub: https://github.com/nitaj3876-pixel
 """)

elif page == "ℹ️ About":

    st.title("👩‍💻 About Developer")

    ...
    st.write("GitHub : github.com/nitaj3876-pixel")

# 👇 इथे add कर
st.markdown("---")

st.caption(
    "🚢 Titanic Survival Prediction | Developed by Nita Jadhav ❤️"
)
st.markdown("---")

st.caption(
    "© 2026 | Developed by Nita Jadhav | Python • Streamlit • Scikit-Learn"
)