import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("📊 Predictive Analysis from CSV Upload")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Data")
    st.write(df.head())

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig = plt.figure(figsize=(8,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    st.pyplot(fig)

    st.subheader("Select Target Column")
    target = st.selectbox("Target Column", df.columns)

    if target:
        df = df.dropna()
        X = df.drop(columns=[target])
        y = df[target]

        X = X.select_dtypes(include='number')  # Only numeric features

        if not X.empty:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("Predictions vs Actual")
            st.write(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))

            st.write("R² Score:", r2_score(y_test, y_pred))
            st.write("MSE:", mean_squared_error(y_test, y_pred))
        else:
            st.warning("Not enough numeric columns to perform prediction.")
