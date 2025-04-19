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
    st.subheader("1️⃣ Preview of Data")
    st.write(df.head())

    st.subheader("2️⃣ Statistical Summary")
    st.write(df.describe())

    st.subheader("3️⃣ Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig = plt.figure(figsize=(8,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    st.pyplot(fig)

    st.subheader("4️⃣ Bar Chart")
    categorical_cols = df.select_dtypes(include='object').columns
    numeric_cols = df.select_dtypes(include='number').columns

    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        cat_col = st.selectbox("Choose a Categorical Column", categorical_cols)
        num_col = st.selectbox("Choose a Numeric Column", numeric_cols)
        bar_data = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
        st.bar_chart(bar_data)

    st.subheader("5️⃣ Histogram")
    hist_col = st.selectbox("Choose a Numeric Column for Histogram", numeric_cols)
    fig = plt.figure()
    sns.histplot(df[hist_col], kde=True)
    st.pyplot(fig)

    st.subheader("6️⃣ Box Plot")
    box_col = st.selectbox("Choose a Numeric Column for Box Plot", numeric_cols, key="box")
    fig = plt.figure()
    sns.boxplot(x=df[box_col])
    st.pyplot(fig)

    st.subheader("7️⃣ Scatter Plot")
    x_axis = st.selectbox("X-axis Column", numeric_cols, key="x_axis")
    y_axis = st.selectbox("Y-axis Column", numeric_cols, key="y_axis")
    fig = plt.figure()
    sns.scatterplot(x=df[x_axis], y=df[y_axis])
    st.pyplot(fig)

    st.subheader("📈 Predictive Model")
    target = st.selectbox("Select Target Column", df.columns)

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

