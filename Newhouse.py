# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import time

# ========== CONFIG ==========
st.set_page_config(page_title="House Price Prediction App", layout="wide")

# Load your trained model
@st.cache_resource
def load_model():
    return joblib.load("my_house_price_model.pkl")  # change filename if needed

# Load your dataset
@st.cache_data
def load_data():
    return pd.read_csv("data\house_price.csv")  # change filename if needed

model = load_model()
data = load_data()

# ========== SIDEBAR ==========
st.sidebar.title("Navigation")
pages = ["🏠 Home", "📊 Data Exploration", "📈 Visualisations", "🤖 Model Prediction", "📉 Model Performance"]
selection = st.sidebar.radio("Go to", pages)

# ========== HOME ==========
if selection == "🏠 Home":
    st.title("House Price Prediction App")
    st.write("""
    This application allows you to explore the housing dataset, visualise patterns,
    and make real-time predictions of house prices using our trained ML model.
    """)
    st.markdown("**Author:** Dilsara Supeshala  |  **Date:** 2025")

# ========== DATA EXPLORATION ==========
elif selection == "📊 Data Exploration":
    st.title("Data Exploration")
    st.write("Overview of the dataset used for training the model.")

    # Dataset shape
    st.subheader("Dataset Overview")
    st.write(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")
    st.write("Columns and Data Types:")
    st.write(pd.DataFrame(data.dtypes, columns=["Data Type"]))

    # Sample data
    st.subheader("Sample Data")
    st.write(data.sample(5))

    # Interactive filtering
    st.subheader("Filter Data")
    col_to_filter = st.selectbox("Select column to filter", data.columns)
    if pd.api.types.is_numeric_dtype(data[col_to_filter]):
        min_val, max_val = float(data[col_to_filter].min()), float(data[col_to_filter].max())
        user_range = st.slider("Select range", min_val, max_val, (min_val, max_val))
        filtered_data = data[data[col_to_filter].between(*user_range)]
    else:
        unique_vals = data[col_to_filter].unique()
        selected_vals = st.multiselect("Select values", unique_vals, default=unique_vals)
        filtered_data = data[data[col_to_filter].isin(selected_vals)]

    st.write(f"Filtered data shape: {filtered_data.shape}")
    st.dataframe(filtered_data)

# ========== VISUALISATIONS ==========
elif selection == "📈 Visualisations":
    st.title("Visualisations")

    # 1. Histogram
    st.subheader("Histogram")
    col_hist = st.selectbox("Select column for histogram", data.select_dtypes(include=np.number).columns)
    fig, ax = plt.subplots()
    sns.histplot(data[col_hist], kde=True, ax=ax)
    st.pyplot(fig)

    # 2. Scatter plot
    st.subheader("Scatter Plot")
    x_axis = st.selectbox("X-axis", data.select_dtypes(include=np.number).columns)
    y_axis = st.selectbox("Y-axis", data.select_dtypes(include=np.number).columns)
    fig, ax = plt.subplots()
    sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
    st.pyplot(fig)

    # 3. Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ========== MODEL PREDICTION ==========
elif selection == "🤖 Model Prediction":
    st.title("House Price Prediction")
    st.write("Enter the features below to get a predicted house price.")

    # Create input fields for model features
    feature_inputs = {}
    for feature in data.drop(columns=["Price"]).columns:  # replace "price" with your target column
        if pd.api.types.is_numeric_dtype(data[feature]):
            feature_inputs[feature] = st.number_input(f"Enter {feature}", 
                                                      min_value=float(data[feature].min()), 
                                                      max_value=float(data[feature].max()), 
                                                      value=float(data[feature].mean()))
        else:
            feature_inputs[feature] = st.selectbox(f"Select {feature}", data[feature].unique())

    input_df = pd.DataFrame([feature_inputs])

    # Prediction with loading spinner
    if st.button("Predict Price"):
        with st.spinner("Predicting..."):
            time.sleep(1)
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted House Price: ${prediction:,.2f}")

# ========== MODEL PERFORMANCE ==========
elif selection == "📉 Model Performance":
    st.title("Model Performance")
    st.write("Evaluation metrics on test data.")

    # Assuming you have test data stored
    test_data = pd.read_csv("data\house_price.csv")  # change if needed
    X_test = test_data.drop(columns=["Price"])
    y_test = test_data["Price"]

    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.metric("Mean Squared Error", f"{mse:,.2f}")
    st.metric("R² Score", f"{r2:.2f}")

    # Residual plot
    st.subheader("Residual Plot")
    fig, ax = plt.subplots()
    sns.residplot(x=y_pred, y=y_test - y_pred, lowess=True, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    st.pyplot(fig)

    # Predicted vs Actual
    st.subheader("Predicted vs Actual")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)
