import streamlit as st
import joblib
import requests
import io
import numpy as np

def load_model():
    model_url = "https://github.com/Alhasdata/loan-scoring/raw/main/app/models/best_model.pkl"
    model_response = requests.get(model_url)
    model_response.raise_for_status()
    model_bytes = io.BytesIO(model_response.content)
    model = joblib.load(model_bytes)
    return model

def load_features():
    features_url = "https://github.com/iassenegal221/loan-scoring/raw/main/app/data/training_features.pkl"
    features_response = requests.get(features_url)
    features_response.raise_for_status()
    features_bytes = io.BytesIO(features_response.content)
    features = np.load(features_bytes, allow_pickle=True)
    return features

def load_dataframe():
    data_url = "https://github.com/iassenegal221/loan-scoring/raw/main/app/data/full_data.pkl"
    data_response = requests.get(data_url)
    data_response.raise_for_status()
    data_bytes = io.BytesIO(data_response.content)
    dataframe = joblib.load(data_bytes)
    return dataframe

def main():
    model = load_model()
    features = load_features()
    dataframe = load_dataframe()

    # Streamlit App
    st.title("Loan Scoring App")
    st.subheader("Data Visualization")

    # Display the dataframe
    st.write(dataframe)

if __name__ == "__main__":
    main()
