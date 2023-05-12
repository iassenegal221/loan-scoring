import streamlit as st
import requests
import io
import joblib
import pandas as pd

def load_dataframe():
    data_url = "https://github.com/iassenegal221/loan-scoring/raw/main/app/data/full_data.pkl"
    data_response = requests.get(data_url)
    data_response.raise_for_status()
    data_bytes = io.BytesIO(data_response.content)
    dataframe = joblib.load(data_bytes)
    return dataframe

def main():
    dataframe = load_dataframe()

    # Streamlit App
    st.title("Loan Scoring App")
    st.subheader("Data Visualization")

    # Display the dataframe
    st.write(dataframe)

if __name__ == "__main__":
    main()
