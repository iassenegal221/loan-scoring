"""
Dashboard aims to....
"""
import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import altair as alt
from streamlit.components import v1 as components
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from lime import lime_tabular




def main():
    import requests
    import numpy as np
    import io

    features_url = "https://github.com/iassenegal221/loan-scoring/raw/main/app/data/training_features.pkl"
    features_response = requests.get(features_url)
    features_response.raise_for_status()

    features_bytes = io.BytesIO(features_response.content)
    features = np.load(features_bytes, allow_pickle=True)

    st.dataframe(features

    




if __name__ == "__main__":
    main()

