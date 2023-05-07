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

import requests
url = "https://github.com/iassenegal221/loan-scoring/raw/main/app/data/logo.png"
response = requests.get(url, stream=True)
response.raise_for_status()

logo_image = Image.open(response.raw)
st.set_page_config(
page_title="CREDIT SCORING - DACHBOARD CLIENT SCORING",
page_icon=logo_image,
layout="wide",
)

@st.cache_data
@st.cache_resource
def load():
    """
    This functions aims to load data and models
    """
    import os
    import io
    import requests
    model_url = "https://github.com/Alhasdata/loan-scoring/raw/main/app/models/best_model.pkl"
    dataframe_url = "https://github.com/Alhasdata/loan-scoring/raw/main/app/models/full_data.pkl"
    model_response = requests.get(model_url)
    model_response.raise_for_status()
    model_bytes = io.BytesIO(model_response.content)
    model = joblib.load(model_bytes)
###### Features #######
    import requests
    import numpy as np
    import io

    features_url = "https://github.com/iassenegal221/loan-scoring/raw/main/app/data/training_features.pkl"
    features_response = requests.get(features_url)
    features_response.raise_for_status()

    features_bytes = io.BytesIO(features_response.content)
    features = np.load(features_bytes, allow_pickle=True)
###### Data #######
    data_url = "https://github.com/iassenegal221/loan-scoring/raw/main/app/data/full_data.pkl"
    data_response = requests.get(data_url)
    data_response.raise_for_status()

    data_bytes = io.BytesIO(data_response.content)
    dataframe = joblib.load(data_bytes)

    return model,features, dataframe


loan_scoring_classifier,features, dataframe= load()

scaler = MinMaxScaler()
data = scaler.fit_transform(dataframe[training_features])
data = pd.DataFrame(data, index=raw_data.index, columns=training_features)
raw_data = raw_data.reset_index()
probas = loan_scoring_classifier.predict_proba(data)
raw_data["proba_true"] = probas[:, 0]
mean_score = raw_data["proba_true"].mean()

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(data),
    feature_names=data.columns,
    # class_names=['bad', 'good'],
    mode="classification",
)

def main():
  # main function
  with st.sidebar:
    col1, col2, col3 = st.columns(3)
    col2.image(logo_image, use_column_width=True)
    st.markdown("""---""")
    st.markdown(
    """
    Wo we are?

    """,
    unsafe_allow_html=True,
    )
    st.info("""We are a financial company that offers consumer credit""")
    st.markdown("""---""")
    url = "https://github.com/iassenegal221/loan-scoring/blob/main/app/data/logo.png"


  tab1, tab2, tab3 = st.tabs(
  ["🏠 About this project", "📈 Make Predictions & Analyze", "🗃 Data Drift Reports"]
  )
  tab1.markdown("""---""")
  tab1.subheader("Credit Score")
  tab1.markdown(
  "This tool gives **guidance in credit granting decision** for our Relationship Managers. Based on customer's loan history and personnal informations, it predicts whether if he can refund a credit. It is based on one of the most powerful boosting algorith: **LightGBM**. \n To start, click on 'Make predictions & Analyze' at the top of the page. "
  )
  tab1.markdown("""---""")
    




if __name__ == "__main__":
    main()

