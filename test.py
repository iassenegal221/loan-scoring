import os
import io
import requests
import joblib
model_url = "https://github.com/Alhasdata/loan-scoring/raw/main/app/models/best_model.pkl"
dataframe_url = "https://github.com/Alhasdata/loan-scoring/raw/main/app/models/full_data.pkl"
model_response = requests.get(model_url)
model_response.raise_for_status()
model_bytes = io.BytesIO(model_response.content)
model = joblib.load(model_bytes)