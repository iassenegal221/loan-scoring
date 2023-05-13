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
## Images ######
url = "https://github.com/iassenegal221/loan-scoring/raw/main/app/data/logo.png"
response = requests.get(url, stream=True)
response.raise_for_status()
logo_image = Image.open(response.raw)
##
url = "https://github.com/iassenegal221/loan-scoring/blob/main/app/data/home_credit.jpeg"
#response = requests.get(url, stream=True)
#response.raise_for_status()
#cover_image = Image.open(response.raw)
##
url = "https://github.com/iassenegal221/loan-scoring/raw/main/app/data/lifecycle.png"
#response = requests.get(url, stream=True)
#response.raise_for_status()
#home_image = Image.open(response.raw)

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
data = scaler.fit_transform(dataframe[features])
data = pd.DataFrame(data, index=dataframe.index, columns=features)
raw_data = dataframe.reset_index()
probas = loan_scoring_classifier.predict_proba(data)
raw_data["proba_true"] = probas[:, 0]
mean_score = raw_data["proba_true"].mean()

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(data),
    feature_names=data.columns,
    # class_names=['bad', 'good'],
    mode="classification",
)

def plot_preds_proba(customer_id):
    """
    This functions aims plot income, annuities and credit vizuals
    """
    user_infos = {
        "Income": raw_data[raw_data["SK_ID_CURR"] == customer_id][
            "AMT_INCOME_TOTAL"
        ].values[0],
        "Credit": raw_data[raw_data["SK_ID_CURR"] == customer_id]["AMT_CREDIT"].values[
            0
        ],
        "Annuity": raw_data[raw_data["SK_ID_CURR"] == customer_id][
            "AMT_ANNUITY"
        ].values[0],
    }
    pred_proba_df = pd.DataFrame(
        {"Amount": user_infos.values(), "Operation": user_infos.keys()}
    )
    c = (
        alt.Chart(pred_proba_df)
        .mark_bar()
        .encode(x="Operation", y="Amount", color="Operation")
        .properties(width=330, height=310)
    )
    st.altair_chart(c)
#st.write(dataframe)
def main():
    # main function
    with st.sidebar:
        col1, col2, col3 = st.columns(3)
        col2.image(logo_image, use_column_width=True)
        st.markdown("""---""")
        st.markdown(
            """
                        <h4 style='text-align: center; color: black;'> Wo we are? </h4>
                        """,
            unsafe_allow_html=True,
        )
        st.info("""We are a recherche développement company that offers AI based services""")
        st.markdown("""---""")
        #st.sidebar.image(cover_image, use_column_width=True)

    tab1, tab2, tab3 = st.tabs(
        ["🏠 About this application", "📈 Make Predictions & Analyze", "🗃 Data Drift Reports"]
    )
    tab1.markdown("""---""")
    tab1.subheader("Credit Score")
    tab1.markdown(
        "This tool gives **guidance in credit granting decision** for our Relationship Managers. Based on customer's loan history and personnal informations, it predicts whether if he can refund a credit. It is based on one of the most powerful boosting algorith: **LightGBM**. \n To start, click on 'Make predictions & Analyze' at the top of the page. "
    )
    tab1.markdown("""---""")


    with tab2.subheader("Loan Scoring Model"):
        with st.form(key="myform"):
            # user_liste = data.index
            # user_id_value = st.selectbox('Select customer id', user_liste)
            user_id_value = st.number_input("Select customer id", min_value=100001)
            submit_button = st.form_submit_button(label="Prédire")

            if submit_button:
                if isinstance(user_id_value, int) and user_id_value in data.index:
                    st.write("Client sélectionné : ", user_id_value)
                    col1, col2 = st.columns(2)
                    # data = data.reset_index()
                    user = data[data.index == int(user_id_value)]
                    # prediction = loan_scoring_classifier.predict(user)[0]
                    probas_user = loan_scoring_classifier.predict_proba(user)
                    probabilities = dict(
                        zip(
                            loan_scoring_classifier.classes_,
                            np.round(probas_user[0], 3),
                        )
                    )

                    # display results
                    # display results
                    col1, col2 = st.beta_columns(2)

                    with col1:
                        st.info("Informations client")
                        user_infos = raw_data[raw_data["SK_ID_CURR"] == user_id_value]

                        dict_infos = {
                            "Age": int(user_infos["DAYS_BIRTH"] / -365),
                            "Sexe": user_infos["CODE_GENDER"]
                                .replace(["F", "M"], ["Female", "Male"])
                                .item(),
                            "Statut matrimonial": user_infos["NAME_FAMILY_STATUS"].item(),
                            "Niveau éducation": user_infos["NAME_EDUCATION_TYPE"].item(),
                            "Expérience professionnelle": int(user_infos["DAYS_EMPLOYED"].values / -365),
                            "Activité": user_infos["NAME_INCOME_TYPE"].item(),
                            "Revenu": user_infos["AMT_INCOME_TOTAL"].item(),
                        }
                        st.write(dict_infos)

                    with col2:
                        st.info("Historique des prêt du client")
                        dict_infos = {
                            "Type de contrat": user_infos["NAME_CONTRACT_TYPE"].item(),
                            "Montant du credit": user_infos["AMT_CREDIT"].item(),
                            "Annuité": user_infos["AMT_ANNUITY"].item(),
                        }
                        st.write(dict_infos)
                        # st.metric(label='Accuracy', value='', delta='1.6')
                        st.markdown("""---""")
                        st.info("Credit Score")
                        # c1, c2, c3, c4, c5 = st.columns(5)
                        if round(probabilities[0] * 100, 2) > 60:
                            st.metric(
                                "Score élevé",
                                value=round(probabilities[0] * 100, 2),
                                delta=f"{round((probabilities[0]-0.6)*100,2)}",
                            )

                            st.success(
                                "This customer is a potential refunder", icon="✅"
                            )
                        elif 50 < round(probabilities[0] * 100, 2) < 60:
                            st.metric(
                                "Score acceptabe",
                                value=round(probabilities[0] * 100, 2),
                                delta=f"{round((probabilities[0]-0.6)*100,2)}",
                            )

                            st.warning(
                                "Ce client peut avoir des difficultés à rembourser",
                                icon="⚠️",
                            )
                        else:
                            st.metric(
                                "Score Faible",
                                value=round(probabilities[0] * 100, 2),
                                delta=f"{round((probabilities[0]-0.6)*100,2)}",
                            )
                            st.error("Ce client ne peut pas rembourser", icon="🚨")
                      with col2:
                        st.info("Features contribution")
                        exp = explainer.explain_instance(
                            data_row=data.loc[user_id_value],
                            predict_fn=loan_scoring_classifier.predict_proba,
                        )
                        components.html(exp.as_html(), height=550)
                        st.markdown("""---""")
                        plot_preds_proba(user_id_value)

                else:
                    st.error("Please, enter a valid customer id.", icon="🚨")

            else:
                st.markdown("""---""")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Quick Tutorial**")
                    st.info(
                        """
                        1. Select or edit **Customer ID**.
                        2. Click **Show Infos** button.
                        4. Analyze user informations and prediction results
                        """,
                    )
                    st.markdown("**Few Tips**")
                    st.success(
                        """
                        1. Consider all the informations, use the vizual oh the bottom right.
                        2. Use Credit Score wisely, delta becomes green for potential refunder.
                        4. Keep an eye on the data drift report is essential.
                        """,
                    )

              #  with col2:
                #    st.markdown("**Global Explainability of the model**")
                #    st.image(
                 #       Image.open("./data/explainability.png"), use_column_width=True
                #    )




if __name__ == "__main__":
    main()
