import streamlit as st
from model_train import train_model
from common_functions import load_scor_model


st.set_page_config(layout="centered")
st.title("Загрузите датасет и обучите модель")

uploaded_file = st.file_uploader(
    "Загрузите датасет (*.csv, *.xls, *.xlsx)", type=["csv", "xls", "xlsx"]
)

if st.button("Обучить модель", type="primary"):
    if uploaded_file is not None:
        # состав ожидаемых колонок для проверки датасета
        features = load_scor_model()
        expected_columns = list(features.name)

        model, metrics = train_model(uploaded_file, expected_columns)
        st.write(f"Метрики обученной модели ")
        st.write(metrics)
