import streamlit as st
from common_functions import MODEL_TYPE
from data_preprocessing import preprocess_data
from model_preparation import train_model
from model_testing import test_model
import pandas as pd


st.set_page_config(layout="centered")

st.title("Загрузите датасет и обучите модель")

uploaded_file = st.file_uploader(
    "Загрузите датасет (*.csv, *.xls, *.xlsx)", type=["csv", "xls", "xlsx"]
)

if st.button("Обучить модель", type="primary"):
    if uploaded_file is not None:
        # по расширению, определяем какую функцию загрузки датафрейма использовать
        if uploaded_file.name.endswith((".xls", ".xlsx")):
            data = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.DataFrame()

        # Проверка на пустоту
        if data.empty:
            raise ValueError("Файл пустой.")

        # предобрабатываем данные
        preprocess_data(data, MODEL_TYPE.CUSTOM)
        # обучаем модель
        train_model(MODEL_TYPE.CUSTOM)
        test_model(MODEL_TYPE.CUSTOM)
        # модель переобучена, требуется перезагрузка модели
        st.session_state.reload_model_required = True

        st.write("Метрики обученной модели")
        st.write(test_model(MODEL_TYPE.CUSTOM))
