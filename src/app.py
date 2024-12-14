import streamlit as st
import pandas as pd
from common_functions import (
    load_scor_model,
    load_pipeline,
    load_model,
    calculate_scor,
    MODEL_TYPE,
    DATA_TYPE,
)

# Инициализация состояний
if "df_params" not in st.session_state:
    st.session_state.df_params = load_scor_model(MODEL_TYPE.DEFAULT)

if "reload_model_required" not in st.session_state:
    st.session_state.reload_model_required = False

if "model" not in st.session_state:
    st.session_state.model = load_model(MODEL_TYPE.DEFAULT)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = load_pipeline(MODEL_TYPE.DEFAULT)


def main():
    """
    Основная функция приложения
    """

    # Выводим тайтл и кратко обозначаем функционал приложения
    st.title("Определитель риска внедрения")
    st.write(
        """Сервис оценки технологического риска внедрений релизов программного обеспечения,
        предназначенного для выполнения процедур трансформации данных (ETL) и
        расчетов на витринах хранилища данных (DWH).
         """
    )

    if st.session_state.df_params is not None:
        features_name = list(st.session_state.df_params.name)
    else:
        st.error(f"Ошибка: файл с данными дефолтной скормодели отсутствует!")

    # Выводим критерии оценки и слайдеры для выбора значения критерия
    num_elements = []
    for _, param in st.session_state.df_params.iterrows():
        element = st.slider(
            param[0],
            min_value=param[1],
            max_value=param[2],
            value=0,
            step=1,
            help=f"{param[6]}.  {param[3]}",
        )
        num_elements.append(element)

    if st.button("Определить риск"):
        user_choice = {}

        # если приложение было запущено первый раз и не через docker,
        # то файлы с моделью и пайплайном отсутствуют
        # поэтому проводим генерацию датасета, препроцессинг данных и обучим модель
        # можно удалить в итоговой версии приложения
        if (st.session_state.pipeline and st.session_state.model) is None:
            from data_generation import dataset_generation
            from data_preprocessing import preprocess_data
            from model_preparation import train_model
            from common_functions import save_dataset

            with st.status("Идет обучение модели..."):
                # генерирем датасет
                data = dataset_generation(st.session_state.df_params, 1000)
                save_dataset(data, DATA_TYPE.BASE, MODEL_TYPE.DEFAULT)
                # предобрабатываем данные
                preprocess_data(data, MODEL_TYPE.DEFAULT)
                # обучаем модель
                train_model(MODEL_TYPE.DEFAULT)

                st.session_state.pipeline = load_pipeline(MODEL_TYPE.DEFAULT)
                st.session_state.model = load_model(MODEL_TYPE.DEFAULT)
                print("Модель обучена!")

        # если пользовтаель переобучил модель, загружаем новую
        if st.session_state.reload_model_required:
            st.session_state.pipeline = load_pipeline(MODEL_TYPE.CUSTOM)
            st.session_state.model = load_model(MODEL_TYPE.CUSTOM)
            # обнуляем флаг, чтобы не перезагружать модель каждый раз
            st.session_state.reload_model_required = False

        for feature, fvalue in zip(features_name, num_elements):
            user_choice[feature] = fvalue

        # Переводим пользовательский вывод в датафрейм
        df = pd.DataFrame(user_choice, index=[0])
        # приводим столбцы в правильное расположение (как в изначальном датафрейме)
        df = df[st.session_state.pipeline.feature_names_in_]
        # print("Датафрейм с корректной последовательность колонок\n", df)

        # Математический расчет скорбала
        scor_math = calculate_scor(df, st.session_state.df_params)

        # преобразуем сырые данные по пайплайну, который использовался при преобразовании датасета
        df_scaled = st.session_state.pipeline.transform(df)
        # print("Преобразованный датафрейм\n", df_scaled)

        scor_model = st.session_state.model.predict(
            pd.DataFrame(df_scaled, columns=df.columns)
        )[0]

        st.write(f"Оценка риск-балла с помощью модели: **:red[{scor_model}]**")
        st.write(f"Оценка риск-балла с помощью мат.алгоритма: **:red[{scor_math}]**")


if __name__ == "__main__":
    pages = [
        st.Page(main, title="Определение скорбала"),
        st.Page("pages/ds_generation.py", title="Генерация датасета"),
        st.Page("pages/upload_train.py", title="Загрузка датасета и обучение"),
        st.Page("pages/version.py", title="Версии приложения"),
    ]

    pg = st.navigation(pages)

    pg.run()
