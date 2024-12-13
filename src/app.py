import streamlit as st
import pandas as pd
import numpy as np
from common_functions import load_scor_model, load_model, calculate_scor


def main():
    """
    Основная функция приложения
    """

    # Загружаем модель
    model = load_model()

    # Выводим тайтл и кратко обозначаем функционал приложения
    st.title("Определитель риска внедрения")
    st.write(
        """Сервис оценки технологического риска внедрений релизов программного обеспечения, 
        предназначенного для выполнения процедур трансформации данных (ETL) и 
        расчетов на витринах хранилища данных (DWH).
         """
    )

    # Состав критериев
    df_params = load_scor_model()
    features = list(df_params.name)

    # Выводим критерии оценки и слайдеры для выбора значения критерия
    num_elements = []
    for _, feature in df_params.iterrows():
        element = st.slider(
            feature[0],
            min_value=feature[1],
            max_value=feature[2],
            value=0,
            step=1,
            help=f"{feature[6]}.  {feature[3]}",
        )
        num_elements.append(element)

    if st.button("Определить риск"):
        user_choice = {}

        for feature, fvalue in zip(features, num_elements):
            user_choice[feature] = fvalue

        # Переводим пользовательский вывод в датафрейм
        df = pd.DataFrame(user_choice, index=[0])

        # Математический расчет скорбала
        scor_math = calculate_scor(df, df_params)

        # Рандом скорбала модели
        # TODO: когда будет готова модель, поменять
        scor_model = np.random.uniform(0, 5, 1)
        # scor_model = model.predict_proba(pd.DataFrame(df, columns=df.columns))

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
