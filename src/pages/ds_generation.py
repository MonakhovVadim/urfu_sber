import streamlit as st
import numpy as np
import pandas as pd
from common_functions import load_scor_model, save_scor_model
from generation import dataset_generation
import os


st.set_page_config(layout="wide")
st.title("Генерация синтетического датасета")
st.markdown(
    """
    Датасет создается на основании указанных ниже критериев и их значимости.
    В поле :red[weight] указаны веса (значимость) каждого критерия. 
    Вы можете скорректировать значимость любого критерия, обладая экспертными знаниями в предметной области.
    Указанные критерии и их значмость используются для алгоритма математической оценки риска внедрения и генерации синтетического датасета. 
    """
)

# загружаем математическую скормодель из файла (критерии, веса, мин, макс и тп)
df = load_scor_model()
if not df.empty:
    # Вывод таблицы с возможностью редактирования
    edited_df = st.data_editor(
        df,
        width=1400,
        height=455,
        column_order=["name", "weight", "description"],
        disabled=["name", "description"],
    )

    # Кнопка сохранения введенных данных в файл
    if st.button("Сохранить изменения"):
        save_scor_model(edited_df)
        st.success("Данные успешно сохранены!")

    # Кнопка генерации датасета
    if st.button("Сгенерировать датасет"):
        # вывод скорректированного датафрейма. УДАЛИТЬ ПОСЛЕ ОТЛАДКИ
        df_synt = dataset_generation(edited_df, 10)
        st.success("Датасет сгенерирован. Можете скачать его")

        os.makedirs("data", exist_ok=True)
        excel_file = "data/synthetic_dataset.xlsx"
        df_synt.to_excel(excel_file, index=False)

        # Кнопка скачивания датасета
        with open(excel_file, "rb") as f:
            st.download_button(
                label="Скачать Excel файл",
                data=f,
                file_name=excel_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.dataframe(df_synt)
else:
    st.error("Отсутствуют файлы с критериями оценки риска", icon="🚨")
