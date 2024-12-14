import streamlit as st
from common_functions import save_scor_model, MODEL_TYPE
from data_generation import dataset_generation
import os

st.set_page_config(layout="wide")

st.title("Генерация синтетического датасета")
st.markdown(
    """
    Датасет создается на основании указанных ниже критериев и их значимости.
    В поле :red[weight] указаны веса (значимость) каждого критерия.
    Вы можете скорректировать значимость любого критерия, обладая 
    экспертными знаниями в предметной области.
    Указанные критерии и их значмость используются для алгоритма математической
    оценки риска внедрения и генерации синтетического датасета.
    """
)

# загружаем математическую скормодель из файла (критерии, веса, мин, макс и тп)
# df_param = load_scor_model(MODEL_TYPE.DEFAULT)
if not st.session_state.df_params.empty:
    # Вывод таблицы с возможностью редактирования
    edited_df = st.data_editor(
        st.session_state.df_params,
        width=1400,
        height=455,
        column_order=["name", "weight", "description"],
        disabled=["name", "description"],
    )

    # Кнопка сохранения введенных данных в файл
    if st.button("Сохранить изменения"):
        save_scor_model(edited_df)
        # обновляем параметры скормодели
        st.session_state.df_params = edited_df
        st.success("Данные успешно сохранены!")

    # Кнопка генерации датасета
    if st.button("Сгенерировать датасет"):
        # генерируем датасет  (во время отладки генерируем 10 строк)
        df_synt = dataset_generation(edited_df, 50)
        # отображаем датасет (УДАЛИТЬ ПОСЛЕ ОТЛАДКИ)
        st.dataframe(df_synt)

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
else:
    st.error("Отсутствуют файлы с критериями оценки риска", icon="🚨")
