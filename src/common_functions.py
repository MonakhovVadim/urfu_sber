from pathlib import Path
from enum import Enum
import joblib
import pandas as pd
import os
import numpy as np


DATA_TYPE = Enum("DATA_TYPE", ["BASE", "TRAIN", "TEST"])
PATH_DATASETS = Path.cwd() / "data"
PATH_BASE_DS = PATH_DATASETS / "raw"
PATH_PROCESSED_DS = PATH_DATASETS / "processed"
PATH_MODEL = Path.cwd() / "models"


def load_scor_model():
    """
    Загружает веса для алгоритма математической оценки.
    Если существует файл scor_model.xlsx (сохраненный пользователем), то подгружаются эти данные
    Если файл не существует, загружаются дефолтные данные из default_scor_model.xlsx
    Возвращает:
     - датафрейм содержищий критерии оценки, их веса, минимально и максимально
     возможные значения критерив, описания критериев
    """
    if os.path.exists("data/scor_model.xlsx"):
        return pd.read_excel("data/scor_model.xlsx", dtype={"weight": float})
    elif os.path.exists("data/default_scor_model.xlsx"):
        return pd.read_excel("data/default_scor_model.xlsx")
    else:
        return pd.DataFrame()


def save_scor_model(df):
    """
    Сохраняет новые веса критериям, заданные пользователем
    """
    df.to_excel("data/scor_model.xlsx", index=False)


def recalculate_values(df_data, df_params):
    """
    Функция возвращает датафрейм с оценкой критериев, измененных с учетом direct_dependence каждого критерия
    Критерии в датафрейме могут иметь либо прямую либо обратную зависимость, между значением критерия и риском.
    Каждый критерий имеет параметр direct_dependence
    - direct_dependence = 1, означает прямую зависмость риска от значения (чем выше значение, тем выше риск)
    - direct_dependence = 0, означает обратную зависимость, чем ниже значение, тем выше риск
    Параметры:
     - df_data: датафрейм с данными
     - df_params: датафрейм с параметрами критериев (вес, зависимость, максимальное значение)
     Возвращает:
     - измененный df с учетом direct_dependence
    """

    # Создаем копию датафрейма с данными для изменения
    df_recalculated = df_data.copy()

    # Проходим по всем фичам в параметрах
    for _, row in df_params.iterrows():
        feature = row["name"]
        direct_dependence = row["direct_dependence"]
        max_value = row["max_value"]

        # Если обратная зависимость, пересчитываем значения
        if direct_dependence == 0:
            df_recalculated[feature] = max_value - df_recalculated[feature]

    return df_recalculated


def normalize_df(df_params):
    """
    Нормализует веса датафрейма с критериями
    """
    # совокупный вес всех критериев
    total_weight = df_params.weight.sum()
    # нормализуем веса
    df_params["weight"] = df_params["weight"] / total_weight

    return df_params


def calculate_scor(df_data, df_params):
    """
    Функция математически рассчитывает и возвращает скорбал
    Параметры:
     - df_data: датафрейм с введенными пользователем значениями критериев
     - df_params: датафрейм с весами критериев
     Возвращает:
     - математически рассчитанный бал оценки риска релиза

    """
    # нормализуем вес критериев
    df_params = normalize_df(df_params)
    # меняем оценки на противоположные для критериев с обатной зависимостью
    df_data = recalculate_values(df_data, df_params)

    return sum(
        df_data.iloc[0][feature]
        * df_params.loc[df_params["name"] == feature, "weight"].values[0]
        for feature in df_params["name"]
    )


def dataset_generation(df_params, num_samples=1000):
    """
    Функция генерации синтетического датасета. Таргетный скорбал рассчитывается математически,
    на основании весов, заданных экспертом и сохраненных в файле scor_model
    либо default_scor_model (если веса по-умолчанию не менялись)
    Параметры:
    - df_params: датафрейм с критериями и их весами
    - num_samples: количесвто строк для генерации в синтетическом датасете (по умолчанию 1000)
    """

    # если датафрейм пустой, значит у нас нет весов и мы не сможем сгенерировать датасет
    if df_params.empty:
        return -1

    df_params = normalize_df(df_params)

    # Генерация синтетических данных
    synthetic_data = {}
    for index, row in df_params.iterrows():
        feature_values = np.random.randint(
            row["min_value"], row["max_value"] + 1, num_samples
        )
        synthetic_data[row["name"]] = feature_values

    # Создание датафрейма из сгенерированных данных
    df_synthetic = pd.DataFrame(synthetic_data)

    # Функция для расчета целевой переменной
    def calc_target(row):
        return sum(
            row[feature]
            * df_params.loc[df_params["name"] == feature, "weight"].values[0]
            for feature in df_params["name"]
        )

    # Вычисление целевой переменной и добавление в датафрейм
    df_synthetic["target"] = df_synthetic.apply(calc_target, axis=1)

    return df_synthetic


# функция осуществляет формирование пути к каталогу по типу датасета
def path_by_type(data_type):

    if data_type == DATA_TYPE.BASE:
        path = PATH_BASE_DS
    elif data_type == DATA_TYPE.TRAIN:
        path = PATH_PROCESSED_DS / "train"
    elif data_type == DATA_TYPE.TEST:
        path = PATH_PROCESSED_DS / "test"

    return path


# функция осущетсвляет разделение датасета на параметры и целевое значение
def features_target(data):
    # Получаем имена предикторов
    features = data.columns.to_list()
    features.remove("target")

    return data[features], data["target"]


# функия сохраняет модель в файл
def save_model(model):

    PATH_MODEL.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(model, PATH_MODEL / "model.pkl")
        print("Модель успешно сохранена.")
    except PermissionError:
        print(
            f"Ошибка доступа. Убедитесь, что у вас есть права на запись в директорию {PATH_MODEL}."
        )
    except Exception as e:
        print(f"Произошла неизвестная ошибка: {e}")


# функия загружает модель из файла и возвращает ее в случае успеха
def load_model():

    try:
        return joblib.load(PATH_MODEL / "model.pkl")
    except Exception as e:
        print("Ошибка при загрузке модели!\n", e)
        return None


# функия возвращает структуру датасета, на котором происходит обучение модели
def desc_dataset():

    # пример: ("exercise angina", ["no", "yes"]),
    bool_features = []

    # пример: ("ST slope", ["upsloping", "flat", "downsloping"])
    cat_features = []

    # пример: ("age", "in years", 10, 110),
    num_features = []

    return {
        "bool_features": bool_features,
        "cat_features": cat_features,
        "num_features": num_features,
    }
