from pathlib import Path
from enum import Enum
import joblib
import pandas as pd
import os
import numpy as np


DATA_TYPE = Enum("DATA_TYPE", ["BASE", "TRAIN", "TEST"])
MODEL_TYPE = Enum("MODEL_TYPE", ["DEFAULT", "CUSTOM"])
ROOT_PATH = Path.cwd()


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
    os.makedirs("data", exist_ok=True)
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


# функция осуществляет формирование пути к каталогу по типу датасета
def data_path(data_type, model_type):
    base_path = ROOT_PATH / "data" / \
        ("custom" if model_type == MODEL_TYPE.CUSTOM else "default")
    if data_type == DATA_TYPE.BASE:
        path = base_path / "raw"
    elif data_type == DATA_TYPE.TRAIN:
        path = base_path / "processed" / "train"
    elif data_type == DATA_TYPE.TEST:
        path = base_path / "processed" / "test"

    return path


def model_path(model_type):
    return ROOT_PATH / "models" / \
        ("custom" if model_type == MODEL_TYPE.CUSTOM else "default")


# функция осущетсвляет разделение датасета на параметры и целевое значение
def features_target(data):
    # Получаем имена предикторов
    features = data.columns.to_list()
    features.remove("target")

    return data[features], data["target"]


# функция осуществляет сохранение датасета в файл
def save_dataset(data, data_type, model_type):
    path = data_path(data_type, model_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data.to_csv(path.with_suffix(".csv"), index=False)
    except PermissionError:
        print(
            "Ошибка доступа! Убедитесь, что у вас есть права на запись в директорию {path}."
        )
    except Exception as e:
        print(f"Ошибка при сохранении датасета {data_type} {model_type}!\n", e)


# функция осуществляет загрузку датасета из файла
def load_dataset(data_type, model_type):
    path = data_path(data_type, model_type)
    try:
        data = pd.read_csv(path.with_suffix(".csv"))
        return data
    except Exception as e:
        print(f"Ошибка при загрузке датасета {data_type} {model_type}!\n", e)
        return None


# функция осуществляет сохранение пайплайна обработки параметров в файл
def save_pipeline(pipeline, model_type):
    try:
        path = model_path(model_type)
        # сохраняем pipeline в туже папку, где хранится модель
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, path / "pipeline.pkl")
        print("Пайплайн успешно сохранен.")
    except PermissionError:
        print(
            f"Ошибка доступа. Убедитесь, что у вас есть права на запись в директорию {path}."
        )
    except Exception as e:
        print(f"Произошла неизвестная ошибка: {e}")


# функия сохраняет модель в файл
def save_model(model, model_type):
    path = model_path(model_type)
    path.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(model, path / "model.pkl")
        print("Модель успешно сохранена.")
    except PermissionError:
        print(
            f"Ошибка доступа. Убедитесь, что у вас есть права на запись в директорию {path}."
        )
    except Exception as e:
        print(f"Произошла неизвестная ошибка: {e}")


# функия загружает модель из файла и возвращает ее в случае успеха
def load_model(model_type):
    path = model_path(model_type)
    try:
        return joblib.load(path / "model.pkl")
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
