from pathlib import Path
from enum import Enum
import joblib
import pandas as pd


DATA_TYPE = Enum("DATA_TYPE", ["BASE", "TRAIN", "TEST"])
PATH_DATASETS = Path.cwd() / "data"
PATH_BASE_DS = PATH_DATASETS / "raw"
PATH_PROCESSED_DS = PATH_DATASETS / "processed"
PATH_MODEL = Path.cwd() / "models"


# функция осуществляет формирование пути к каталогу по типу датасета
def path_by_type(data_type):

    if data_type == DATA_TYPE.BASE:
        path = PATH_BASE_DS
    elif data_type == DATA_TYPE.TRAIN:
        path = PATH_PROCESSED_DS / "train"
    elif data_type == DATA_TYPE.TEST:
        path = PATH_PROCESSED_DS / "test"

    return path


# функция осуществляет сохранение датасета в файл
def save_dataset(data, data_type, name=""):

    path = path_by_type(data_type) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data.to_csv(path.with_suffix(".csv"), index=False)
    except PermissionError:
        print(
            "Ошибка доступа! Убедитесь, что у вас есть права на запись в директорию {path}."
        )
    except Exception as e:
        print(f"Ошибка при сохранении датасета {name}!\n", e)


# функция осуществляет загрузку датасета из файла
def load_dataset(data_type, name=""):

    path = path_by_type(data_type) / name
    try:
        data = pd.read_csv(path.with_suffix(".csv"))
        return data
    except Exception as e:
        print(f"Ошибка при загрузке датасета {name}!\n", e)
        return None


# функция осуществляет сохранение пайплайна обработки параметров в файл
def save_pipeline(pipeline):

    try:
        # сохраняем pipeline в туже папку, где хранится модель
        PATH_MODEL.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, PATH_MODEL / "pipeline.pkl")
        print("Пайплайн успешно сохранен.")
    except PermissionError:
        print(
            f"Ошибка доступа. Убедитесь, что у вас есть права на запись в директорию {PATH_MODEL}."
        )
    except Exception as e:
        print(f"Произошла неизвестная ошибка: {e}")


# сохраняем осуществляет загрузку pipeline из файла
def load_pipeline():

    try:
        return joblib.load(PATH_MODEL / "pipeline.pkl")
    except Exception as e:
        print("Ошибка при загрузке пайплайна!\n", e)
        return None


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
