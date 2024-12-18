from pathlib import Path
from enum import Enum
import joblib
import pandas as pd
import os


DATA_TYPE = Enum("DATA_TYPE", ["BASE", "TRAIN", "TEST"])
MODEL_TYPE = Enum("MODEL_TYPE", ["DEFAULT", "CUSTOM"])
ROOT_PATH = Path.cwd()


def load_scor_model(model_type):
    """
    Загрузка скормодели (параметры, веса, мин/макс значение, описание)
    Аргументы:
        model_type (str): тип скормодели ("DEFAULT", "CUSTOM")
    Возвращает:
        - data (pd.Dataframe): dataframe с параметрами матмодели
        и None при ошибке загрузки
    """

    base_path = (
        ROOT_PATH
        / "data"
        / ("custom" if model_type == MODEL_TYPE.CUSTOM else "default")
    )
    path = f"{base_path}_scor_model.xlsx"

    try:
        scor_param = pd.read_excel(path, dtype={"weight": float})
        return scor_param
    except Exception as e:
        print(f"Ошибка при загрузке скормодели {model_type}!\n", e)
        return None


def save_scor_model(df):
    """
    Сохранение новых весов параметров, заданных пользователем

    Аргументы:
        df (pd.Dataframe): pandas dataframe с весами критериев
    """

    base_path = ROOT_PATH / "data" / "custom"
    path = f"{base_path}_scor_model.xlsx"

    try:
        os.makedirs("data", exist_ok=True)
        df.to_excel(path, index=False)
    except Exception as e:
        print(f"Ошибка при сохранении скормодели {path}!\n", e)
        return None


def recalculate_values(df_data, df_params):
    """
    Функция возвращает датафрейм с оценкой критериев, измененных с учетом
    direct_dependence каждого критерия. Критерии в датафрейме могут иметь
    либо прямую либо обратную зависимость, между значением критерия и риском.
    Каждый критерий имеет параметр direct_dependence
    - direct_dependence = 1, означает прямую зависмость риска от значения
    - direct_dependence = 0, означает обратную зависимость
    Параметры:
     - df_data (pd.Dataframe): датафрейм с данными
     - df_params (pd.Dataframe): датафрейм с параметрами критериев
     Возвращает:
     - df_recalculated (pd.Dataframe): измененный датафрейм с учетом зависиости
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
    Нормализация веса датафрейма с критериями

    Аргументы:
        df_params (pd.Dataframe): dataframe с параметрами критериев

    Возвращает:
        df_params (pd.Dataframe): dataframe с нормализованными весами
    """

    # Расчитываем совокупный вес всех критериев
    total_weight = df_params.weight.sum()

    # Нормализуем веса
    df_params["weight"] = df_params["weight"] / total_weight

    return df_params


def calculate_scor(df_data, df_params):
    """
    Математический расчет скорбала

    Аргументы:
        df_data (pd.Dataframe): dataframe со значениями критериев
        df_params (pd.Dataframe): dataframe с весами критериев

    Возвращает:
        sum (int): математически рассчитанный бал оценки риска релиза
    """

    # Нормализуем вес критериев
    df_params = normalize_df(df_params)
    # Меняем оценки на противоположные для критериев с обатной зависимостью
    df_data = recalculate_values(df_data, df_params)

    return sum(
        df_data.iloc[0][feature]
        * df_params.loc[df_params["name"] == feature, "weight"].values[0]
        for feature in df_params["name"]
    )


def data_path(data_type, model_type):
    """
    Формирование пути к каталогу по типу датасета

    Аргументы:
        data_type (str): тип данных ("BASE", "TRAIN", "TEST")
        model_type (str): тип модели ("DEFAULT", "CUSTOM")

    Возвращает:
        path (str): путь к каталогу
    """

    base_path = (
        ROOT_PATH
        / "data"
        / ("custom" if model_type == MODEL_TYPE.CUSTOM else "default")
    )
    if data_type == DATA_TYPE.BASE:
        path = base_path / "raw"
    elif data_type == DATA_TYPE.TRAIN:
        path = base_path / "processed" / "train"
    elif data_type == DATA_TYPE.TEST:
        path = base_path / "processed" / "test"

    return path


def model_path(model_type):
    """
    Функция возвращает путь к модели в зависимости от её типа

    Аргументы:
        model_type (str): тип модели ("DEFAULT", "CUSTOM")

    Returns:
        str: путь к модели
    """

    return (
        ROOT_PATH
        / "models"
        / ("custom" if model_type == MODEL_TYPE.CUSTOM else "default")
    )


def features_target(data):
    """
    Разделение датасета на параметры и целевое значение

    Args:
        data (pd.Dataframe): dataframe с данными

    Returns:
     - features: параметры
     - target: целевое значение
    """

    # Получаем имена предикторов
    features = data.columns.to_list()
    features.remove("target")

    return data[features], data["target"]


def save_dataset(data, data_type, model_type):
    """
    Сохранение датасета в файл

    Аргументы:
        data (pd.Dataframe): dataframe с данными
        data_type (str): тип данных ("BASE", "TRAIN", "TEST")
        model_type (str): тип модели ("DEFAULT", "CUSTOM")
    """

    path = data_path(data_type, model_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data.to_csv(path.with_suffix(".csv"), index=False)
        print(f"Датасет {model_type} сохранен: {path}")
    except PermissionError:
        print(
            f"Ошибка доступа! Убедитесь, что у вас есть права на запись в директорию {path}.\n"
        )
    except Exception as e:
        print(f"Ошибка при сохранении датасета {data_type} {model_type}!\n", e)


def load_dataset(data_type, model_type):
    """
    Загрузка датасета из файла

    Аргументы:
        data_type (str): тип данных ("BASE", "TRAIN", "TEST")
        model_type (str): тип модели ("DEFAULT", "CUSTOM")

    Возвращает:
        data (pd.Dataframe): dataframe с данными
    """

    path = data_path(data_type, model_type)
    try:
        data = pd.read_csv(path.with_suffix(".csv"))
        return data
    except Exception as e:
        print(f"Ошибка при загрузке датасета {data_type} {model_type}!\n", e)
        return None


def save_pipeline(pipeline, model_type):
    """
    Сохранение пайплайна обработки параметров в файл

    Аргументы:
        pipeline (Pipeline): пайплайн с предобработкой данных
        model_type (str): тип модели ("DEFAULT", "CUSTOM")
    """

    try:
        path = model_path(model_type)
        # сохраняем pipeline в туже папку, где хранится модель
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, path / "pipeline.pkl")
        print("Пайплайн успешно сохранен.")
    except PermissionError:
        print(
            f"Ошибка доступа. Убедитесь, что у вас есть права на запись в директорию {path}.\n"
        )
    except Exception as e:
        print(f"Произошла неизвестная ошибка: {e}")


# загружаем pipeline из файла
def load_pipeline(model_type):
    try:
        path = model_path(model_type)
        return joblib.load(path / "pipeline.pkl")
    except Exception as e:
        print("Ошибка при загрузке пайплайна!\n", e)
        return None


def save_model(model, model_type):
    """
    Сохранение модели в файл

    Аргументы:
        model : модель
        model_type (str): тип модели ("DEFAULT", "CUSTOM")
    """

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


def load_model(model_type):
    """
    Загрузка модели из файла

    Аргументы:
        model_type (str): тип модели ("DEFAULT", "CUSTOM")

    Возвращает:
        model: модель
    """

    path = model_path(model_type)
    try:
        return joblib.load(path / "model.pkl")
    except Exception as e:
        print("Ошибка при загрузке модели!\n", e)
        return None


def desc_dataset():
    """
    Получение структуры датасета, на котором происходит обучение модели

    Возвращает:
        dict: словарь ключ-значение с ключами:
            bool_features: признаки с булевым значением
            cat_features: категориальные признаки
            num_features: числовые признаки
    """

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
