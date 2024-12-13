import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from common_functions import (
    load_dataset,
    save_dataset,
    save_pipeline,
    features_target,
    DATA_TYPE,
    MODEL_TYPE,
)


def load_data(file):
    """
    Загрузка датасета из файла

    Аргументы:
        file : файл с данными

    Returns:
        pd.Dataframe: датафрейм с данными
    """

    # по расширению, определяем какую функцию загрузки датафрейма использовать
    if file.name.endswith((".xls", ".xlsx")):
        return pd.read_excel(file)
    elif file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.DataFrame()


def validate_data(data, required_columns=[]):
    """
    Валидация данных

    Аргументы:
        data (pd.Dataframe): dataframe с данными
        required_columns (list, optional): колонки, которые должны быть в датасете. По умолчанию пустой список.

    Поднимает:
        ValueError: если файл пустой.
        ValueError: если отсутствуют необходимые колонки
    """

    # Проверка на пустоту
    if data.empty:
        raise ValueError("Файл пустой.")

    # Проверка наличия необходимых колонок
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            f"Отсутствуют необходимые колонки: {', '.join(missing_columns)}"
        )


def preprocess_data(data, model_type):
    """
    Предварительная обработка данных

    Аргументы:
        data (pd.Dataframe): dataframe с данными
        model_type (str): тип модели ("DEFAULT", "CUSTOM")
    """

    X, y = features_target(data)
    # print(X.describe())

    # Избавляемся от дубликатов
    len_before_drop_publicates = len(data)
    data.drop_duplicates(inplace=True)
    count_dublicates = len_before_drop_publicates - len(data)
    if count_dublicates > 0:
        print(f"Дубликаты удалены из датасета в количестве:{count_dublicates}")

    # Создаем пайплайн с предобработкой
    pipeline = Pipeline([("scaler", StandardScaler())])

    # Делаем преобразования
    data = pd.DataFrame(pipeline.fit_transform(X), columns=X.columns)
    data[y.name] = y

    # Делим на тренировочные и тестовые данные
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)

    # Сохраняем датасет для тренировки
    save_dataset(data_train, DATA_TYPE.TRAIN, model_type)

    # Сохраняем датасет для тестирования
    save_dataset(data_test, DATA_TYPE.TEST, model_type)

    # Сохраняем пайплайн для обработки "сырых" данных, которые будут вводить пользователи
    save_pipeline(pipeline, model_type)


def main():
    # Загружаем базовый датасет
    data = load_dataset(DATA_TYPE.BASE, MODEL_TYPE.DEFAULT)

    # проверяем данный
    validate_data(data)

    # предобрабатываем данные
    preprocess_data(data, MODEL_TYPE.DEFAULT)


if __name__ == "__main__":
    main()
