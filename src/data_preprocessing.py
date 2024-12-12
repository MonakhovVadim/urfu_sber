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
)


def main():

    # Загружаем базовый датасет
    data = load_dataset(DATA_TYPE.BASE, "dataset")
    X, y = features_target(data)
    print(X.describe())

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
    save_dataset(data_train, DATA_TYPE.TRAIN)

    # Сохраняем датасет для тестирования
    save_dataset(data_test, DATA_TYPE.TEST)

    # Сохраняем пайплайн для обработки "сырых" данных, которые будут вводить пользователи
    save_pipeline(pipeline)


if __name__ == "__main__":
    main()
