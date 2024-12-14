from sklearn.linear_model import LinearRegression

from common_functions import (
    features_target,
    load_dataset,
    save_model,
    DATA_TYPE,
    MODEL_TYPE,
)


def train_model(model_type):
    """
    Обучение модели

    Аргументы:
        model_type (str): тип модели ("DEFAULT", "CUSTOM")
    """

    # Загружаем датасет
    data = load_dataset(DATA_TYPE.TRAIN, model_type)
    if data is not None:
        # Получаем имена предикторов и целевого признака
        X, y = features_target(data)

        # TODO использовать модель получше
        # Обучаем модель
        model = LinearRegression()
        model.fit(X, y)

        # Сохраняем модель
        save_model(model, model_type)


def main():
    train_model(MODEL_TYPE.DEFAULT)


if __name__ == "__main__":
    main()
