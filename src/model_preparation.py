from sklearn.ensemble import RandomForestClassifier

from common_functions import (
    features_target,
    load_dataset,
    save_model,
    DATA_TYPE,
)


def main():
    # Загружаем датасет
    data = load_dataset(DATA_TYPE.TRAIN)

    # Получаем имена предикторов и целевого признака
    X, y = features_target(data)

    # Обучаем модель
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Сохраняем модель
    save_model(model)


if __name__ == "__main__":
    main()
