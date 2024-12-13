from sklearn.metrics import accuracy_score

from common_functions import (
    load_dataset,
    load_model,
    features_target,
    DATA_TYPE,
)


def main():

    model = load_model()
    data = load_dataset(DATA_TYPE.TEST)
    X, y = features_target(data)

    # Оценка модели
    predictions = model.predict(X)

    print("Метрики при тестировании модели:")
    print(f"accuracy: {accuracy_score(y, predictions)}")


if __name__ == "__main__":
    main()
