from sklearn.metrics import r2_score

from urfu_sber.src.common_functions import (
    load_dataset,
    load_model,
    features_target,
    DATA_TYPE,
)


def test_model():
    model = load_model()
    data = load_dataset(DATA_TYPE.TEST)
    X, y = features_target(data)

    # Оценка модели
    predictions = model.predict(X)
    test_score = r2_score(y, predictions)

    assert test_score > 0.93, f"Failed model test, f1 score on test data: {test_score}"
