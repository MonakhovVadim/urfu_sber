from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score, 
    root_mean_squared_error
)

from common_functions import (
    load_dataset,
    load_model,
    features_target,
    DATA_TYPE,
    MODEL_TYPE
)

def test_model(model_type):
    model = load_model(model_type)
    data = load_dataset(DATA_TYPE.TEST, model_type)
    X, y = features_target(data)    

    # Оценка модели
    predictions = model.predict(X)
    #print("y", y)
    #print("predictions", predictions)
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = root_mean_squared_error(y, predictions)

    print("Метрики при тестировании модели:")
    print(f"r2: {r2}")
    print(f"mae: {mae}")
    print(f"mse: {mse}")
    return [
        model.__class__.__name__,
        {
            "R2": r2,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse
        },
    ]


def main():
    test_model(MODEL_TYPE.DEFAULT)


if __name__ == "__main__":
    main()
