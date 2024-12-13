from common_functions import normalize_df
import numpy as np
import pandas as pd
from common_functions import load_scor_model, data_path, DATA_TYPE, MODEL_TYPE


def dataset_generation(df_params, num_samples=1000):
    """
    Генерация синтетического датасета. Таргетный скорбал рассчитывается математически,
    на основании весов, заданных экспертом и сохраненных в файле scor_model
    либо default_scor_model (если веса по-умолчанию не менялись)

    Параметры:
    - df_params: датафрейм с критериями и их весами
    - num_samples: количесвто строк для генерации в синтетическом датасете (по умолчанию 1000)
    """

    # если датафрейм пустой, значит у нас нет весов и мы не сможем сгенерировать датасет
    if df_params.empty:
        return -1

    df_params = normalize_df(df_params)

    # Генерация синтетических данных
    synthetic_data = {}
    for index, row in df_params.iterrows():
        feature_values = np.random.randint(
            row["min_value"], row["max_value"] + 1, num_samples
        )
        synthetic_data[row["name"]] = feature_values

    # Создание датафрейма из сгенерированных данных
    df_synthetic = pd.DataFrame(synthetic_data)

    # Функция для расчета целевой переменной
    def calc_target(row):
        return sum(
            row[feature]
            * df_params.loc[df_params["name"] == feature, "weight"].values[0]
            * (df_params.loc[df_params["name"] == feature, "direct_dependence"].values[0] * 2 - 1)
            for feature in df_params["name"]
        )

    # Вычисление целевой переменной и добавление в датафрейм
    df_synthetic["target"] = df_synthetic.apply(calc_target, axis=1)

    return df_synthetic


def main():
    df = load_scor_model()
    data = dataset_generation(df, 1000)
    path = data_path(DATA_TYPE.BASE, MODEL_TYPE.DEFAULT).with_suffix(".csv")
    data.to_csv(path, index=False)


if __name__ == "__main__":
    main()
