import numpy as np
import pandas as pd
from common_functions import (
    load_scor_model,
    save_dataset,
    normalize_df,
    recalculate_values,
    DATA_TYPE,
    MODEL_TYPE,
)


def dataset_generation(df_params, num_samples=1000):
    """
    Генерация синтетического датасета. Таргетный скорбал рассчитывается математически,
    на основании весов, заданных экспертом и сохраненных в файле scor_model
    либо default_scor_model (если веса по-умолчанию не менялись)

    Параметры:
    - df_params: датафрейм с критериями и их весами
    - num_samples: количесвто строк для генерации датасета
    """

    # если датафрейм пустой, значит у нас нет весов и мы не сможем сгенерировать датасет
    if df_params.empty:
        return -1

    df_params = normalize_df(df_params)

    # Генерация синтетических данных
    synthetic_data = {}
    for _, row in df_params.iterrows():
        feature_values = np.random.randint(
            row["min_value"], row["max_value"] + 1, num_samples
        )
        synthetic_data[row["name"]] = feature_values

    # Создание датафрейма из сгенерированных данных
    df_synthetic = pd.DataFrame(synthetic_data)

    # учитываем direct_dependece в оценках критериев
    df_recalculated = recalculate_values(df_synthetic, df_params)

    # Функция для расчета целевой переменной
    def calc_target(row):
        return sum(
            row[feature]
            * df_params.loc[df_params["name"] == feature, "weight"].values[0]
            for feature in df_params["name"]
        )

    # Вычисление целевой переменной в пересчитанном датафрейме
    df_recalculated["target"] = df_recalculated.apply(calc_target, axis=1)
    df_synthetic["target"] = df_recalculated["target"]

    return df_synthetic


def main():
    df = load_scor_model(MODEL_TYPE.DEFAULT)
    data = dataset_generation(df, 1000)
    save_dataset(data, DATA_TYPE.BASE, MODEL_TYPE.DEFAULT)


if __name__ == "__main__":
    main()
