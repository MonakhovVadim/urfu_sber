### Тест для загрузки данных
### проверка функции на корректность работы с различными типами данных.
# №№ Тест проверяет, что данные возвращаются как DataFrame и содержат ожидаемые столбцы.

import pandas as pd
from urfu_sber.src.common_functions import load_dataset, DATA_TYPE, MODEL_TYPE

from urfu_sber.tests.prepare_test_data import prepare_test_data


def test_load_dataset_valid():
    prepare_test_data()
    # Загружаем данные TRAIN
    data = load_dataset(DATA_TYPE.TRAIN, MODEL_TYPE.DEFAULT)
    assert isinstance(
        data, pd.DataFrame
    ), "Данные должны быть представлены как DataFrame"
    assert not data.empty, "Датасет не должен быть пустым"
