### Тест для загрузки данных
### проверка функции на корректность работы с различными типами данных.
# №№ Тест проверяет, что данные возвращаются как DataFrame и содержат ожидаемые столбцы.

import pandas as pd
from common_functions import load_dataset, DATA_TYPE


def test_load_dataset_valid():
    # Загружаем данные TRAIN
    data = load_dataset(DATA_TYPE.TRAIN, "DEFAULT")
    assert isinstance(
        data, pd.DataFrame
    ), "Данные должны быть представлены как DataFrame"
    assert not data.empty, "Датасет не должен быть пустым"
