### Тест на обработку пропусков:

import pandas as pd
import pytest
from unittest.mock import patch
from main import train_model


@patch("common_functions.load_dataset")
def test_train_model_with_nan(mock_load_dataset):
    # Данные с пропусками
    mock_load_dataset.return_value = pd.DataFrame(
        {
            "feature1": [1.0, None, 3.0],
            "feature2": [4.0, 5.0, 6.0],
            "target": [0, 1, 0],
        }
    )

    # Ожидаем выброс ошибки
    with pytest.raises(
        ValueError, match="В тренировочном датасете присутствуют пропуски."
    ):
        train_model("DEFAULT")
