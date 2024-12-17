### Проверка, что функция корректно сохраняет модель. 
#№№ Делаем создавая временный файл.

import os
import tempfile
from sklearn.linear_model import LinearRegression
from common_functions import save_model

def test_save_model():
    # Создаем временное хранилище
    model = LinearRegression()
    model_type = "DEFAULT"
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, f"{model_type}_model.sav")
        save_model(model, model_type)
        
        # Проверяем существование сохраненного файла
        assert os.path.exists(save_path), "Файл модели должен существовать"
