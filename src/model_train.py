# модуль обучения модели
import pandas as pd


def ds_preprocessing(ds, required_columns):
    """
    Загрузка датасета из файла, предобработка данных
    Параметры:
    - ds - файл с данными
    - required_columns - колонки, которые должны быть в датасете
    Возвращает: 
    - Pandas датафрейм / или ошибки
    Исключения:
     - ValueError: Если файл имеет неправильный формат.
    """

    # по расширению, определяем какую функцию загрузки датафрейма использовать
    if ds.name.endswith(('.xls', '.xlsx')):
        data = pd.read_excel(ds)
    elif ds.name.endswith('.csv'):
        data = pd.read_csv(ds)
    else:
        data = pd.DataFrame()

    # Проверка на пустоту
    if data.empty:
        raise ValueError("Файл пустой.")
    
    # Проверка наличия необходимых колонок
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Отсутствуют необходимые колонки: {', '.join(missing_columns)}")


    # ЗДЕСЬ БЛОК ДАЛЬНЕЙШЕЙ ПРЕДОБРАБОТКИ ДАННЫХ (ПРОВЕРКА ПРОПУСКОВ, СТАТИСТИК, КОЛИЧЕСТВА СТРОК И ТП)
    
    
    return data


def train_model(ds, required_columns):
    """
    Обучение модели
    Параметры:
    - ds - файл с данными
    - required_columns - колонки, которые должны быть в датасете
    Возвращает: 
    - model - обученная модель 
    - metrics - метрики обученнной модели
    """  
    
    # загружаем датасет в датафрем и проводим его предобработку
    df = ds_preprocessing(ds, required_columns)
    
    # ЗДЕСЬ КОД ОБУЧЕНИЯ МОДЕЛИ, ПРОВЕРКИ МЕТРИК И ТП
    
    
    
    # сохрание модели на диск
    # save_model(model)
    model = 'The best model!'
    # это болванка для проверки интерфейса! УДАЛИТЬ ПОСЛЕ РЕАЛИЗАЦИИ 
    return [model, {"Средняя абсолютная ошибка (MAE)": 1.34,
            "Среднеквадратичная ошибка (MSE)": 0.96,
            "Корень из среднеквадратичной ошибки (RMSE)": 0.97}]  