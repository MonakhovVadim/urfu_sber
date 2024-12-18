# Проектный практикум - задача от ПАО Сбербанк. 
## Группа 19

## Описание задачи

Разработать алгоритм (модель, методику) и программу его реализующую для оценки технологического риска внедрений релизов программного обеспечения, предназначенного для выполнения процедур трансформации данных (ETL) и расчетов на витринах хранилища данных (DWH).

Для реализации проекта использованы такие технологии, как:
- Docker для контейнеризации;
- streamlit для пользовательского интерфейса;
- GitHub Actions для тестов.
  
## Состав команды и роли:
 - Вадим Монахов (teamlead)
 - Евгений Брылин (ML engineer, developer)
 - Олег Перевиспа (frontend developer)
 - Игорь Ерошин (QA engineer)
 - Татьяна Меркурьева (DEVOps)
 - Клим Колчин (technical writer)


## Архитектура приложения

Веб-приложение написано на языке python. Интерфейс реализован с помощью библиотеки Streamlit. Приложение использует функции из файлов data_preprocessing.py, model_preparation.py и model_testing.py для предобработки данных, тренировки модели и расчета метрик модели.
Интерфейс представлен следующими страницами: 
- определение скорбала (файл app.py)
- настройка скоркарты, генерация датасета (ds_generation.py)
- обучение модели (upload_train.py)
- версии приложения (versions.py)

Общие функции приложения вынесены в отдельный python файл common_functions.py. 

Обучение ML модели реализовано в виде набора python файлов, отвечающих за следующие части процесса:
- предобработка данных (data_preprocessing.py)
- обучение модели (model_preparation.py)
- тестирование модели (model_testing.py)

Также реализованы автотесты модели в файле test_model.py.
Приложение работает в Docker контейнере. В процессе сборки Docker контейнера реализован pipeline автоматического обучения модели на синтетическом датасете, который формируется на основании дефолтных критериев и их весов (data_generation.py). 



## Структура репозитория

```plaintext
├── data                                  # Данные
│   └── default_scor_model.xlsx        # Дефолтная скормодель
├── docs                                  # Документация
│   ├── README.md
│   ├── USAGE_README.md
│   └── app_screenshot.png
├── src
│   ├── pages
│   │   ├── __init__.py
│   │   ├── ds_generation.py     # Интерфейс генератора синтетического датасета
│   │   ├── upload_train.py      # Интерфейс загрузки датасета и запуск процедуры обучения модели
│   │   └── version.py           # Описание версий приложения
│   ├── __init__.py
│   ├── app.py                         # Главная страница веб-приложения
│   ├── common_functions.py            # Основные функции для использования в других скриптах
│   ├── data_generation.py             # Генерация данных для синтетического датасета
│   ├── data_preprocessing.py          # Обработка данных
│   ├── model_preparation.py           # Инициализация и обучение модели
│   └── model_testing.py               # Оценка модели
├── tests
│   ├── __init__.py
│   └── test_model.py                  # Автотесты
├── Dockerfile
├── README.md
├── __init__.py
├── docker-compose.yaml
└── requirements.txt
```

## Использование приоложения 
Ознакомиться с приложением можно по ссылке: http://80.234.33.45:8501/

## Установка приложения 
Проект реализован как микросервис и запускается в Docker-контейнере, поэтому для развертывания приложения потребуется Docker и Docker compose, с установкой которых можно ознакомиться в [этой документации](https://docs.docker.com/). Также для клонирования репозитария потребуется git.

Шаги по установке и запуску проекта:

1. Клонирование репозитория:
   ```bash
   git clone https://github.com/MonakhovVadim/urfu_sber.git
   cd urfu_sber/
   ```

2. Сборка образа и запуск контейнера::
   ```bash
   docker compose up -d --build
   ```

После запуска сервис будет доступен в браузере по адресу ```http://localhost:8501/```. Для доступа к этому адресу порт ```8501``` должен быть открыт.
