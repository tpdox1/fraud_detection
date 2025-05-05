
# Выявление мошенничества в финансовых транзакциях с помощью модели машинного обучения XGBoost

 Проект представляет собой серверный модуль для выявления мошеннических транзакций. Система реализована в виде backend-компонента и взаимодействует через интерфейс командной строки, что позволяет интеграцию в любые аналитические пайплайны.

## Структура проекта

```
fraud_backend_project/
├── data/                               # Данные для обучения и тестирования
    ├── test_identity.parquet.gzip      # Таблица идентификаторов для тестирования
    ├── test_transaction.parquet.gzip   # Таблица транзакций для тестирования
    ├── train_identity.parquet.gzip     # Таблица идентификаторов для обучения
    └── train_transaction.parquet.gzip  # Таблица транзакций для обучения 
├── mlruns/                             # Локальное сохранение экспериментов MLFlow   
├── models/                             # Обученные модели
│   ├── model_transaction.json          # Модель по отдельным транзакциям
│   ├── model_chain.json                # Модель по цепочкам транзакций
│   └── model_final.json                # Финальная модель
├── notebooks/                          # Jupiter-notebooks
│   └── fraud_analysis.ipynb            # Разведочный анализ и построение моделей
├── src/                                # Вспомогательные модули
│   └── module.py                       # Модуль с функциями обработки, формирования цепочек и кодирования
├── venv/                               # Виртуальное окружение
├── train.py                            # Обучение всех трёх моделей
├── predict.py                          # Предсказание всех трёх моделей
├── requirements.txt                    # Python зависимости
└── README.md                           # Инструкция для запуска
```

---

## Настройка окружения для проекта


1. Проверить установку в системе языка программирования Python и проектного менеджера pip
    ```bash
    python --version
    pip --version
    ```

2. Создать виртуальное окружение

    ```bash
    python -m venv venv
    ```

3. Активировать виртуальное окружение

    ```bash
    source venv/bin/activate # для macOS/Linux
    ```

    либо

    ```bash
    venv\Scripts\activate # для Windows
    ```

4. Установить зависимости из списка зависимостей `requirements.txt`

    ```bash
    pip install -r requirements.txt
    ```


## Запуск обучения моделей

Запуск обучения возможен при помощи команды

```bash
python train.py \
  --trans data/train_transaction.parquet.gzip \
  --id data/train_identity.parquet.gzip \
  --output_dir models/
```

Где:
-  `data/train_transaction.parquet.gzip` - путь к файлу с информацией о транзакциях, обязательно содержащий столбец 'isFraud', для обучающей выборки
-  `data/train_identity.parquet.gzip` - путь к файлу с идентификационными данными для обучающей выборки

В результате будут обучены и сохранены три модели:
- `model_transaction.json` - модель по отдельным транзакциям
- `model_chain.json` - модель по цепочкам транзакций
- `model_final.json` - финальная модель

## Запуск предсказания на тестовых данных

Запуск предсказания возможен при помощи команды

```bash
python predict.py \
  --trans data/test_transaction.parquet.gzip \
  --id data/test_identity.parquet.gzip \
  --model_transaction models/model_transaction.json \
  --model_chain models/model_chain.json \
  --model_final models/model_final.json \
  --output predictions.csv
```
Где:
- `data/test_transaction.parquet.gzip` - путь к файлу с информацией о транзакциях для тестовой выборки
- `data/test_identity.parquet.gzip` - путь к файлу с идентификационными данными для тестовой выборки
- `models/model_transaction.json` - путь к обученной модели для отдельных транзакций
- `models/model_chain.json` - путь к обученной модели для цепочек транзакций
- `models/model_final.json` - путь к финальной обученной модели
- `predictions.csv` - путь к итоговому файлу с предсказаниями

Файл `predictions.csv` содержит следующие столбцы:
- `isFraud_transaction` — предсказание модели `model_transaction`
- `fraud_score_transaction` - вероятность модели `model_transaction`
- `isFraud_chain` — предсказание модели `model_chain`
- `fraud_score_chain` - вероятность модели `model_chain`
- `isFraud_final` — итоговое предсказание
- `fraud_score_final` - вероятность модели `model_final`

## Обработка данных

В файле module.py содержатся функции базовой предобработки данных для их последующей загрузки в модели:

- `preprocess(df)` — извлекает календарные признаки (`день`, `месяц`, `час`, `часть суток`) и формирует `user_id` на основе `card1–card6`;
- `build_chains(df)` — объединяет транзакции одного пользователя в поведенческие цепочки с помощью кластеризации `HDBSCAN`;
- `aggregate_chains(df)` — вычисляет агрегированные данные по каждой цепочке;
- `encode_and_merge(df, chain_df)` — кодирует категориальные признаки с помощью `LabelEncoder`, бинарные по схеме `'T' -> 1`, `'F' -> 0`, `NaN -> -999` и объединяет с агрегированными признаками.


## Отслеживание экспериментов

Логирование информации о параметрах и метриках каждой модели реализовано в среде MLflow. 

Запуск интерфейса MlFlow:

    ```bash
    mlflow ui --backend-store-uri file://$(pwd)/mlruns
    ```

Интерфейс откроется по адресу по адресу http://localhost:5000

## Системные требования

- Python 3.8+
- pandas
- numpy
- xgboost
- scikit-learn
- mlflow
- hdbscan
- seaborn
