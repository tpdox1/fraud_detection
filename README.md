
# Выявление мошенничества в финансовых транзакциях с помощью модели машинного обучения XGBoost

 Проект представляет собой серверный модуль для выявления мошеннических транзакций. Система реализована в виде backend-компонента и взаимодействует через интерфейс командной строки, что позволяет интеграцию в любые аналитические пайплайны.

---

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

## Быстрый старт

Для активации виртуальной среды необходимо в терминале выполнить команду 

```bash
source venv/bin/activate # для macOS/Linux
```

```bash
venv\Scripts\activate # для Windows
```

Затем необходимо установить необходимые библиотеки из списка зависимостей `requirements.txt`

```bash
pip install -r requirements.txt
```
<!-- 
Запуск MLflow-сервера для просмотра результатов экспериментов

```bash
mlflow ui --host 127.0.0.1 --port 5000
``` -->

---

## Обучение моделей

Путь к входному `.parquet`-файлу должен содержать столбец `isFraud`.

```bash
python train.py \
  --trans data/train_transaction.parquet.gzip \
  --id data/train_identity.parquet.gzip \
  --output_dir models/
```

В результате будут обучены и сохранены три модели:
- `model_transaction.json`
- `model_chain.json`
- `model_final.json`

---

## Предсказания

Для запуска предсказаний модели на новых данных необходимо выполнить следующую команду:

```bash
python predict.py \
  --trans data/test_transaction.parquet.gzip \
  --id data/test_identity.parquet.gzip \
  --model_transaction models/model_transaction.json \
  --model_chain models/model_chain.json \
  --model_final models/model_final.json \
  --output predictions.csv
```

Результат:
- `isFraud_transaction` — предсказание модели `model_transaction`
- `isFraud_chain` — предсказание модели `model_chain`
- `isFraud_final` — итоговое предсказание
- Также сохраняются вероятности (`fraud_score_transaction`, `fraud_score_chain`, `fraud_score_final`)

---

## Обработка данных

В код встроена базовая предобработка:
- Категориальные признаки кодируются через `LabelEncoder`.
- Бинарные (`M1–M9`) переводятся в `0/1`.
- Пропуски заменяются на `-999`.

---

## Трекинг экспериментов

Логирование через MLflow. Интерфейс доступен через команду:

```bash
mlflow ui --backend-store-uri file://$(pwd)/mlruns
```

---

## Системные требования

- Python 3.8+
- pandas
- numpy
- xgboost
- scikit-learn
- mlflow
- hdbscan
- seaborn

---

## Авторизация не требуется
Весь сервис предназначен для запуска в защищённой среде и не имеет веб-интерфейса.