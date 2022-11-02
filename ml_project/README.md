homework1
==============================

HW1 Technopark MLOps

## Установка
1. clone repo
2. `pip install -e .`

## Инструкция по запуску обучения.

1. Кладем датасет в папку data/raw
2. Правим конфиг в train_config.yaml. Указываем необходимые параметры для запуска
3. Вводим в корневой папке `make train`

## Инструкция по использованию готовых моделей.

1. Кладем датасет в папку data/raw
2. Правим конфиг в train_config.yaml. Указываем параметры для запуска
3. Вводим в корневой папке `make predict`

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    |── configs
    |   |── predict_config.yaml
    |   └── train_config.yaml
    |
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models
        |── predictions    <- Model predictions
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    |   |── enties         <- Enties of project (dataclasses)
    |   |
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    |   |   
    |   |── tests          <- Tests
    |   |
    |   |── predict_pipeline.py
    |   |
    |   |── train_pipeline.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

