# MLops_2-final_task
Практическое задание №4 по дисциплине "Автоматизация машинного обучения"(MLops). 
В качестве инструмента выполненения задачи выбран **ClearML**

## Реализация
* Поставленная задача - достижение лучшей метрики F1_score в Kaggle соревновании [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/overview) путем подбора гиперпараметров модели
* Модель - [Метод опорных векторов (SVC)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) реализации scikit-learn
* Для трекинга ML-экспериментов использован [ClearML](https://app.clear.ml/settings/workspace-configuration)
* Основной скрипт `main.py` содержит код предобработки данных и обучения модели с гиперпараметрами, описанными в коде под переменной `parameters` 
* Для подбора гиперпараметров модели `С`, `gamma` и `kernel` использовался **HyperParameterOptimizer** в скрипте `optimize.py`

## Состав команды
* Семерикова Ксения (РИМ-220907)
* Фарафонов Данил (РИМ-220907)
