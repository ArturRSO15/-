import pandas as pd  # Работа с таблицами
from sklearn.model_selection import train_test_split  # Деление на train/test
from sklearn.tree import DecisionTreeClassifier  # Модель дерева решений
from sklearn.metrics import accuracy_score  # Метрика точности
from collections import Counter  # Подсчет классов
from imblearn.over_sampling import SMOTE, ADASYN  # Методы oversampling
from imblearn.under_sampling import TomekLinks  # Метод undersampling

df = pd.read_csv("weather_data.csv")  # Загружаем данные

# Создаем столбец target:
# 1 — если температура следующего дня выше медианы, 0 — если ниже или равна
df["target"] = (df["tmax_next_day"] > df["tmax_next_day"].median()).astype(int)

X = df.drop(["target", "date"], axis=1)  # Признаки (без target и даты)
y = df["target"]  # Целевая переменная

# Разделяем данные: 80% train, 20% test, без перемешивания (важно для временных данных)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

base_acc = accuracy_score(y_test, DecisionTreeClassifier().fit(X_train, y_train).predict(X_test))  # Базовая точность
print("1) Базовая accuracy:", base_acc)

counts = Counter(y)  # Считаем количество каждого класса
majority = max(counts, key=counts.get)  # Класс большинства
minority = min(counts, key=counts.get)  # Класс меньшинства
keep_n = int(counts[majority] * 0.1)  # Оставляем только 10% объектов большого класса

df_major = df[df["target"] == majority].sample(keep_n, random_state=42)  # 10% большого класса
df_minor = df[df["target"] == minority]  # Все объекты малого класса

# Создаем новый датасет с искусственным дисбалансом
df_imbal = pd.concat([df_major, df_minor], ignore_index=True)

X2 = df_imbal.drop(["target", "date"], axis=1)  # Признаки дисбалансного набора
y2 = df_imbal["target"]  # Target дисбалансного набора

# Делим дисбалансный набор
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, shuffle=False)

imbal_acc = accuracy_score(y2_test, DecisionTreeClassifier().fit(X2_train, y2_train).predict(X2_test))
# Модель работает хуже, потому что данные дисбалансные
print("2) Accuracy после создания дисбаланса:", imbal_acc)

methods = {
    "SMOTE": SMOTE(),  # Создание синтетических данных меньшего класса
    "ADASYN": ADASYN(),  # Адаптивное создание новых точек
    "TomekLinks": TomekLinks()  # Удаление шумовых точек
}

for name, method in methods.items():  # Перебираем методы балансировки
    X_res, y_res = method.fit_resample(X_train, y_train)  # Балансируем тренировочные данные

    # Обучаем модель и считаем точность после балансировки
    acc = accuracy_score(y_test, DecisionTreeClassifier().fit(X_res, y_res).predict(X_test))

    print(f"3) {name}: accuracy = {acc}")  # Выводим точность каждого метода
