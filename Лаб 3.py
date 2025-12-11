import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---- ЗАГРУЗКА ДАННЫХ ИЗ ЛР1 ----
df = pd.read_csv("weather_data.csv")   # замените на свой файл

# ---- 1. ПРОВЕРКА ТИПОВ И ПРЕОБРАЗОВАНИЕ ----
df["date"] = pd.to_datetime(df["date"])
numeric_cols = ["tmax", "tmin", "precip", "tmax_lag1", "t_range", "day_of_year", "tmax_next_day"]
df[numeric_cols] = df[numeric_cols].astype(float)

# ---- 2. ПРОВЕРКА ПРОПУСКОВ ----
print("Пропуски в данных:")
print(df.isna().sum())

# Заполнение пропусков средним значением
df = df.fillna(df.mean(numeric_only=True))

# ---- 3. ВИЗУАЛИЗАЦИИ ----

# 3.1. распределения признаков
df[numeric_cols].hist(figsize=(12, 8))
plt.suptitle("Распределения признаков")
plt.show()

# 3.2. корреляционная матрица
plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Корреляционная матрица")
plt.show()

# 3.3. график tmax по времени
plt.figure(figsize=(12, 4))
plt.plot(df["date"], df["tmax"])
plt.title("Температура tmax по времени")
plt.xlabel("Дата")
plt.ylabel("Температура")
plt.grid(True)
plt.show()

# ---- 4. NORMALIZATION / STANDARDIZATION ----
scaler = StandardScaler()
scaled = scaler.fit_transform(df[numeric_cols])

df_scaled = pd.DataFrame(scaled, columns=numeric_cols)

# ---- 5. SPLIT DATA ----
X = df_scaled.drop("tmax_next_day", axis=1)
y = df_scaled["tmax_next_day"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print("Готово! Данные стандартизированы и разделены.")
