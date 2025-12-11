import pandas as pd #подключаем библиотеку pandas для работы с таблицами
import matplotlib.pyplot as plt#подключаем matplotlib для построения графиков.
import seaborn as sns#библиотека seaborn — для красивой графики и тепловых карт.
from sklearn.preprocessing import StandardScaler#StandardScaler — инструмент для стандартизации признаков.
from sklearn.model_selection import train_test_split#функция для разделения данных на обучающую и тестовую выборку.

df = pd.read_csv("weather_data.csv")   #Мы считываем данные, которые были сформированы в Лабораторной работе №1.


df["date"] = pd.to_datetime(df["date"]) #приводим дату к формату timestamp
numeric_cols = ["tmax", "tmin", "precip", "tmax_lag1", "t_range", "day_of_year", "tmax_next_day"]#
df[numeric_cols] = df[numeric_cols].astype(float)#все признаки переводим в числовой тип float, потому что модели не работают со строками


print("Пропуски в данных:")
print(df.isna().sum()) #выводим количество пропусков в каждой колонке.


df = df.fillna(df.mean(numeric_only=True))#




df[numeric_cols].hist(figsize=(12, 8))
plt.suptitle("Распределения признаков")
plt.show()#строим гистограммы распределений всех числовых признаков.


plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Корреляционная матрица")
plt.show()#вычисляем корреляции между признаками и отображаем в виде тепловой карты.


plt.figure(figsize=(12, 4))
plt.plot(df["date"], df["tmax"])
plt.title("Температура tmax по времени")
plt.xlabel("Дата")
plt.ylabel("Температура")
plt.grid(True)
plt.show()#строим график изменения температуры tmax по датам.


scaler = StandardScaler()#создаём объект стандартизатора.
scaled = scaler.fit_transform(df[numeric_cols])#вычисляем среднее и стандартное отклонение, затем преобразуем данные.

df_scaled = pd.DataFrame(scaled, columns=numeric_cols)#делаем обратно DataFrame из стандартизованных данных.


X = df_scaled.drop("tmax_next_day", axis=1)#признаки (всё кроме целевой переменной).
y = df_scaled["tmax_next_day"]#целевой признак — температура следующего дня.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)#делим данные: 80% — обучение, 20% — тест; без перемешивания (важно для временных рядов).

print("Готово! Данные стандартизированы и разделены.")#сообщение о завершении обработки.
