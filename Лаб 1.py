import requests
import pandas as pd             # для работы с табличными данными
from datetime import date, timedelta  # для формирования дат
from sklearn.model_selection import train_test_split  # разбиение на обуч./тест
from sklearn.linear_model import LinearRegression     # простая модель
from sklearn.metrics import mean_squared_error        # метрика RMSE
import numpy as np              # числовые операции

latitude = 56.9496   # пример: Рига (Lat)
longitude = 24.1052  # пример: Рига (Lon)
end_date = date.today() - timedelta(days=3)  # берем данные до 3 дней назад (защита от задержек)
start_date = end_date - timedelta(days=365*2)  # последние 2 года данных
timezone = "Europe/Riga"  # удобно для меток времени

base_url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": start_date.isoformat(),
    "end_date": end_date.isoformat(),
    # Запрашиваем дневные переменные: max temp, min temp, суммарные осадки
    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
    "timezone": timezone
}

# Выполняем запрос и проверяем ответ
resp = requests.get(base_url, params=params)
resp.raise_for_status()  # бросит ошибку, если статус != 200
data = resp.json()

daily = data.get("daily", {})
df = pd.DataFrame({
    "date": pd.to_datetime(daily.get("time")),
    "tmax": daily.get("temperature_2m_max"),
    "tmin": daily.get("temperature_2m_min"),
    "precip": daily.get("precipitation_sum")
})

# Сортируем по дате и индексируем
df = df.sort_values("date").reset_index(drop=True)

df["tmax_next_day"] = df["tmax"].shift(-1)  # сдвиг вверх: значение следующего дня
# Удаляем последнюю строку, где tmax_next_day == NaN
df = df.dropna(subset=["tmax_next_day"]).reset_index(drop=True)

df["tmax_lag1"] = df["tmax"].shift(1)  # вчерашняя tmax
df["tmax_lag1"].fillna(df["tmax"].mean(), inplace=True)  # заполним NaN средним (крайний простейший подход)
df["t_range"] = df["tmax"] - df["tmin"]
df["day_of_year"] = df["date"].dt.dayofyear

feature_cols = ["tmax", "tmin", "precip", "tmax_lag1", "t_range", "day_of_year"]
X = df[feature_cols].astype(float)
y = df["tmax_next_day"].astype(float)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # shuffle=False, т.к. временной ряд (предпочтительно)
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Данные с {start_date.isoformat()} по {end_date.isoformat()} для локации ({latitude},{longitude})")
print(f"Размер датасета: {len(df)} строк")
print(f"RMSE на тесте: {rmse:.3f} °C")

result = X_test.copy()
result["actual_tmax_next_day"] = y_test.values
result["pred_tmax_next_day"] = y_pred
print("\nПримеры (последние 5 строк теста):")
print(result.tail(5))
