import requests #Используется для отправки НТТР-запрос к интернет API. Нам нужно обращаться к открытому погодному API Open-Meteo,
import pandas as pd #Импортируем pandas, чтобы удобно работать с табличными данными как с DataFrame.Именно в DataFrame мы будем хранить наш погодный датасет.
from datetime import date, timedelta  #Импортируем функции для работы с датами
from sklearn.model_selection import train_test_split  # Функция для разделения данных на обучающую и тестовую выборки.
from sklearn.linear_model import LinearRegression     # Подключаем модель линейной регрессии, которую будем обучать.
from sklearn.metrics import mean_squared_error        # Метрика ошибки MSE — мы используем её для вычисления RMSE (корень из средней квадратичной ошибки).
import numpy as np# числовые операции

latitude = 56.9496    
longitude = 24.1052  # координаты участка земли чью историю погоды мы хоти получить 
end_date = date.today() - timedelta(days=3)  # берем данные до 3 дней назад (данные могут быть не загружены)
start_date = end_date - timedelta(days=365*2)  #берем данные последних двух лет 
timezone = "Europe/Moscow"#Указываем тайм зону 

base_url = "https://archive-api.open-meteo.com/v1/archive" #Адрес API, которое предоставляет исторические погодные данные.
params = { # Мы формируем словарь параметров запроса 
    "latitude": latitude,# координаты 
    "longitude": longitude,# координаты 
    "start_date": start_date.isoformat(),# период запроса 
    "end_date": end_date.isoformat(),# период запроса 
    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",# запрашиваем дневные параметры макс,мин,сумарные осадки 
    "timezone": timezone # чтобы даты были в человека понятном формате 
}

resp = requests.get(base_url, params=params) # выполняем запрос (GET-запрос к API)
resp.raise_for_status()  # выводит ошибку если программа не равна 200
data = resp.json() #Преобразуем ответ сервера из формата JSON в обычный Python-словарь.

daily = data.get("daily", {}) #Получаем раздел "daily" из ответа API. Если его нет — получим пустой словарь.
df = pd.DataFrame({ # создаем таблицу с колонками 
    "date": pd.to_datetime(daily.get("time")),# время 
    "tmax": daily.get("temperature_2m_max"),# макс температура
    "tmin": daily.get("temperature_2m_min"),# мин температура 
    "precip": daily.get("precipitation_sum")#  осадки 
})

# Сортируем по дате и индексируем
df = df.sort_values("date").reset_index(drop=True)

df["tmax_next_day"] = df["tmax"].shift(-1)  #создаем новую колонку значение максимальной колонки 
df = df.dropna(subset=["tmax_next_day"]).reset_index(drop=True)#Удаляем последнюю строку, где не было данных следующего дня.

df["tmax_lag1"] = df["tmax"].shift(1)  # максимальная погода вчера 
df["tmax_lag1"].fillna(df["tmax"].mean(), inplace=True)  # Для первой строки лаг пустой — заменяем средним значением температуры.
df["t_range"] = df["tmax"] - df["tmin"]# суточный диапозон температуры 
df["day_of_year"] = df["date"].dt.dayofyear# получаем номер дня в году 

feature_cols = ["tmax", "tmin", "precip", "tmax_lag1", "t_range", "day_of_year"]#Список признаков, которые будут входить в модель.
X = df[feature_cols].astype(float)#Матрица признаков X — таблица из 6 колонок.
y = df["tmax_next_day"].astype(float)#Целевая переменная — температура следующего дня.


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # 80% → обучение 
)

model = LinearRegression()#создаем объект модели 
model.fit(X_train, y_train)#обучаем модель на исторических данных 
y_pred = model.predict(X_test)#Получаем предсказания температуры на тестовом наборе.
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) #Вычисляем корень из среднеквадратичной ошибки (RMSE) — популярная метрика в задачах регрессии.

print(f"Данные с {start_date.isoformat()} по {end_date.isoformat()} для локации ({latitude},{longitude})")# Сообщаем пользователю, за какой период загружены данные.
print(f"Размер датасета: {len(df)} строк")#Печатаем, сколько строк данных получили из API.
print(f"RMSE на тесте: {rmse:.3f} °C")#Печатаем ошибку модели в градусах Цельсия.

result = X_test.copy()#Копируем тестовые признаки.
result["actual_tmax_next_day"] = y_test.values#добовляем в таблицу значения температуры 
result["pred_tmax_next_day"] = y_pred# предсказанные моделью 
print("\nПримеры (последние 5 строк теста):")#
print(result.tail(5))#
