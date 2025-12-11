import requests    #Подключает библиотеку requests, она нужна для выполнения HTTP-запросов к сайтам
from bs4 import BeautifulSoup     #Импорт класса BeautifulSoup из пакета bs4. Этот класс используется для разбора (парсинга) HTML-страниц
import pandas as pd       #Подключает библиотеку pandas и даёт ей короткое имя pd. Pandas облегчает работу с таблицами
import time      #импортируем модуль time В коде используется функция time.sleep() для задержки между запросами к сайту

def fetch_news(num_news=5):#объявляем функцию которая соберет указанное количество новостей 
    base_url = "https://lenta.ru"#Адрес главной страницы Lenta.ru, с которой начинаем сбор данных.
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    } #Заголовок HTTP-запроса, чтобы сайт воспринимал программу как обычный браузер.

    response = requests.get(base_url, headers=headers, timeout=10)#Загружаем HTML главной страницы. timeout=10 — ограничение по времени.
    if response.status_code != 200: #Если сервер вернул не 200 (успех), значит сайт не открылся.
        return pd.DataFrame() #Возвращаем пустую таблицу — сигнал ошибки.

    soup = BeautifulSoup(response.text, 'html.parser')#Преобразуем HTML-код в объект BeautifulSoup для дальнейшего разбора.
    news_links = extract_news_links(soup, num_news)#Ищем на главной странице ссылки на новости.

    news_data = []#Список, куда будем складывать собранные статьи.
    for i, news in enumerate(news_links[:num_news]):#Перебираем найденные новости, но берём только нужное количество.
        try:
            content = fetch_article_content(news['link'], #headers) Загружаем HTML каждой новости и выделяем текст статьи.
            news_data.append({
                'title': news['title'],
                'link': news['link'],
                'content': content # сохроняе заголовок, сылку и текст статьи в спимок 
            })
            time.sleep(1) # делаем паузу между запросами 
        except:
            pass # если программа не запускается то просто пропускаем 

    return pd.DataFrame(news_data) # превращаем список новостей в таблицу 

def extract_news_links(soup, num_news): #Функция ищет на странице ссылки на отдельные статьи.

    selectors = [
        'a.card-mini__title',
        'a.card-full-news__title',
        'a[href*="/news/"]'#писок CSS-селекторов, по которым мы ищем ссылки на новости
    ]

    links = [] # Пустой список, куда будем добавлять найденные ссылки.
    for selector in selectors: #перебор каждого селектора из списка
        for link in soup.select(selector): #для текущего селектора находит все соответствующие элементы на странице.
            href = link.get('href') #извлекает значение атрибута href (адрес ссылки)
            title = link.get_text(strip=True) #извлекает видимый текст внутри тега (заголовок новости), убирая начальные и конечные пробелы.

            if href and title and len(title) > 10: #пропускает элементы без ссылки или с очень коротким заголовком
                if href.startswith('/'):#проверяет, является ли ссылка относительной
                    href = 'https://lenta.ru' + href #если относительная, добавляет базовую часть, чтобы получить полный URL.

                if href not in [x['link'] for x in links]:#проверяет, что такую ссылку ещё не добавляли
                    links.append({'title': title, 'link': href})#добавляет словарь в список links.

            if len(links) >= num_news * 2:# если собрали запас ссылок (в 2 раза больше желаемого)
                break# выходим из внутреннего цикла по элементам селектора

    return links #возвращает список найденных ссылок

def fetch_article_content(article_url, headers): #бъявляет функцию, которая по article_url скачивает страницу и возвращает текст статьи.
    try: #начинаем блок, чтобы поймать возможные сетевые ошибки.
        response = requests.get(article_url, headers=headers, timeout=10)#делаем запрос на страницу статьи; используем те же заголовки и таймаут 10 сек.
        soup = BeautifulSoup(response.text, 'html.parser')

        content_parts = []#список для накопления фрагментов текста статьи.
        text_selectors = [
            '.topic-body__content',
            '.topic-body__title',
            'p'
        ]

        for selector in text_selectors: #перебираем селекторы по очереди.
            for element in soup.select(selector): #для текущего селектора получаем все соответствующие элементы.
                text = element.get_text(strip=True) #извлекаем чистый текст из элемента
                if text and len(text) > 20:#отбрасываем короткие фрагменты (ниже 20 символов)
                    content_parts.append(text)#добавляем содержательный фрагмент в список
            if content_parts:# если что-то найдено по текущему селектору, то не проверяем остальные селекторы, берём найденный набор фрагментов.
                break

        return ' '.join(content_parts) if content_parts else "Текст не найден" #если были фрагменты, соединяем их в одну строку и возвращаем
    except:#в случае исключения
        return "Ошибка загрузки"

if __name__ == "__main__": #проверка: этот блок выполняется только если файл запущен напрямую
    news_dataframe = fetch_news(3)# просит 3 новости, сохраняет результат в переменную

    if news_dataframe.empty:#проверяет, вернулся ли пустой DataFrame
        print("Не удалось получить новости")#если пустой, печатает сообщение об ошибке
    else:#
        print(f"Получено {len(news_dataframe)} новостей")#печатает, сколько новостей удалось собрать.
        for i, row in news_dataframe.iterrows():# перебирает строки DataFrame
            print(f"\n{i + 1}. {row['title']}")#печатает номер и заголовок статьи.
            print(f"Ссылка: {row['link']}")#печатает URL статьи.
            print(f"Текст: {row['content'][:120]}...")#печатает первые 120 символов текста

        news_dataframe.to_csv('lenta_news.csv', index=False, encoding='utf-8')#сохраняет таблицу в CSV файл
        print("\nСохранено в lenta_news.csv")#сообщает, что файл сохранён.


 
