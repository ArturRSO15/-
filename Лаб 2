import requests    #Подключает библиотеку requests, она нужна для выполнения HTTP-запросов к сайтам
from bs4 import BeautifulSoup     #Импорт класса BeautifulSoup из пакета bs4. Этот класс используется для разбора (парсинга) HTML-страниц
import pandas as pd       #Подключает библиотеку pandas и даёт ей короткое имя pd. Pandas облегчает работу с таблицами
import time      #импортируем модуль time В коде используется функция time.sleep() для задержки между запросами к сайту

def fetch_news(num_news=5):#
    base_url = "https://lenta.ru"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    response = requests.get(base_url, headers=headers, timeout=10)
    if response.status_code != 200:
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, 'html.parser')
    news_links = extract_news_links(soup, num_news)

    news_data = []
    for i, news in enumerate(news_links[:num_news]):
        try:
            content = fetch_article_content(news['link'], headers)
            news_data.append({
                'title': news['title'],
                'link': news['link'],
                'content': content
            })
            time.sleep(1)
        except:
            pass

    return pd.DataFrame(news_data)

def extract_news_links(soup, num_news):
    selectors = [
        'a.card-mini__title',
        'a.card-full-news__title',
        'a[href*="/news/"]'
    ]

    links = []
    for selector in selectors:
        for link in soup.select(selector):
            href = link.get('href')
            title = link.get_text(strip=True)

            if href and title and len(title) > 10:
                if href.startswith('/'):
                    href = 'https://lenta.ru' + href

                if href not in [x['link'] for x in links]:
                    links.append({'title': title, 'link': href})

            if len(links) >= num_news * 2:
                break

    return links

def fetch_article_content(article_url, headers):
    try:
        response = requests.get(article_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        content_parts = []
        text_selectors = [
            '.topic-body__content',
            '.topic-body__title',
            'p'
        ]

        for selector in text_selectors:
            for element in soup.select(selector):
                text = element.get_text(strip=True)
                if text and len(text) > 20:
                    content_parts.append(text)
            if content_parts:
                break

        return ' '.join(content_parts) if content_parts else "Текст не найден"
    except:
        return "Ошибка загрузки"

if __name__ == "__main__":
    news_dataframe = fetch_news(3)

    if news_dataframe.empty:
        print("Не удалось получить новости")
    else:
        print(f"Получено {len(news_dataframe)} новостей")
        for i, row in news_dataframe.iterrows():
            print(f"\n{i + 1}. {row['title']}")
            print(f"Ссылка: {row['link']}")
            print(f"Текст: {row['content'][:120]}...")

        news_dataframe.to_csv('lenta_news.csv', index=False, encoding='utf-8')
        print("\nСохранено в lenta_news.csv")


 
