from urllib.parse import quote

import feedparser

QUERIES = [
    "SK하이닉스",
    "SK하이닉스 목표주가",  # 증권가 리포트/애널리스트 의견
    "반도체 지정학",
    "반도체 업황",  # 산업/경제 전반 뉴스
]


def _rss_url(query):
    return f"https://news.google.com/rss/search?q={quote(query)}&hl=ko&gl=KR&ceid=KR:ko"


def fetch_news(max_per_query=15):
    articles = []
    for query in QUERIES:
        feed = feedparser.parse(_rss_url(query))
        for entry in feed.entries[:max_per_query]:
            articles.append(
                {
                    "query": query,
                    "title": entry.title,
                    "published": entry.get("published", ""),
                    "link": entry.link,
                }
            )
    return articles


if __name__ == "__main__":
    news = fetch_news()
    for item in news[:5]:
        print(item["query"], "-", item["title"])
    print(f"total: {len(news)}")
