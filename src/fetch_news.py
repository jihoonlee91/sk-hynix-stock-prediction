import difflib
import re
from datetime import datetime
from urllib.parse import quote

import feedparser

DEDUP_SIMILARITY_THRESHOLD = 0.75

# 검색어별 가중치: 종목·주가 직결 정보(목표주가/리포트)는 일반 산업 뉴스보다
# 주가에 더 직접적인 신호이므로 감성 융합 시 더 크게 반영한다.
# 국내(ko) 검색어: 국내 증권가/투자자 심리를 대표.
QUERIES_KO = [
    {"text": "SK하이닉스", "weight": 1.2, "lang": "ko"},
    {"text": "SK하이닉스 목표주가", "weight": 1.5, "lang": "ko"},  # 증권가 리포트/애널리스트 의견
    {"text": "SK하이닉스 실적", "weight": 1.3, "lang": "ko"},  # 실적/가이던스
    {"text": "HBM 고대역폭메모리", "weight": 1.2, "lang": "ko"},  # HBM 수요 사이클
    {"text": "반도체 지정학", "weight": 1.0, "lang": "ko"},
    {"text": "반도체 업황", "weight": 1.0, "lang": "ko"},  # 산업/경제 전반
    {"text": "메모리 반도체 가격", "weight": 1.0, "lang": "ko"},  # D램/낸드 현물가
]

# 해외(en) 검색어: SK하이닉스는 HBM(고대역폭메모리) 수요·미중 반도체 수출규제·
# 경쟁사(삼성전자/Micron) 동향 등 해외 이슈에 민감하다. Google News 미국판(en-US)은
# Reuters/Bloomberg/CNBC뿐 아니라 Nikkei Asia(일본), SCMP(중국) 등 영문으로 발행되는
# 아시아 매체도 함께 잡히므로, 별도 언어 모델 없이 미/일/중 시각을 폭넓게 커버한다.
QUERIES_EN = [
    {"text": "SK Hynix", "weight": 1.2, "lang": "en"},
    {"text": "SK Hynix price target analyst", "weight": 1.5, "lang": "en"},  # 애널리스트/목표주가
    {"text": "SK Hynix earnings", "weight": 1.3, "lang": "en"},  # 실적/가이던스
    {"text": "HBM AI chip demand", "weight": 1.2, "lang": "en"},  # HBM 수요 사이클
    {"text": "memory chip export controls China", "weight": 1.0, "lang": "en"},  # 지정학
    {"text": "semiconductor memory chip industry outlook", "weight": 1.0, "lang": "en"},  # 업황
    {"text": "DRAM NAND price", "weight": 1.0, "lang": "en"},  # D램/낸드 현물가
]

QUERIES = QUERIES_KO + QUERIES_EN

_LOCALE_PARAMS = {
    "ko": {"hl": "ko", "gl": "KR", "ceid": "KR:ko"},
    "en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
}


def _rss_url(query_text, lang="ko", after=None, before=None):
    q = query_text
    if after:
        q += f" after:{after}"
    if before:
        q += f" before:{before}"
    locale = _LOCALE_PARAMS[lang]
    return (
        f"https://news.google.com/rss/search?q={quote(q)}"
        f"&hl={locale['hl']}&gl={locale['gl']}&ceid={locale['ceid']}"
    )


def _parse_published(entry):
    """RSS의 published_parsed(struct_time)를 YYYY-MM-DD로 정규화한다.

    파싱 가능한 날짜가 있어야 감성 점수의 최신성 가중치(sentiment.py)를
    계산할 수 있다. 실패 시 빈 문자열을 반환해 최신성 가중치를 1.0으로 폴백.
    """
    parsed = getattr(entry, "published_parsed", None)
    if parsed:
        return datetime(*parsed[:6]).strftime("%Y-%m-%d")
    return ""


def _normalize_title(title):
    # Google News 제목은 보통 " - 매체명"으로 끝난다. 같은 기사를 다른 매체가
    # 받아쓴 경우 매체명만 다르고 본문 제목은 동일/유사하므로, 비교 전에 제거한다.
    title = re.split(r" - [^-]+$", title, maxsplit=1)[0]
    return re.sub(r"[^\w\s]", "", title).strip().lower()


def _deduplicate(articles, threshold=DEDUP_SIMILARITY_THRESHOLD):
    """검색어 4개(국문)+4개(영문)를 각각 돌리다 보면 같은 사건을 다룬 기사가
    여러 검색어·매체에서 중복으로 잡힌다. 이를 그대로 감성 평균에 넣으면 화제성이
    큰 사건 하나가 표본을 지배해 버리므로(예: 같은 속보를 10개 매체가 받아쓴 경우),
    같은 언어 내에서 정규화한 제목의 유사도가 threshold 이상이면 하나로 합친다.
    합칠 때는 여러 검색어에 걸쳐 잡힌 만큼 더 중요한 사건일 수 있다고 보고
    query_weight는 max로 취하고, 몇 건이 뭉쳐졌는지 duplicate_count로 남긴다.
    """
    deduped = []
    seen_links = set()
    for article in articles:
        if article["link"] in seen_links:
            continue
        seen_links.add(article["link"])

        normalized = _normalize_title(article["title"])
        match = next(
            (
                kept
                for kept in deduped
                if kept["lang"] == article["lang"]
                and difflib.SequenceMatcher(
                    None, normalized, _normalize_title(kept["title"])
                ).ratio()
                >= threshold
            ),
            None,
        )

        if match is not None:
            match["query_weight"] = max(match["query_weight"], article["query_weight"])
            match["duplicate_count"] += 1
            continue

        article = dict(article, duplicate_count=1)
        deduped.append(article)

    return deduped


def fetch_news(max_per_query=30, after=None, before=None):
    """뉴스 헤드라인을 수집하고, 같은 사건을 다룬 중복 기사를 합쳐서 반환한다.

    `after`/`before`(YYYY-MM-DD)를 지정하면 Google News의 날짜 연산자를 이용해
    해당 기간에 게재된 기사만 가져온다. 백테스트에서 특정 과거 시점 기준으로
    "그 시점까지 실제로 존재했던 뉴스"만 사용하려 할 때(look-ahead 방지) 필요하다.
    """
    articles = []
    for query in QUERIES:
        feed = feedparser.parse(
            _rss_url(query["text"], lang=query["lang"], after=after, before=before)
        )
        for entry in feed.entries[:max_per_query]:
            articles.append(
                {
                    "query": query["text"],
                    "query_weight": query["weight"],
                    "lang": query["lang"],
                    "title": entry.title,
                    "published": _parse_published(entry) or entry.get("published", ""),
                    "link": entry.link,
                }
            )
    return _deduplicate(articles)


if __name__ == "__main__":
    news = fetch_news()
    by_lang = {}
    for item in news:
        by_lang.setdefault(item["lang"], []).append(item)
    for lang, items in by_lang.items():
        print(f"--- lang={lang}, {len(items)}건 ---")
        for item in items[:3]:
            print(" ", item["query"], "-", item["title"], f"({item['published']})")
    print(f"total: {len(news)}")

    print("\n[과거 시점 검색 예시] 2026-05-01 ~ 2026-05-10")
    historical = fetch_news(max_per_query=5, after="2026-05-01", before="2026-05-10")
    for item in historical[:5]:
        print(item["lang"], item["query"], "-", item["title"], f"({item['published']})")
    print(f"total: {len(historical)}")
