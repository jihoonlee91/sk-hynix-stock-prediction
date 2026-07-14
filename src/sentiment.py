import os
from datetime import datetime

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from transformers import pipeline

# 뉴스는 국내(한국어)뿐 아니라 해외(영어: 미국/일본/중국계 영문매체 포함, fetch_news.py
# 참고) 소스도 함께 수집하므로, 언어별로 도메인 특화 분류기를 따로 둔다.
# - 한국어: 금융 도메인 특화 BERT(~110M). 과거엔 개발 샌드박스의 프로세스당 메모리
#   한도(~500-700MB) 때문에 로딩이 실패해 범용 모델(koelectra-small)로 대체했었으나,
#   이 환경은 그 제약이 없음을 확인(로딩 테스트 통과)하여 다시 금융 특화 모델로 교체.
# - 영어: FinBERT(ProsusAI). 금융 뉴스/문장 감성 분류로 가장 널리 쓰이는 사전학습 모델.
# 자세한 배경은 ARCHITECTURE.md 참고.
MODEL_NAMES = {
    "ko": "snunlp/KR-FinBert-SC",
    "en": "ProsusAI/finbert",
}

_classifiers = {}


def _get_classifier(lang="ko"):
    if lang not in _classifiers:
        _classifiers[lang] = pipeline(
            "text-classification", model=MODEL_NAMES[lang], top_k=None
        )
    return _classifiers[lang]


LABEL_SCORE = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}


def score_texts(texts, lang="ko"):
    """각 텍스트를 분류기 확신도로 가중된 감성 점수([-1, 1])로 변환한다."""
    if not texts:
        return []

    classifier = _get_classifier(lang)
    results = classifier(texts, truncation=True)

    scores = []
    for result in results:
        best = max(result, key=lambda r: r["score"])
        label = best["label"].lower()
        scores.append(LABEL_SCORE.get(label, 0.0) * best["score"])
    return scores


def score_articles(articles):
    """기사 목록을 각자의 `lang` 필드에 맞는 분류기로 채점하고, 원 순서를 보존해 반환한다."""
    if not articles:
        return []

    scores = [None] * len(articles)
    indices_by_lang = {}
    for i, article in enumerate(articles):
        indices_by_lang.setdefault(article.get("lang", "ko"), []).append(i)

    for lang, indices in indices_by_lang.items():
        texts = [articles[i]["title"] for i in indices]
        for i, score in zip(indices, score_texts(texts, lang=lang)):
            scores[i] = score
    return scores


def average_sentiment(texts, lang="ko"):
    scores = score_texts(texts, lang=lang)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def weighted_average_sentiment(articles, reference_date=None, half_life_days=3.0):
    """뉴스 기사 목록을 3가지 가중치를 곱한 가중평균 감성 점수로 융합한다.

    단순 평균은 "5일 전 낡은 기사"와 "오늘 나온 목표주가 상향 리포트"를 동등하게
    취급해 버린다. 이를 보완하기 위해 기사별로 다음을 곱해 가중치를 매긴다:
    - 분류기 확신도: score_texts()가 이미 반영 (예: 0.98 확신 긍정 vs 0.51 확신 긍정).
    - 검색어 중요도(query_weight): 목표주가/애널리스트 리포트 뉴스는 주가에 더
      직접적인 신호이므로 일반 산업 뉴스보다 가중치를 높게 둔다 (fetch_news.py 참고).
    - 최신성(recency): 게재일 기준 지수감쇠(half-life=half_life_days일)로,
      오래된 기사일수록 현재 예측에 대한 영향력을 줄인다.
    """
    if not articles:
        return 0.0

    raw_scores = score_articles(articles)

    if reference_date is None:
        reference_date = datetime.now()
    elif isinstance(reference_date, str):
        reference_date = datetime.strptime(reference_date, "%Y-%m-%d")

    weighted_sum = 0.0
    weight_sum = 0.0
    for article, score in zip(articles, raw_scores):
        query_weight = article.get("query_weight", 1.0)

        recency_weight = 1.0
        published = article.get("published")
        if published:
            try:
                pub_date = datetime.strptime(published, "%Y-%m-%d")
                age_days = max((reference_date - pub_date).days, 0)
                recency_weight = 0.5 ** (age_days / half_life_days)
            except ValueError:
                pass

        weight = query_weight * recency_weight
        weighted_sum += score * weight
        weight_sum += weight

    return weighted_sum / weight_sum if weight_sum else 0.0


if __name__ == "__main__":
    sample = ["SK하이닉스 실적 호조로 주가 급등", "SK하이닉스 반도체 수출 규제로 우려 확대"]
    print(list(zip(sample, score_texts(sample))))

    articles = [
        {"title": sample[0], "query_weight": 1.5, "published": datetime.now().strftime("%Y-%m-%d")},
        {"title": sample[1], "query_weight": 1.0, "published": "2020-01-01"},
    ]
    print("weighted:", weighted_average_sentiment(articles))
