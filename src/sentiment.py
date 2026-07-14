import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from transformers import pipeline

# NOTE: 원래 금융 특화 모델(snunlp/KR-FinBert-SC, ~110M params)을 쓰려 했으나,
# 이 실행 환경의 프로세스당 메모리 한도(~500-700MB) 근처에서 로딩이 간헐적으로
# 실패했다. koelectra-small(~14M params)은 안정적으로 동작하는 대신 영화 리뷰
# 데이터로 학습된 범용 감성분류기라 금융 도메인 정확도는 떨어진다.
# 신뢰성 있게 동작하는 것을 우선해 이 모델로 대체했다 (자세한 내용은
# ARCHITECTURE.md 참고).
MODEL_NAME = "monologg/koelectra-small-finetuned-nsmc"

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None)
    return _classifier


LABEL_SCORE = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}


def score_texts(texts):
    if not texts:
        return []

    classifier = _get_classifier()
    results = classifier(texts, truncation=True)

    scores = []
    for result in results:
        best = max(result, key=lambda r: r["score"])
        label = best["label"].lower()
        scores.append(LABEL_SCORE.get(label, 0.0) * best["score"])
    return scores


def average_sentiment(texts):
    scores = score_texts(texts)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


if __name__ == "__main__":
    sample = ["SK하이닉스 실적 호조로 주가 급등", "SK하이닉스 반도체 수출 규제로 우려 확대"]
    print(list(zip(sample, score_texts(sample))))
