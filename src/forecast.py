import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from chronos import BaseChronosPipeline

# Chronos-Bolt: 기존 chronos-t5 계열보다 최근에 나온 버전으로, 같은 크기에서
# 더 정확하고(Amazon 공식 벤치마크 기준) 추론도 훨씬 빠르다(자기회귀 샘플링 대신
# 분위수를 직접 예측). 사전학습 모델 중 최신/최선을 쓴다는 목표에 맞춰 채택.
MODEL_NAME = "amazon/chronos-bolt-small"
HORIZON = 5  # 향후 5거래일 예측
QUANTILE_LEVELS = [0.1, 0.5, 0.9]

# 뉴스 감성 점수(-1~1)를 예측치에 반영하는 간단한 보정 계수.
# 학습된 결합 모델이 아니라, "감성이 강하게 긍/부정적이면 예측을 몇 % 정도
# 같은 방향으로 밀어준다"는 수준의 휴리스틱임을 명확히 밝혀둔다.
SENTIMENT_ADJUSTMENT = 0.01  # 감성 점수 1.0당 최대 ±1% 보정

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = BaseChronosPipeline.from_pretrained(
            MODEL_NAME, device_map="cpu", torch_dtype=torch.float32
        )
    return _pipeline


def forecast_prices(close_prices, sentiment_score=0.0, horizon=HORIZON):
    context = torch.tensor(close_prices, dtype=torch.float32)
    pipeline = _get_pipeline()
    quantiles, _mean = pipeline.predict_quantiles(
        inputs=context, prediction_length=horizon, quantile_levels=QUANTILE_LEVELS
    )
    quantiles = quantiles[0].numpy()  # (horizon, len(QUANTILE_LEVELS))

    low = quantiles[:, 0]
    median = quantiles[:, 1]
    high = quantiles[:, 2]

    adjustment = 1.0 + sentiment_score * SENTIMENT_ADJUSTMENT
    median_adjusted = median * adjustment
    low_adjusted = low * adjustment
    high_adjusted = high * adjustment

    return {
        "median": median_adjusted.tolist(),
        "low": low_adjusted.tolist(),
        "high": high_adjusted.tolist(),
        "sentiment_score": sentiment_score,
        "adjustment_factor": adjustment,
    }


if __name__ == "__main__":
    from fetch_price import fetch_price_history

    df = fetch_price_history()
    prices = df["Close"].tolist()[-60:]
    result = forecast_prices(prices, sentiment_score=0.3)
    print(result)
