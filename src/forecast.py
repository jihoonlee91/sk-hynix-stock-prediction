import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from chronos import BaseChronosPipeline

MODEL_NAME = "amazon/chronos-t5-small"
HORIZON = 5  # 향후 5거래일 예측

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
    samples = pipeline.predict(inputs=context, prediction_length=horizon)  # (1, num_samples, horizon)
    samples = samples[0].numpy()  # (num_samples, horizon)

    median = np.median(samples, axis=0)
    low = np.quantile(samples, 0.1, axis=0)
    high = np.quantile(samples, 0.9, axis=0)

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
