import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from chronos import BaseChronosPipeline
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from technical_model import forecast_technical

# Chronos-Bolt-**base**: 이 저장소를 처음 만든 샌드박스는 프로세스당 메모리 한도가
# 약 500-700MB였는데, 이 환경은 그런 제약이 없음(로딩 테스트 통과, ~19GB 여유)을
# 확인했으므로 같은 계열에서 더 크고 정확한 base 크기로 올린다. 여전히 사전학습
# 파운데이션 모델을 zero-shot으로만 쓴다(별도 학습 없음).
MODEL_NAME = "amazon/chronos-bolt-base"
HORIZON = 5  # 향후 5거래일 예측
QUANTILE_LEVELS = [0.1, 0.5, 0.9]

# Chronos(딥러닝 파운데이션 모델)와 고전적 통계 모델(감쇠추세 지수평활)을 앙상블한다.
# 서로 다른 가정을 가진 모델을 블렌딩하면 한쪽이 특정 구간에서 잘못 짚어도
# 전체 오차의 분산이 줄어든다는 것이 시계열 예측 대회(M4/M5 등)에서 반복 확인된
# 경험칙이다. Chronos 비중을 높게 둔 것은 백테스트상 Chronos 단독이 더 낮은 MAPE를
# 보였기 때문(backtest.py의 비교 결과 참고)이고, 추세모델은 이를 보정하는 역할.
CHRONOS_WEIGHT = 0.7
TREND_WEIGHT = 0.3

# 뉴스 감성 점수(-1~1)를 예측치에 반영하는 보정 계수. 학습된 결합 모델이 아니라
# "감성이 강하게 긍/부정적이면 예측을 몇 % 정도 같은 방향으로 밀어준다"는 휴리스틱임을
# 명확히 밝혀둔다 (진짜 학습 기반 융합이 왜 어려운지는 ARCHITECTURE.md 참고).
SENTIMENT_ADJUSTMENT = 0.015  # 감성 점수 1.0당 최근접 거래일 기준 최대 ±1.5% 보정
# 뉴스의 가격 영향력은 며칠 안에 빠르게 희석된다고 보는 것이 합리적이므로,
# 예측 horizon이 멀어질수록(스텝마다) 보정 폭을 지수적으로 줄인다.
SENTIMENT_DECAY = 0.6

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = BaseChronosPipeline.from_pretrained(
            MODEL_NAME, device_map="cpu", torch_dtype=torch.float32
        )
    return _pipeline


def _chronos_quantiles(close_prices, horizon):
    context = torch.tensor(close_prices, dtype=torch.float32)
    pipeline = _get_pipeline()
    quantiles, _mean = pipeline.predict_quantiles(
        inputs=context, prediction_length=horizon, quantile_levels=QUANTILE_LEVELS
    )
    quantiles = quantiles[0].numpy()  # (horizon, len(QUANTILE_LEVELS))
    return quantiles[:, 0], quantiles[:, 1], quantiles[:, 2]  # low, median, high


def _trend_forecast(close_prices, horizon):
    """감쇠추세(damped trend) 지수평활 예측. Chronos와 독립적인 가정(선형 추세가
    시간이 갈수록 완만해진다)에 기반한 고전 통계 모델로, 앙상블의 두 번째 축이다.
    데이터가 짧거나 모델 적합이 실패하면 마지막 값을 유지하는 보수적 폴백을 쓴다.
    """
    series = np.asarray(close_prices, dtype=float)
    try:
        model = ExponentialSmoothing(series, trend="add", damped_trend=True).fit()
        return np.asarray(model.forecast(horizon))
    except Exception:
        return np.full(horizon, series[-1])


def forecast_prices(close_prices, sentiment_score=0.0, horizon=HORIZON, use_ensemble=True):
    low, median, high = _chronos_quantiles(close_prices, horizon)
    trend = _trend_forecast(close_prices, horizon)
    # 기술적 지표 기반 GBM 모델은 세 번째 관점으로 함께 노출만 하고(리포트의 "모델별
    # 예측 비교" 표), 앙상블 가중 블렌딩에는 넣지 않는다 — 블렌딩 비중은 backtest.py의
    # 실측 비교로 보정된 값인데, 이 모델을 추가하면 그 비교를 다시 다 돌려야 하기 때문.
    technical = forecast_technical(close_prices, horizon=horizon)

    if use_ensemble:
        # 밴드 폭(불확실성)은 Chronos의 분위수 추정을 그대로 신뢰하고, 중앙값만
        # 앙상블로 재중심화한다. low/high를 각각 독립적으로 블렌딩하면 두 모델의
        # 밴드 폭 정의가 달라 불확실성 구간이 부정확해지기 때문이다.
        offset_low = low - median
        offset_high = high - median
        median_blend = CHRONOS_WEIGHT * median + TREND_WEIGHT * trend
        low_blend = median_blend + offset_low
        high_blend = median_blend + offset_high
    else:
        median_blend, low_blend, high_blend = median, low, high

    steps = np.arange(len(median_blend))
    decay = SENTIMENT_DECAY ** steps
    adjustment = 1.0 + sentiment_score * SENTIMENT_ADJUSTMENT * decay

    return {
        "median": (median_blend * adjustment).tolist(),
        "low": (low_blend * adjustment).tolist(),
        "high": (high_blend * adjustment).tolist(),
        "sentiment_score": sentiment_score,
        "adjustment_factor": adjustment.tolist(),
        # 리포트에서 "모델별 예측치를 함께 보여준다" — 블렌딩 전 개별 모델 산출물을
        # 그대로 남겨 어떤 모델이 어떤 값을 냈는지 투명하게 노출한다.
        "model_breakdown": {
            "chronos_median": median.tolist(),
            "trend_median": trend.tolist(),
            "technical_median": technical["median"] if technical else None,
            "ensemble_median_before_sentiment": median_blend.tolist(),
        },
    }


if __name__ == "__main__":
    from fetch_price import fetch_price_history

    df = fetch_price_history()
    prices = df["Close"].tolist()[-90:]
    result = forecast_prices(prices, sentiment_score=0.3)
    for k, v in result.items():
        print(k, v)
