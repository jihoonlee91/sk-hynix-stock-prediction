import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from technical_indicators import FEATURE_COLUMNS, compute_indicators

# 기술적 지표를 입력으로 미래 수익률을 예측하는 gradient boosting 분위수 회귀는
# 퀀트 업계와 시계열 경진대회(M5 등)에서 가장 널리 검증된 방식 중 하나다.
# Chronos(가격 패턴 자체를 보는 파운데이션 모델), 추세모델(선형 추세)과는 완전히
# 다른 각도(모멘텀/과매수·과매도/변동성 레짐)에서 접근하므로 세 번째 비교축으로 둔다.
QUANTILE_LEVELS = (0.1, 0.5, 0.9)
MIN_TRAINING_ROWS = 60


def _training_set(indicators, close, step_ahead):
    """`step_ahead`일 뒤 로그수익률을 타깃으로 하는 (피처, 타깃) 학습셋을 만든다.

    피처는 시점 t의 지표, 타깃은 t -> t+step_ahead의 로그수익률이다. 실제 서비스라면
    미래 시점의 정보가 피처에 섞여 들어가지 않도록 항상 이 방향(과거 피처 -> 미래 타깃)만
    사용해야 하며, 아래 shift(-step_ahead)가 그 방향을 강제한다.
    """
    target = np.log(close.shift(-step_ahead) / close)
    features = indicators[FEATURE_COLUMNS]
    valid = features.notna().all(axis=1) & target.notna()
    return features[valid], target[valid]


def forecast_technical(close_prices, horizon=5, quantile_levels=QUANTILE_LEVELS):
    """기술적 지표 기반 gradient boosting 분위수 회귀로 향후 `horizon`일을 예측한다.

    학습 데이터가 부족(지표 계산에 필요한 최소 구간 미만)하면 None을 반환해,
    호출 측(forecast.py)이 이 모델 없이 진행하도록 한다.
    """
    indicators = compute_indicators(close_prices)
    close = indicators["Close"]
    last_features = indicators[FEATURE_COLUMNS].iloc[[-1]]
    last_close = close.iloc[-1]

    if last_features.isna().any(axis=1).iloc[0]:
        return None

    low_q, mid_q, high_q = quantile_levels
    quantile_paths = {low_q: [], mid_q: [], high_q: []}

    for step_ahead in range(1, horizon + 1):
        X, y = _training_set(indicators, close, step_ahead)
        if len(X) < MIN_TRAINING_ROWS:
            return None

        for q in quantile_levels:
            model = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_depth=3, max_iter=150, random_state=0
            )
            model.fit(X, y)
            log_return = model.predict(last_features)[0]
            quantile_paths[q].append(float(last_close * np.exp(log_return)))

    return {
        "low": quantile_paths[low_q],
        "median": quantile_paths[mid_q],
        "high": quantile_paths[high_q],
    }


if __name__ == "__main__":
    from fetch_price import fetch_price_history

    prices = fetch_price_history()["Close"].tolist()
    result = forecast_technical(prices)
    print(result)
