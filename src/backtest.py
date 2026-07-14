import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from forecast import forecast_prices


def backtest(close_prices, horizon=5, context_length=90):
    """가장 최근 horizon일을 정답으로 숨기고, 그 이전 데이터만으로 예측해 정확도를 측정한다."""
    if len(close_prices) < context_length + horizon:
        raise ValueError("백테스트에 필요한 만큼 데이터가 충분하지 않습니다.")

    actual = close_prices[-horizon:]
    context = close_prices[-(context_length + horizon):-horizon]

    result = forecast_prices(context, sentiment_score=0.0, horizon=horizon)
    predicted = result["median"]

    errors = np.array(predicted) - np.array(actual)
    mae = float(np.mean(np.abs(errors)))
    mape = float(np.mean(np.abs(errors) / np.array(actual)) * 100)

    return {
        "actual": actual,
        "predicted": predicted,
        "mae": mae,
        "mape": mape,
    }


def rolling_backtest(close_prices, horizon=5, context_length=90, num_windows=10, step=5):
    """과거 여러 시점을 순회하며 반복적으로 예측/채점해 성능을 검증한다 (walk-forward).

    단일 구간 backtest()는 우연히 잘/못 맞을 수 있어 통계적으로 부족하다.
    이 함수는 최근부터 과거로 `step`일씩 이동하며 최대 `num_windows`개의 독립적인
    예측 구간에 대해 MAE/MAPE를 계산하고 평균/표준편차를 함께 낸다.
    """
    windows = []
    for i in range(num_windows):
        end = len(close_prices) - i * step
        start = end - horizon - context_length
        if start < 0:
            break

        context = close_prices[start : end - horizon]
        actual = close_prices[end - horizon : end]

        result = forecast_prices(context, sentiment_score=0.0, horizon=horizon)
        predicted = result["median"]

        errors = np.array(predicted) - np.array(actual)
        mae = float(np.mean(np.abs(errors)))
        mape = float(np.mean(np.abs(errors) / np.array(actual)) * 100)
        windows.append({"mae": mae, "mape": mape})

    mapes = [w["mape"] for w in windows]
    maes = [w["mae"] for w in windows]

    return {
        "num_windows": len(windows),
        "windows": windows,
        "mape_mean": float(np.mean(mapes)),
        "mape_std": float(np.std(mapes)),
        "mae_mean": float(np.mean(maes)),
    }


if __name__ == "__main__":
    from fetch_price import fetch_price_history

    df = fetch_price_history()
    prices = df["Close"].tolist()

    result = backtest(prices)
    print(f"[단일 구간] MAE: {result['mae']:,.0f}원, MAPE: {result['mape']:.2f}%")
    for a, p in zip(result["actual"], result["predicted"]):
        print(f"  실제 {a:,.0f} vs 예측 {p:,.0f}")

    rolling = rolling_backtest(prices)
    print(
        f"\n[롤링 백테스트, {rolling['num_windows']}개 구간] "
        f"평균 MAPE: {rolling['mape_mean']:.2f}% (표준편차 {rolling['mape_std']:.2f}%p), "
        f"평균 MAE: {rolling['mae_mean']:,.0f}원"
    )
