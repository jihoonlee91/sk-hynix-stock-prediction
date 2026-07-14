import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from forecast import forecast_prices


def _mape(actual, predicted):
    return float(np.mean(np.abs(np.array(predicted) - np.array(actual)) / np.array(actual)) * 100)


def _mae(actual, predicted):
    return float(np.mean(np.abs(np.array(predicted) - np.array(actual))))


def _directional_accuracy(actual, predicted, last_known):
    """직전 종가 대비 상승/하락 방향을 맞췄는지 비율로 측정한다.

    가격 오차(MAPE)가 작아도 방향을 계속 반대로 짚으면 실전에서 쓸모가 없으므로,
    가격 자체의 정확도와 별개로 "다음 거래일이 오를지 내릴지"를 얼마나 맞히는지
    따로 검증한다. 실제 변화가 정확히 0(보합)인 표본은 방향 판단이 무의미해 제외한다.
    """
    actual_direction = np.sign(np.array(actual) - last_known)
    predicted_direction = np.sign(np.array(predicted) - last_known)
    mask = actual_direction != 0
    if not mask.any():
        return None
    return float(np.mean(actual_direction[mask] == predicted_direction[mask]))


def backtest(close_prices, horizon=5, context_length=90):
    """가장 최근 horizon일을 정답으로 숨기고, 그 이전 데이터만으로 예측해 정확도를 측정한다.

    비교 기준(baseline)으로 "내일도 오늘과 같은 값일 것"이라는 단순 랜덤워크(persistence)
    예측을 함께 계산한다. 어떤 예측 모델이든 이 단순 기준보다 못하면 실전 가치가 없다는
    것이 시계열 예측 검증의 기본 원칙이다.
    """
    if len(close_prices) < context_length + horizon:
        raise ValueError("백테스트에 필요한 만큼 데이터가 충분하지 않습니다.")

    actual = close_prices[-horizon:]
    context = close_prices[-(context_length + horizon):-horizon]
    last_known = context[-1]

    result = forecast_prices(context, sentiment_score=0.0, horizon=horizon)
    predicted = result["median"]
    naive_predicted = [last_known] * horizon

    return {
        "actual": actual,
        "predicted": predicted,
        "mae": _mae(actual, predicted),
        "mape": _mape(actual, predicted),
        "naive_mape": _mape(actual, naive_predicted),
        "directional_accuracy": _directional_accuracy(actual, predicted, last_known),
    }


def rolling_backtest(close_prices, horizon=5, context_length=90, num_windows=12, step=5):
    """과거 여러 시점을 순회하며 반복적으로 예측/채점해 성능을 검증한다 (walk-forward).

    단일 구간 backtest()는 우연히 잘/못 맞을 수 있어 통계적으로 부족하다.
    이 함수는 최근부터 과거로 `step`일씩 이동하며 최대 `num_windows`개의 독립적인
    예측 구간에 대해 앙상블 모델·Chronos 단독·랜덤워크(naive) 세 가지를 나란히
    검증한다. 앙상블(Chronos+추세모델)이 Chronos 단독보다 실제로 더 나은지,
    그리고 둘 다 naive 기준보다 나은지를 이 비교로 판단한다.
    """
    windows = []
    for i in range(num_windows):
        end = len(close_prices) - i * step
        start = end - horizon - context_length
        if start < 0:
            break

        context = close_prices[start : end - horizon]
        actual = close_prices[end - horizon : end]
        last_known = context[-1]

        ensemble_result = forecast_prices(context, sentiment_score=0.0, horizon=horizon, use_ensemble=True)
        chronos_only_result = forecast_prices(context, sentiment_score=0.0, horizon=horizon, use_ensemble=False)
        naive_predicted = [last_known] * horizon

        # 실제/예측 종가를 직전 종가(last_known) 대비 수익률로 환산해 둔다.
        # report.py가 "실제 수익률 vs 예측 수익률" 산점도를 그릴 때 쓴다.
        actual_returns = [(a - last_known) / last_known for a in actual]
        predicted_returns = [(p - last_known) / last_known for p in ensemble_result["median"]]

        windows.append(
            {
                "mae": _mae(actual, ensemble_result["median"]),
                "mape": _mape(actual, ensemble_result["median"]),
                "chronos_only_mape": _mape(actual, chronos_only_result["median"]),
                "naive_mape": _mape(actual, naive_predicted),
                "actual_returns": actual_returns,
                "predicted_returns": predicted_returns,
                "directional_accuracy": _directional_accuracy(actual, ensemble_result["median"], last_known),
            }
        )

    mapes = [w["mape"] for w in windows]
    maes = [w["mae"] for w in windows]
    chronos_only_mapes = [w["chronos_only_mape"] for w in windows]
    naive_mapes = [w["naive_mape"] for w in windows]
    directional_accuracies = [w["directional_accuracy"] for w in windows if w["directional_accuracy"] is not None]

    return {
        "num_windows": len(windows),
        "windows": windows,
        "mape_mean": float(np.mean(mapes)),
        "mape_std": float(np.std(mapes)),
        "mae_mean": float(np.mean(maes)),
        "chronos_only_mape_mean": float(np.mean(chronos_only_mapes)),
        "naive_mape_mean": float(np.mean(naive_mapes)),
        "directional_accuracy_mean": float(np.mean(directional_accuracies)) if directional_accuracies else None,
    }


if __name__ == "__main__":
    from fetch_price import fetch_price_history

    df = fetch_price_history()
    prices = df["Close"].tolist()

    result = backtest(prices)
    print(
        f"[단일 구간] MAE: {result['mae']:,.0f}원, MAPE: {result['mape']:.2f}% "
        f"(naive 기준 {result['naive_mape']:.2f}%), 방향 적중률: {result['directional_accuracy']}"
    )
    for a, p in zip(result["actual"], result["predicted"]):
        print(f"  실제 {a:,.0f} vs 예측 {p:,.0f}")

    rolling = rolling_backtest(prices)
    print(
        f"\n[롤링 백테스트, {rolling['num_windows']}개 구간] "
        f"앙상블 평균 MAPE: {rolling['mape_mean']:.2f}% (표준편차 {rolling['mape_std']:.2f}%p) | "
        f"Chronos 단독 평균 MAPE: {rolling['chronos_only_mape_mean']:.2f}% | "
        f"naive 평균 MAPE: {rolling['naive_mape_mean']:.2f}% | "
        f"방향 적중률: {rolling['directional_accuracy_mean']:.1%}"
    )
