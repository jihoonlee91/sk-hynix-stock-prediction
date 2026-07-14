"""2단계(가격 예측·백테스트) 워커. 1단계(stage_collect.py)와 완전히 분리된 프로세스로
실행한다. 감성분류기(BERT 2종, ~900MB)를 이미 로드했던 프로세스가 여전히 살아있는
채로 같은 프로세스 안에서 Chronos-Bolt-base(~800MB)까지 로드하면 이 개발 환경에서
간헐적으로 세그폴트/페이징 파일 부족 오류가 발생했다. 1단계 프로세스가 완전히
종료된 뒤(OS가 그 메모리를 완전히 회수한 뒤) 이 프로세스가 새로 시작되므로 그 문제가
재현되지 않는다.
"""

import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(input_path, output_path):
    from backtest import backtest, rolling_backtest
    from forecast import forecast_prices

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    all_prices = [row["Close"] for row in data["price_df"]]
    avg_sentiment = data["avg_sentiment"]

    print("[1/3] Chronos-Bolt + 추세모델 앙상블 예측 중...")
    forecast = forecast_prices(all_prices[-90:], sentiment_score=avg_sentiment)
    print(f"  -> 다음 거래일 예측 중앙값: {forecast['median'][0]:,.0f}원")

    print("[2/3] 백테스트(최근 구간) 검증 중...")
    backtest_result = backtest(all_prices)
    print(
        f"  -> 최근 {len(backtest_result['actual'])}거래일 기준 MAPE {backtest_result['mape']:.2f}% "
        f"(naive 기준 {backtest_result['naive_mape']:.2f}%)"
    )

    print("[3/3] 롤링 백테스트(여러 과거 시점) 검증 중...")
    rolling_result = rolling_backtest(all_prices)
    print(
        f"  -> {rolling_result['num_windows']}개 구간 앙상블 평균 MAPE {rolling_result['mape_mean']:.2f}% "
        f"(Chronos단독 {rolling_result['chronos_only_mape_mean']:.2f}%, naive {rolling_result['naive_mape_mean']:.2f}%), "
        f"방향 적중률 {rolling_result['directional_accuracy_mean']:.1%}"
    )

    payload = {
        "forecast": forecast,
        "backtest_result": backtest_result,
        "rolling_result": rolling_result,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
