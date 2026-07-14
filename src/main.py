import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from backtest import backtest, rolling_backtest
from fetch_news import fetch_news
from fetch_price import fetch_price_history
from forecast import forecast_prices
from report import generate_report
from sentiment import score_texts


def main():
    print("[1/6] 주가 데이터 수집 중...")
    price_df = fetch_price_history()
    print(f"  -> {len(price_df)}일치 데이터 확보 (최근 종가: {price_df['Close'].iloc[-1]:,.0f}원)")

    print("[2/6] 뉴스 수집 및 감성분석 중...")
    news = fetch_news()
    sentiment_scores = score_texts([item["title"] for item in news])
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
    print(f"  -> 뉴스 {len(news)}건, 평균 감성 점수 {avg_sentiment:+.2f}")

    print("[3/6] Chronos로 주가 예측 중...")
    all_prices = price_df["Close"].tolist()
    recent_prices = all_prices[-90:]
    forecast = forecast_prices(recent_prices, sentiment_score=avg_sentiment)
    print(f"  -> 다음 거래일 예측 중앙값: {forecast['median'][0]:,.0f}원")

    print("[4/6] 백테스트(최근 구간)로 예측 정확도 검증 중...")
    backtest_result = backtest(all_prices)
    print(f"  -> 최근 {len(backtest_result['actual'])}거래일 기준 MAPE {backtest_result['mape']:.2f}%")

    print("[5/6] 롤링 백테스트(여러 과거 시점)로 성능 검증 중...")
    rolling_result = rolling_backtest(all_prices)
    print(
        f"  -> {rolling_result['num_windows']}개 구간 평균 MAPE "
        f"{rolling_result['mape_mean']:.2f}% (표준편차 {rolling_result['mape_std']:.2f}%p)"
    )

    print("[6/6] 정적 리포트 생성 중...")
    report_path = generate_report(
        price_df, news, sentiment_scores, forecast, backtest_result, rolling_result
    )
    print(f"  -> 생성 완료: {report_path}")


if __name__ == "__main__":
    main()
