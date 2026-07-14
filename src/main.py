import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from fetch_news import fetch_news
from fetch_price import fetch_price_history
from forecast import forecast_prices
from report import generate_report
from sentiment import score_texts


def main():
    print("[1/4] 주가 데이터 수집 중...")
    price_df = fetch_price_history()
    print(f"  -> {len(price_df)}일치 데이터 확보 (최근 종가: {price_df['Close'].iloc[-1]:,.0f}원)")

    print("[2/4] 뉴스 수집 및 감성분석 중...")
    news = fetch_news()
    sentiment_scores = score_texts([item["title"] for item in news])
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
    print(f"  -> 뉴스 {len(news)}건, 평균 감성 점수 {avg_sentiment:+.2f}")

    print("[3/4] Chronos로 주가 예측 중...")
    recent_prices = price_df["Close"].tolist()[-90:]
    forecast = forecast_prices(recent_prices, sentiment_score=avg_sentiment)
    print(f"  -> 다음 거래일 예측 중앙값: {forecast['median'][0]:,.0f}원")

    print("[4/4] 정적 리포트 생성 중...")
    report_path = generate_report(price_df, news, sentiment_scores, forecast)
    print(f"  -> 생성 완료: {report_path}")


if __name__ == "__main__":
    main()
