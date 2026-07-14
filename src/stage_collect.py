"""1단계(데이터 수집·감성분석) 워커. 완전히 독립된 프로세스로 실행되어 끝나면
바로 종료된다 (main.py 참고: Chronos-Bolt-base를 로드하는 2단계 이전에 이 프로세스가
쓴 메모리가 OS에 확실히 반환되도록, 같은 프로세스 안에서 이어서 실행하지 않는다).
"""

import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _df_to_records(df):
    return df.to_dict(orient="records")


def main(output_path):
    from fetch_news import fetch_news
    from fetch_price import fetch_price_history
    from index_scan import scan_index_correlations
    from related_stocks import compute_correlations, diagnose_move, fetch_related_prices
    from sentiment import score_articles, weighted_average_sentiment

    print("[1/4] 주가 데이터 수집 중...")
    price_df = fetch_price_history()
    print(f"  -> {len(price_df)}일치 데이터 확보 (최근 종가: {price_df['Close'].iloc[-1]:,.0f}원)")

    print("[2/4] 코스피/S&P500/나스닥/반도체지수/동종업계 비교 분석 중...")
    related_prices = fetch_related_prices()
    correlations = compute_correlations(price_df, related_prices)
    move_diagnosis = diagnose_move(price_df, related_prices)
    print(
        f"  -> 최근 5거래일 하이닉스 {move_diagnosis['hynix_cumulative_return']:+.1%} vs "
        + ", ".join(f"{k} {v:+.1%}" for k, v in move_diagnosis["category_cumulative_return"].items())
    )

    print("[3/4] 국내외 뉴스 수집 및 감성분석 중 (한국어+영어, 중복 제거)...")
    news = fetch_news()
    sentiment_scores = score_articles(news)
    avg_sentiment = weighted_average_sentiment(news, reference_date=price_df["Date"].iloc[-1])
    lang_counts = {}
    for item in news:
        lang_counts[item["lang"]] = lang_counts.get(item["lang"], 0) + 1
    print(
        f"  -> 뉴스 {len(news)}건({', '.join(f'{k}:{v}' for k, v in lang_counts.items())}, 중복 제거 후), "
        f"가중평균 감성 점수 {avg_sentiment:+.2f}"
    )

    print("[4/4] 코스피200/나스닥100/S&P500 대표종목 상관관계 스캔 중...")
    index_scan_result = scan_index_correlations(price_df)
    for index_name, top in index_scan_result.items():
        print(f"  -> {index_name} 상위 상관: " + ", ".join(f"{t['name']}({t['correlation']:+.2f})" for t in top[:3]))

    payload = {
        "price_df": _df_to_records(price_df),
        "related_prices": {name: _df_to_records(df) for name, df in related_prices.items()},
        "correlations": correlations,
        "move_diagnosis": move_diagnosis,
        "news": news,
        "sentiment_scores": sentiment_scores,
        "avg_sentiment": avg_sentiment,
        "index_scan_result": index_scan_result,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1])
