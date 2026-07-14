import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
ASSETS_DIR = os.path.join(DOCS_DIR, "assets")


def _forecast_dates(last_date_str, horizon):
    last_date = pd.Timestamp(last_date_str)
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)


def _plot_chart(price_df, forecast, forecast_dates, out_path):
    history = price_df.tail(90)

    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(history["Date"]), history["Close"], label="실제 종가", color="#1f77b4")

    plt.plot(forecast_dates, forecast["median"], label="예측 중앙값", color="#d62728", linestyle="--", marker="o")
    plt.fill_between(
        forecast_dates,
        forecast["low"],
        forecast["high"],
        color="#d62728",
        alpha=0.15,
        label="예측 구간 (10~90%)",
    )

    plt.title("SK하이닉스(000660.KS) 주가 예측")
    plt.xlabel("날짜")
    plt.ylabel("종가 (KRW)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _news_rows_html(news, sentiment_scores):
    rows = []
    for item, score in zip(news, sentiment_scores):
        mood = "긍정" if score > 0.2 else "부정" if score < -0.2 else "중립"
        rows.append(
            f"<tr><td>{item['query']}</td><td>{item['title']}</td>"
            f"<td>{mood} ({score:+.2f})</td></tr>"
        )
    return "\n".join(rows)


def _forecast_rows_html(forecast_dates, forecast):
    rows = []
    for date, low, median, high in zip(
        forecast_dates, forecast["low"], forecast["median"], forecast["high"]
    ):
        rows.append(
            f"<tr><td>{date.strftime('%Y-%m-%d')}</td>"
            f"<td>{low:,.0f}</td><td>{median:,.0f}</td><td>{high:,.0f}</td></tr>"
        )
    return "\n".join(rows)


def generate_report(price_df, news, sentiment_scores, forecast):
    os.makedirs(ASSETS_DIR, exist_ok=True)

    last_date = price_df["Date"].iloc[-1]
    last_close = price_df["Close"].iloc[-1]
    forecast_dates = _forecast_dates(last_date, len(forecast["median"]))

    chart_path = os.path.join(ASSETS_DIR, "chart.png")
    _plot_chart(price_df, forecast, forecast_dates, chart_path)

    avg_sentiment = forecast["sentiment_score"]
    mood = "긍정적" if avg_sentiment > 0.2 else "부정적" if avg_sentiment < -0.2 else "중립적"

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>SK하이닉스 주가 예측 PoC</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 16px; color: #222; }}
  h1 {{ font-size: 1.6rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.9rem; }}
  th {{ background: #f5f5f5; }}
  img {{ max-width: 100%; }}
  .disclaimer {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 12px; border-radius: 6px; font-size: 0.85rem; }}
  .meta {{ color: #666; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1>SK하이닉스(000660.KS) 주가 예측 PoC</h1>
<p class="meta">기준일: {last_date} (종가 {last_close:,.0f}원) · 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="disclaimer">
  ⚠️ 이 페이지는 사전학습 모델(Amazon Chronos, koelectra) 활용 가능성을 검증하는 <b>PoC(Proof of Concept)</b>이며, 실제 투자 판단에 사용해서는 안 됩니다.
</div>

<h2>예측 차트</h2>
<img src="assets/chart.png" alt="주가 예측 차트">

<h2>향후 {len(forecast['median'])}거래일 예측</h2>
<table>
<tr><th>날짜</th><th>하한(10%)</th><th>중앙값</th><th>상한(90%)</th></tr>
{_forecast_rows_html(forecast_dates, forecast)}
</table>

<h2>뉴스 감성 분석</h2>
<p>최근 수집 뉴스 {len(news)}건의 평균 감성 점수: <b>{avg_sentiment:+.2f}</b> ({mood}) — 예측 중앙값에 {(forecast['adjustment_factor'] - 1):+.2%} 보정 반영됨 (감성 점수 기반 단순 휴리스틱)</p>
<table>
<tr><th>검색어</th><th>제목</th><th>감성</th></tr>
{_news_rows_html(news, sentiment_scores)}
</table>

<h2>구성 요소</h2>
<ul>
  <li>주가 데이터: Yahoo Finance (yfinance)</li>
  <li>뉴스: Google News RSS ("SK하이닉스", "반도체 지정학" 검색)</li>
  <li>감성분석: <code>monologg/koelectra-small-finetuned-nsmc</code> (사전학습, 공개 모델)</li>
  <li>가격 예측: <code>amazon/chronos-t5-small</code> (Amazon Chronos, 사전학습 시계열 파운데이션 모델, zero-shot)</li>
</ul>

</body>
</html>
"""

    with open(os.path.join(DOCS_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    return os.path.join(DOCS_DIR, "index.html")
