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


def _backtest_rows_html(backtest_result):
    rows = []
    for actual, predicted in zip(backtest_result["actual"], backtest_result["predicted"]):
        error_pct = (predicted - actual) / actual * 100
        rows.append(
            f"<tr><td>{actual:,.0f}</td><td>{predicted:,.0f}</td><td>{error_pct:+.2f}%</td></tr>"
        )
    return "\n".join(rows)


def _plot_rolling_chart(rolling_result, out_path):
    # windows[0]이 가장 최근 구간이므로, 그래프는 과거->현재 순으로 보이도록 뒤집는다.
    windows = list(reversed(rolling_result["windows"]))
    mapes = [w["mape"] for w in windows]
    labels = [f"-{len(windows) - i}" for i in range(len(windows))]

    plt.figure(figsize=(10, 3.5))
    plt.bar(labels, mapes, color="#2ca02c")
    plt.axhline(rolling_result["mape_mean"], color="#d62728", linestyle="--", label="평균 MAPE")
    plt.title("롤링 백테스트: 과거 예측 구간별 오차율(MAPE)")
    plt.xlabel("예측 구간 (과거 → 최근)")
    plt.ylabel("MAPE (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _plot_correlation_chart(price_df, related_prices, out_path):
    history = price_df.tail(180).copy()
    history["Date"] = pd.to_datetime(history["Date"])
    base = history["Close"].iloc[0]
    normalized = history["Close"] / base * 100

    plt.figure(figsize=(10, 4.5))
    plt.plot(history["Date"], normalized, label="SK하이닉스", linewidth=2, color="#1f77b4")

    colors = ["#ff7f0e", "#2ca02c", "#9467bd"]
    for (name, df), color in zip(related_prices.items(), colors):
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[df["Date"] >= history["Date"].iloc[0]]
        if df.empty:
            continue
        df_base = df["Close"].iloc[0]
        plt.plot(df["Date"], df["Close"] / df_base * 100, label=name, color=color, alpha=0.8)

    plt.title("SK하이닉스 vs 관련 종목 (최근 180일, 시작일=100 기준 정규화)")
    plt.xlabel("날짜")
    plt.ylabel("정규화 지수 (시작일=100)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _correlation_rows_html(correlations):
    rows = []
    for name, corr in correlations.items():
        strength = "강함" if abs(corr) > 0.6 else "보통" if abs(corr) > 0.3 else "약함"
        rows.append(f"<tr><td>{name}</td><td>{corr:+.2f}</td><td>{strength}</td></tr>")
    return "\n".join(rows)


def _outlook_text(forecast, last_close, avg_sentiment):
    horizon = len(forecast["median"])
    final_median = forecast["median"][-1]
    change_pct = (final_median - last_close) / last_close * 100
    direction = "상승" if change_pct > 0.5 else "하락" if change_pct < -0.5 else "보합"
    mood = "긍정적" if avg_sentiment > 0.2 else "부정적" if avg_sentiment < -0.2 else "중립적"

    band_low = forecast["low"][-1]
    band_high = forecast["high"][-1]

    return (
        f"Chronos 모델은 향후 {horizon}거래일 뒤 종가를 현재({last_close:,.0f}원) 대비 "
        f"<b>{change_pct:+.2f}% ({direction})</b>한 <b>{final_median:,.0f}원</b> 안팎으로 전망합니다 "
        f"(10~90% 구간: {band_low:,.0f}원 ~ {band_high:,.0f}원). "
        f"같은 기간 수집된 뉴스 논조는 <b>{mood}</b>({avg_sentiment:+.2f})으로, 예측치에 소폭 반영되었습니다."
    )


def generate_report(
    price_df,
    news,
    sentiment_scores,
    forecast,
    backtest_result,
    rolling_result,
    related_prices,
    correlations,
):
    os.makedirs(ASSETS_DIR, exist_ok=True)

    last_date = price_df["Date"].iloc[-1]
    last_close = price_df["Close"].iloc[-1]
    forecast_dates = _forecast_dates(last_date, len(forecast["median"]))

    chart_path = os.path.join(ASSETS_DIR, "chart.png")
    _plot_chart(price_df, forecast, forecast_dates, chart_path)

    rolling_chart_path = os.path.join(ASSETS_DIR, "rolling_backtest.png")
    _plot_rolling_chart(rolling_result, rolling_chart_path)

    correlation_chart_path = os.path.join(ASSETS_DIR, "correlation.png")
    _plot_correlation_chart(price_df, related_prices, correlation_chart_path)

    avg_sentiment = forecast["sentiment_score"]
    mood = "긍정적" if avg_sentiment > 0.2 else "부정적" if avg_sentiment < -0.2 else "중립적"
    outlook = _outlook_text(forecast, last_close, avg_sentiment)

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

<h2>향후 전망 요약</h2>
<p>{outlook}</p>

<h2>예측 기법 설명</h2>
<ol>
  <li><b>가격 데이터 수집</b> — Yahoo Finance에서 SK하이닉스(000660.KS) 최근 약 2년치 일별 종가를 가져옵니다.</li>
  <li><b>뉴스 감성분석</b> — Google News RSS로 "SK하이닉스", "SK하이닉스 목표주가"(증권가 리포트/애널리스트 의견), "반도체 지정학", "반도체 업황"(산업 전반 경제 뉴스) 4개 검색어의 최신 기사 제목을 모아, 사전학습된 한국어 감성분류 모델로 각 제목을 긍정/부정 점수(-1~1)로 변환하고 평균을 냅니다.</li>
  <li><b>가격 예측</b> — 최근 90거래일 종가만을 입력으로, 별도 학습 과정 없이 <b>Amazon Chronos-Bolt</b>(사전학습된 최신 시계열 파운데이션 모델)가 향후 {len(forecast['median'])}거래일의 가격 분포를 zero-shot으로 예측합니다.</li>
  <li><b>관련 종목 비교</b> — 메모리 반도체 업황을 함께 움직이는 삼성전자/Micron/SanDisk와의 일별 수익률 상관관계를 참고 지표로 함께 제공합니다.</li>
  <li><b>감성 보정</b> — Chronos는 가격 외 정보를 직접 입력받지 않으므로, 뉴스 평균 감성 점수를 예측 중앙값에 곱셈 형태의 단순 보정(최대 ±1%)으로만 반영합니다. 학습된 결합 모델이 아니라 휴리스틱임을 명시합니다.</li>
  <li><b>성능 검증</b> — 아래 "성능 분석"에서 설명하듯, <b>미래 시점의 실제 값은 예측 시점에 전혀 사용하지 않고</b>(look-ahead 없이) 과거 여러 시점을 기준으로 같은 방식을 반복 적용해 정확도를 측정합니다.</li>
</ol>

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

<h2>관련 종목 비교</h2>
<p>메모리 반도체 업황을 함께 좌우하는 종목들과의 최근 180일 추세 비교 및 일별 수익률 상관계수입니다.</p>
<img src="assets/correlation.png" alt="관련 종목 비교 차트">
<table>
<tr><th>종목</th><th>상관계수</th><th>강도</th></tr>
{_correlation_rows_html(correlations)}
</table>

<h2>성능 분석</h2>
<h3>최근 구간 백테스트</h3>
<p>가장 최근 {len(backtest_result['actual'])}거래일을 정답으로 숨기고, <b>그 이전 데이터까지의 정보만으로</b> 같은 방식으로 예측해본 결과입니다.</p>
<p>MAE(평균 절대 오차): <b>{backtest_result['mae']:,.0f}원</b> · MAPE(평균 절대 백분율 오차): <b>{backtest_result['mape']:.2f}%</b></p>
<table>
<tr><th>실제 종가</th><th>예측 종가</th><th>오차</th></tr>
{_backtest_rows_html(backtest_result)}
</table>

<h3>롤링 백테스트 (여러 과거 시점 반복 검증)</h3>
<p>단일 구간만으로는 우연히 잘/못 맞을 수 있어, 과거 {rolling_result['num_windows']}개 시점 각각에서 "그 시점까지의 데이터만" 사용해 반복적으로 예측/채점한 결과입니다 (walk-forward, look-ahead 없음).</p>
<p>평균 MAPE: <b>{rolling_result['mape_mean']:.2f}%</b> (표준편차 {rolling_result['mape_std']:.2f}%p) · 평균 MAE: <b>{rolling_result['mae_mean']:,.0f}원</b></p>
<img src="assets/rolling_backtest.png" alt="롤링 백테스트 오차율 차트">

<h2>구성 요소</h2>
<ul>
  <li>주가 데이터: Yahoo Finance (yfinance) — SK하이닉스, 삼성전자, Micron, SanDisk</li>
  <li>뉴스: Google News RSS ("SK하이닉스", "SK하이닉스 목표주가", "반도체 지정학", "반도체 업황" 검색)</li>
  <li>감성분석: <code>monologg/koelectra-small-finetuned-nsmc</code> (사전학습, 공개 모델)</li>
  <li>가격 예측: <code>amazon/chronos-bolt-small</code> (Amazon Chronos-Bolt, 사전학습 시계열 파운데이션 모델, zero-shot)</li>
</ul>

</body>
</html>
"""

    with open(os.path.join(DOCS_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    return os.path.join(DOCS_DIR, "index.html")
