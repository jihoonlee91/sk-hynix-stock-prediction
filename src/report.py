import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
ASSETS_DIR = os.path.join(DOCS_DIR, "assets")

_PLAIN_NUMBER_FORMATTER = FuncFormatter(lambda x, _: f"{x:,.0f}")
_PERCENT_FORMATTER = FuncFormatter(lambda x, _: f"{x:+.0%}")


def _style_axes(ax, y_formatter=None):
    """모든 차트에 공통으로 grid를 켜고, y축을 지수표기(3e+06) 대신 천단위 콤마로 표기한다."""
    ax.grid(True, alpha=0.3, linestyle="--")
    if y_formatter is not None:
        ax.yaxis.set_major_formatter(y_formatter)
    else:
        ax.yaxis.set_major_formatter(_PLAIN_NUMBER_FORMATTER)


def _forecast_dates(last_date_str, horizon):
    last_date = pd.Timestamp(last_date_str)
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)


def _plot_chart(price_df, forecast, forecast_dates, out_path):
    history = price_df.tail(90)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pd.to_datetime(history["Date"]), history["Close"], label="실제 종가", color="#1f77b4")

    ax.plot(forecast_dates, forecast["median"], label="예측 중앙값 (앙상블)", color="#d62728", linestyle="--", marker="o")
    ax.fill_between(
        forecast_dates,
        forecast["low"],
        forecast["high"],
        color="#d62728",
        alpha=0.15,
        label="예측 구간 (10~90%)",
    )

    breakdown = forecast.get("model_breakdown", {})
    if breakdown.get("technical_median"):
        ax.plot(
            forecast_dates, breakdown["technical_median"], label="기술적 지표 모델(GBM)",
            color="#9467bd", linestyle=":", marker="^", alpha=0.8,
        )

    ax.set_title("SK하이닉스(000660.KS) 주가 예측")
    ax.set_xlabel("날짜")
    ax.set_ylabel("종가 (KRW)")
    ax.legend()
    _style_axes(ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def _news_rows_html(news, sentiment_scores, top_n=25):
    """뉴스가 많아지면(다국어+검색어 확장으로 수백 건) 표를 다 채우기보다,
    감성 융합에 실제로 영향을 크게 준 상위 기사만 추려서 보여준다.

    언어별로 따로 상위 기사를 뽑은 뒤 합친다 — 한국어(KR-FinBert-SC)와 영어(FinBERT)
    분류기의 확신도 캘리브레이션이 서로 달라(한국어 쪽이 점수가 더 극단적으로 나오는
    경향), 언어 구분 없이 |점수|로만 정렬하면 실제 수집량과 무관하게 한쪽 언어가
    표를 지배해 버린다(예: 수집은 영어가 더 많은데 표시는 국내 기사가 대부분).
    """
    scored = list(zip(news, sentiment_scores))

    langs = sorted({item.get("lang", "ko") for item, _ in scored})
    per_lang_quota = max(1, top_n // len(langs)) if langs else top_n

    selected = []
    for lang in langs:
        lang_scored = [pair for pair in scored if pair[0].get("lang", "ko") == lang]
        lang_scored.sort(key=lambda pair: abs(pair[1]) * pair[0].get("query_weight", 1.0), reverse=True)
        selected.extend(lang_scored[:per_lang_quota])

    selected.sort(key=lambda pair: abs(pair[1]) * pair[0].get("query_weight", 1.0), reverse=True)

    rows = []
    for item, score in selected[:top_n]:
        mood = "긍정" if score > 0.2 else "부정" if score < -0.2 else "중립"
        lang_label = "국내" if item.get("lang") == "ko" else "해외"
        dup = item.get("duplicate_count", 1)
        dup_label = f" (유사기사 {dup}건 통합)" if dup > 1 else ""
        rows.append(
            f"<tr><td>{lang_label}</td><td>{item['query']}</td><td>{item['title']}{dup_label}</td>"
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


def _model_comparison_rows_html(forecast_dates, forecast):
    """Chronos/추세모델/기술적 지표 모델/최종 앙상블 예측치를 나란히 비교한다."""
    breakdown = forecast["model_breakdown"]
    technical = breakdown.get("technical_median")
    rows = []
    for i, date in enumerate(forecast_dates):
        technical_cell = f"{technical[i]:,.0f}" if technical else "N/A"
        rows.append(
            f"<tr><td>{date.strftime('%Y-%m-%d')}</td>"
            f"<td>{breakdown['chronos_median'][i]:,.0f}</td>"
            f"<td>{breakdown['trend_median'][i]:,.0f}</td>"
            f"<td>{technical_cell}</td>"
            f"<td>{breakdown['ensemble_median_before_sentiment'][i]:,.0f}</td>"
            f"<td><b>{forecast['median'][i]:,.0f}</b></td></tr>"
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

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(labels, mapes, color="#2ca02c", label="앙상블 MAPE (구간별)")
    ax.axhline(rolling_result["mape_mean"], color="#d62728", linestyle="--", label=f"앙상블 평균 {rolling_result['mape_mean']:.1f}%")
    ax.axhline(rolling_result["chronos_only_mape_mean"], color="#ff7f0e", linestyle=":", label=f"Chronos 단독 평균 {rolling_result['chronos_only_mape_mean']:.1f}%")
    ax.axhline(rolling_result["naive_mape_mean"], color="#7f7f7f", linestyle=":", label=f"naive(직전값 유지) 평균 {rolling_result['naive_mape_mean']:.1f}%")
    ax.set_title("롤링 백테스트: 과거 예측 구간별 오차율(MAPE) 비교")
    ax.set_xlabel("예측 구간 (과거 → 최근)")
    ax.set_ylabel("MAPE (%)")
    ax.legend(fontsize=8)
    _style_axes(ax, y_formatter=FuncFormatter(lambda x, _: f"{x:.0f}%"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_return_scatter(rolling_result, out_path):
    """롤링 백테스트의 각 예측 구간에서 '실제 수익률 vs 예측 수익률'을 점으로 찍는다.
    대각선(y=x)에 가까울수록 정확한 예측이고, 대각선을 기준으로 몇 사분면에
    찍히는지를 보면 방향(상승/하락)을 얼마나 잘 맞추는지 한눈에 보인다.
    """
    actual_returns = []
    predicted_returns = []
    for w in rolling_result["windows"]:
        actual_returns.extend(w.get("actual_returns", []))
        predicted_returns.extend(w.get("predicted_returns", []))

    if not actual_returns:
        return False

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual_returns, predicted_returns, alpha=0.6, color="#1f77b4", edgecolors="white", linewidths=0.5)

    lim = max(abs(v) for v in actual_returns + predicted_returns) * 1.15 if actual_returns else 0.1
    ax.plot([-lim, lim], [-lim, lim], color="#d62728", linestyle="--", label="완벽한 예측 (y = x)")
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.set_title("백테스트: 실제 수익률 vs 예측 수익률")
    ax.set_xlabel("실제 수익률 (직전 종가 대비)")
    ax.set_ylabel("예측 수익률 (직전 종가 대비)")
    ax.legend()
    _style_axes(ax, y_formatter=_PERCENT_FORMATTER)
    ax.xaxis.set_major_formatter(_PERCENT_FORMATTER)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def _plot_correlation_chart(price_df, related_prices, out_path):
    from related_stocks import CATEGORY_OF

    history = price_df.tail(180).copy()
    history["Date"] = pd.to_datetime(history["Date"])
    base = history["Close"].iloc[0]
    normalized = history["Close"] / base * 100

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(history["Date"], normalized, label="SK하이닉스", linewidth=2.4, color="#1f77b4")

    # 같은 계층(예: 시장 전체 3개 지수)을 같은 linestyle로 묶어 계층을 구분하고,
    # 계층 안에서는 서로 다른 색을 배정해 개별 시리즈를 구분한다. 이전에는 계층별로
    # 색+선스타일을 통째로 고정해서 같은 계층 안의 여러 지수가 겹쳐 안 보였다.
    category_linestyles = {
        "시장 전체": "-.",
        "반도체 섹터": "--",
        "메모리 동종업계": ":",
    }
    category_colors = {
        "시장 전체": ["#7f7f7f", "#8c564b", "#17becf"],
        "반도체 섹터": ["#ff7f0e"],
        "메모리 동종업계": ["#2ca02c", "#9467bd", "#e377c2"],
    }
    color_cursor = {cat: 0 for cat in category_colors}

    for name, df in related_prices.items():
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[df["Date"] >= history["Date"].iloc[0]]
        if df.empty:
            continue
        df_base = df["Close"].iloc[0]
        category = CATEGORY_OF.get(name, "")
        palette = category_colors.get(category, ["#333333"])
        color = palette[color_cursor.get(category, 0) % len(palette)]
        color_cursor[category] = color_cursor.get(category, 0) + 1
        linestyle = category_linestyles.get(category, "-")
        ax.plot(df["Date"], df["Close"] / df_base * 100, label=name, alpha=0.9, color=color, linestyle=linestyle)

    ax.set_title("SK하이닉스 vs 시장지수·반도체섹터·동종업계 (최근 180일, 시작일=100 기준)")
    ax.set_xlabel("날짜")
    ax.set_ylabel("정규화 지수 (시작일=100)")
    ax.legend(fontsize=8, ncol=2)
    _style_axes(ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_correlation_bars(items, out_path, title):
    """상관계수를 막대 길이+색으로 한눈에 보여준다 (양의 상관=빨강 계열, 음의 상관=파랑 계열).
    표의 숫자는 정확하지만 하나하나 읽어야 하는 반면, 색과 길이는 바로 눈에 들어온다."""
    items = sorted(items, key=lambda x: x[1])
    labels = [x[0] for x in items]
    values = [x[1] for x in items]
    colors = plt.cm.RdBu_r([(v + 1) / 2 for v in values])

    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.32 * len(items))))
    ax.barh(labels, values, color=colors, edgecolor="#333333", linewidth=0.5)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("상관계수 (일별 수익률 기준, 음수=반대로 움직임 · 양수=같이 움직임)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.1f}"))
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def _correlation_rows_html(correlations):
    from related_stocks import CATEGORY_OF

    rows = []
    for name, corr in correlations.items():
        strength = "강함" if abs(corr) > 0.6 else "보통" if abs(corr) > 0.3 else "약함"
        rows.append(
            f"<tr><td>{CATEGORY_OF.get(name, '기타')}</td><td>{name}</td>"
            f"<td>{corr:+.2f}</td><td>{strength}</td></tr>"
        )
    return "\n".join(rows)


def _index_scan_rows_html(index_scan_result):
    rows = []
    for index_name, items in index_scan_result.items():
        for item in items:
            strength = "강함" if abs(item["correlation"]) > 0.6 else "보통" if abs(item["correlation"]) > 0.3 else "약함"
            rows.append(
                f"<tr><td>{index_name}</td><td>{item['name']} ({item['ticker']})</td>"
                f"<td>{item['correlation']:+.2f}</td><td>{strength}</td></tr>"
            )
    return "\n".join(rows)


def _move_diagnosis_text(move_diagnosis):
    hynix = move_diagnosis["hynix_cumulative_return"]
    category_returns = move_diagnosis["category_cumulative_return"]
    rel_market = move_diagnosis["relative_to_market"]
    rel_sector = move_diagnosis["relative_to_sector"]
    rel_peers = move_diagnosis["relative_to_peers"]

    if abs(rel_market) < 0.02 and abs(rel_sector) < 0.02:
        judgment = "시장·반도체 섹터와 비슷한 폭으로 움직여 <b>시장 전체 흐름에 동조</b>한 구간으로 보입니다."
    elif abs(rel_sector) < 0.03:
        judgment = "코스피/S&P500 등 시장 전체보다는 <b>반도체 섹터 전반</b>의 움직임에 더 가깝습니다."
    elif abs(rel_peers) < 0.04:
        judgment = "반도체 섹터보다도 <b>메모리 동종업계(삼성전자/Micron/SanDisk)와 유사한 흐름</b>입니다."
    else:
        judgment = "시장·섹터·동종업계 평균 대비 괴리가 커, <b>SK하이닉스 고유 이슈(실적/수급/개별 뉴스)</b>가 더 크게 작용한 것으로 보입니다."

    lines = [
        f"최근 5거래일 SK하이닉스 누적 수익률은 <b>{hynix:+.1%}</b>이며, "
        + ", ".join(f"{k} 평균 {v:+.1%}" for k, v in category_returns.items())
        + f"와 비교됩니다. {judgment}"
    ]
    return " ".join(lines)


def _outlook_text(forecast, last_close, avg_sentiment, rolling_result):
    horizon = len(forecast["median"])
    final_median = forecast["median"][-1]
    change_pct = (final_median - last_close) / last_close * 100
    direction = "상승" if change_pct > 0.5 else "하락" if change_pct < -0.5 else "보합"
    mood = "긍정적" if avg_sentiment > 0.2 else "부정적" if avg_sentiment < -0.2 else "중립적"

    band_low = forecast["low"][-1]
    band_high = forecast["high"][-1]

    return (
        f"Chronos-Bolt(파운데이션 모델)와 감쇠추세 모델을 앙상블한 결과, 향후 {horizon}거래일 뒤 종가를 "
        f"현재({last_close:,.0f}원) 대비 <b>{change_pct:+.2f}% ({direction})</b>한 <b>{final_median:,.0f}원</b> 안팎으로 전망합니다 "
        f"(10~90% 구간: {band_low:,.0f}원 ~ {band_high:,.0f}원). "
        f"같은 기간 수집된 국내외 뉴스 논조는 <b>{mood}</b>({avg_sentiment:+.2f})으로, 예측치에 근거리일수록 크게 반영되었습니다. "
        f"참고로 과거 {rolling_result['num_windows']}개 구간 검증에서 이 앙상블의 평균 오차율(MAPE)은 "
        f"<b>{rolling_result['mape_mean']:.1f}%</b>, 방향 적중률은 <b>{rolling_result['directional_accuracy_mean']:.0%}</b>였습니다."
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
    move_diagnosis,
    index_scan_result,
):
    os.makedirs(ASSETS_DIR, exist_ok=True)

    last_date = price_df["Date"].iloc[-1]
    last_close = price_df["Close"].iloc[-1]
    forecast_dates = _forecast_dates(last_date, len(forecast["median"]))

    chart_path = os.path.join(ASSETS_DIR, "chart.png")
    _plot_chart(price_df, forecast, forecast_dates, chart_path)

    rolling_chart_path = os.path.join(ASSETS_DIR, "rolling_backtest.png")
    _plot_rolling_chart(rolling_result, rolling_chart_path)

    scatter_path = os.path.join(ASSETS_DIR, "return_scatter.png")
    has_scatter = _plot_return_scatter(rolling_result, scatter_path)

    correlation_chart_path = os.path.join(ASSETS_DIR, "correlation.png")
    _plot_correlation_chart(price_df, related_prices, correlation_chart_path)

    from related_stocks import CATEGORY_OF

    correlation_bar_path = os.path.join(ASSETS_DIR, "correlation_bars.png")
    _plot_correlation_bars(
        [(f"{name} ({CATEGORY_OF.get(name, '기타')})", corr) for name, corr in correlations.items()],
        correlation_bar_path,
        "SK하이닉스와의 상관관계 (시장·섹터·동종업계)",
    )

    index_scan_bar_path = os.path.join(ASSETS_DIR, "index_scan_bars.png")
    index_scan_items = [
        (f"{item['name']} ({index_name.split(' ')[0]})", item["correlation"])
        for index_name, items in index_scan_result.items()
        for item in items
    ]
    if index_scan_items:
        _plot_correlation_bars(
            index_scan_items, index_scan_bar_path, "코스피200 · 나스닥100 · S&P500 대표종목 상관관계"
        )

    avg_sentiment = forecast["sentiment_score"]
    mood = "긍정적" if avg_sentiment > 0.2 else "부정적" if avg_sentiment < -0.2 else "중립적"
    outlook = _outlook_text(forecast, last_close, avg_sentiment, rolling_result)
    move_text = _move_diagnosis_text(move_diagnosis)

    lang_counts = {}
    for item in news:
        lang_counts[item.get("lang", "ko")] = lang_counts.get(item.get("lang", "ko"), 0) + 1
    lang_summary = ", ".join(
        f"{'국내' if lang == 'ko' else '해외'} {count}건" for lang, count in lang_counts.items()
    )

    scatter_section = ""
    if has_scatter:
        scatter_section = (
            '<h3>실제 vs 예측 수익률 산점도</h3>'
            '<p>롤링 백테스트에서 나온 모든 예측 구간을 "직전 종가 대비 실제 수익률(가로축)"과 '
            '"직전 종가 대비 예측 수익률(세로축)"로 점을 찍은 것입니다. 점이 빨간 대각선(y=x)에 가까울수록 '
            '정확한 예측이고, 대각선과 같은 사분면(1·3사분면)에 있으면 방향은 맞춘 것입니다.</p>'
            '<img src="assets/return_scatter.png" alt="실제 vs 예측 수익률 산점도">'
        )

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>SK하이닉스 주가 전망 리포트</title>
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
<h1>SK하이닉스(000660.KS) 주가 전망 리포트</h1>
<p class="meta">기준일: {last_date} (종가 {last_close:,.0f}원) · 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="disclaimer">
  ⚠️ 이 리포트는 공개된 사전학습 모델과 시장 데이터를 결합한 정량 분석 참고 자료입니다.
  투자 결정에 대한 법적 책임을 지지 않으며, 최종 판단과 책임은 이용자 본인에게 있습니다.
</div>

<h2>향후 전망 요약</h2>
<p>{outlook}</p>

<h2>시장·섹터·개별 종목 이슈 구분</h2>
<p>{move_text}</p>

<h2>예측 기법 설명</h2>
<ol>
  <li><b>가격 데이터</b> — Yahoo Finance에서 SK하이닉스(000660.KS) 최근 약 2년치 일별 종가를 가져옵니다.</li>
  <li><b>시장 맥락 비교</b> — 코스피/S&amp;P500/나스닥(시장 전체), 필라델피아반도체지수(반도체 섹터), 삼성전자/Micron/SanDisk(메모리 동종업계) 세 계층과 비교해 하이닉스의 움직임이 어느 층위에서 비롯됐는지 진단합니다.</li>
  <li><b>국내외 뉴스 감성분석</b> — Google News RSS로 국내(한국어) 및 해외(영어, Reuters/Bloomberg/Nikkei Asia/SCMP 등 포함) 총 14개 검색어에서 뉴스를 수집하고, 제목이 유사한 중복 기사를 하나로 합친 뒤, 한국어는 <code>KR-FinBert-SC</code>, 영어는 <code>FinBERT</code>로 각각 감성을 채점합니다. 검색어 중요도(목표주가·실적 뉴스는 가중치 상향)와 최신성(지수감쇠)을 곱해 가중평균합니다.</li>
  <li><b>가격 예측 — 3개 모델 비교</b>
    <ul>
      <li><b>Chronos-Bolt-base</b>: Amazon이 공개한 사전학습 시계열 파운데이션 모델. 별도 학습 없이 zero-shot으로 가격 분포를 예측합니다.</li>
      <li><b>감쇠추세 지수평활</b>: 고전 통계 기법으로, 추세가 시간이 갈수록 완만해진다는 가정으로 예측합니다.</li>
      <li><b>기술적 지표 GBM</b>: RSI/MACD/이동평균 이격도/변동성 등 표준 기술적 지표를 입력으로 gradient boosting 분위수 회귀가 예측합니다.</li>
    </ul>
    최종 예측치는 Chronos와 추세모델을 실측 백테스트로 검증된 비중(각 70%/30%)으로 앙상블한 값이며, 기술적 지표 모델은 비교 참고용으로 함께 표시합니다.
  </li>
  <li><b>감성 보정</b> — 위 앙상블은 가격 시계열만 입력받으므로, 뉴스 감성 점수를 예측 중앙값에 곱셈 형태로 보정합니다. 뉴스의 가격 영향력은 며칠 안에 빠르게 옅어진다고 보고, 보정 폭을 예측일이 멀어질수록 지수적으로 감쇠시킵니다.</li>
  <li><b>성능 검증</b> — 아래 "성능 분석"에서 미래 시점의 실제 값을 예측 시점에 전혀 사용하지 않고(look-ahead 없이) 과거 여러 시점을 기준으로 반복 검증합니다.</li>
</ol>

<h2>예측 차트</h2>
<img src="assets/chart.png" alt="주가 예측 차트">

<h2>향후 {len(forecast['median'])}거래일 예측</h2>
<table>
<tr><th>날짜</th><th>하한(10%)</th><th>중앙값</th><th>상한(90%)</th></tr>
{_forecast_rows_html(forecast_dates, forecast)}
</table>

<h3>모델별 예측치 비교</h3>
<p>같은 데이터를 서로 다른 방식으로 보는 3개 모델의 중앙값 예측을 나란히 비교합니다. 모델 간 편차가 크면 그만큼 불확실성이 크다는 뜻입니다.</p>
<table>
<tr><th>날짜</th><th>Chronos-Bolt</th><th>추세모델</th><th>기술적 지표(GBM)</th><th>앙상블(감성반영 전)</th><th>최종 예측(감성반영)</th></tr>
{_model_comparison_rows_html(forecast_dates, forecast)}
</table>

<h2>뉴스 감성 분석</h2>
<p>수집 뉴스 {len(news)}건({lang_summary}, 중복 통합 후) 기준 가중평균 감성 점수: <b>{avg_sentiment:+.2f}</b> ({mood}) — 다음 거래일 예측 중앙값에 {(forecast['adjustment_factor'][0] - 1):+.2%} 보정 반영 (거래일이 멀어질수록 보정 폭은 감소)</p>
<p>아래는 감성 융합에 영향을 크게 준 상위 기사입니다 (전체 {len(news)}건 중 상위 {min(25, len(news))}건 발췌):</p>
<table>
<tr><th>구분</th><th>검색어</th><th>제목</th><th>감성</th></tr>
{_news_rows_html(news, sentiment_scores)}
</table>

<h2>시장 · 섹터 · 동종업계 비교</h2>
<p>SK하이닉스의 움직임이 시장 전체/반도체 섹터/메모리 동종업계 중 어디에 더 가까운지 참고할 수 있는 최근 180일 추세 비교 및 일별 수익률 상관계수입니다.</p>
<img src="assets/correlation.png" alt="시장·섹터·동종업계 비교 차트">
<img src="assets/correlation_bars.png" alt="시장·섹터·동종업계 상관계수 막대그래프">
<table>
<tr><th>분류</th><th>종목/지수</th><th>상관계수</th><th>강도</th></tr>
{_correlation_rows_html(correlations)}
</table>

<h3>코스피200 · 나스닥100 · S&amp;P500 대표종목 상관관계</h3>
<p>각 지수 전체를 매번 스캔하기보다(수백 종목, 실행 시간·API 호출 부담), 지수를 대표하는 표본 종목들과의
상관관계를 계산해 SK하이닉스와 가장 동조하는 종목을 지수별로 보여줍니다. 전체 지수 구성종목 중 상관관계
1위라는 뜻이 아니라, 아래 표본 안에서의 순위입니다.</p>
<img src="assets/index_scan_bars.png" alt="지수별 대표종목 상관계수 막대그래프">
<table>
<tr><th>지수</th><th>종목</th><th>상관계수</th><th>강도</th></tr>
{_index_scan_rows_html(index_scan_result)}
</table>

<h2>성능 분석</h2>
<h3>최근 구간 백테스트</h3>
<p>가장 최근 {len(backtest_result['actual'])}거래일을 정답으로 숨기고, <b>그 이전 데이터까지의 정보만으로</b> 같은 방식으로 예측해본 결과입니다.</p>
<p>MAE: <b>{backtest_result['mae']:,.0f}원</b> · MAPE: <b>{backtest_result['mape']:.2f}%</b> (naive 기준 {backtest_result['naive_mape']:.2f}%) · 방향 적중률: <b>{backtest_result['directional_accuracy']:.0%}</b></p>
<table>
<tr><th>실제 종가</th><th>예측 종가</th><th>오차</th></tr>
{_backtest_rows_html(backtest_result)}
</table>

<h3>롤링 백테스트 (여러 과거 시점 반복 검증)</h3>
<p>과거 {rolling_result['num_windows']}개 시점 각각에서 "그 시점까지의 데이터만" 사용해 반복적으로 예측/채점한 결과입니다 (walk-forward, look-ahead 없음). 앙상블 모델을 Chronos 단독, naive(직전값 유지) 기준과 함께 비교합니다.</p>
<p>
  앙상블 평균 MAPE: <b>{rolling_result['mape_mean']:.2f}%</b> (표준편차 {rolling_result['mape_std']:.2f}%p) ·
  Chronos 단독 평균 MAPE: <b>{rolling_result['chronos_only_mape_mean']:.2f}%</b> ·
  naive 평균 MAPE: <b>{rolling_result['naive_mape_mean']:.2f}%</b> ·
  방향 적중률: <b>{rolling_result['directional_accuracy_mean']:.0%}</b>
</p>
<img src="assets/rolling_backtest.png" alt="롤링 백테스트 오차율 차트">
{scatter_section}

<h2>구성 요소</h2>
<ul>
  <li>주가/지수 데이터: Yahoo Finance (yfinance) — SK하이닉스, 코스피, S&amp;P500, 나스닥, 필라델피아반도체지수, 삼성전자, Micron, SanDisk</li>
  <li>뉴스: Google News RSS (한국어 7개 + 영어 7개 검색어, 유사 기사 자동 중복 제거)</li>
  <li>감성분석: <code>snunlp/KR-FinBert-SC</code>(한국어), <code>ProsusAI/finbert</code>(영어) — 둘 다 금융 도메인 특화 사전학습 모델</li>
  <li>가격 예측: <code>amazon/chronos-bolt-base</code>(사전학습 시계열 파운데이션 모델) + 감쇠추세 지수평활(statsmodels) 앙상블, 기술적 지표 기반 gradient boosting 분위수 회귀(scikit-learn) 비교 모델</li>
</ul>

</body>
</html>
"""

    with open(os.path.join(DOCS_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    return os.path.join(DOCS_DIR, "index.html")
