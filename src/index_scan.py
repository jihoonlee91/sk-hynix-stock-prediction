"""코스피200/나스닥100/S&P500 전체 구성종목을 매번 스캔하는 것은 (수백 개 티커를
매 실행마다 내려받아야 해 느리고 yfinance 레이트리밋에 취약하다) 이 파이프라인의
"매일 돌려서 정적 리포트를 재생성한다"는 목적에 맞지 않는다. 대신 각 지수를 대표하는
대형주/업종 대표주 표본을 골라, SK하이닉스와 상관관계가 높은 순으로 보여준다.
표본은 전체 구성종목의 일부이므로 "이 지수 안에서 가장 상관관계가 높은 종목"이 아니라
"대표 표본 안에서 상관관계가 높은 종목"이라는 점을 리포트에 명시한다.
"""

import pandas as pd
import yfinance as yf

KOSPI200_SAMPLE = {
    "삼성전자": "005930.KS",
    "NAVER": "035420.KS",
    "카카오": "035720.KS",
    "현대차": "005380.KS",
    "기아": "000270.KS",
    "LG화학": "051910.KS",
    "삼성SDI": "006400.KS",
    "KB금융": "105560.KS",
    "신한지주": "055550.KS",
    "한국전력": "015760.KS",
    "삼성전기": "009150.KS",
    "고려아연": "010130.KS",
    "현대모비스": "012330.KS",
    "LG전자": "066570.KS",
    "한미반도체": "042700.KS",
    "하나금융지주": "086790.KS",
    "포스코홀딩스": "005490.KS",
}

NASDAQ100_SAMPLE = {
    "Nvidia": "NVDA",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Alphabet": "GOOGL",
    "Meta": "META",
    "Broadcom": "AVGO",
    "Tesla": "TSLA",
    "AMD": "AMD",
    "Intel": "INTC",
    "Qualcomm": "QCOM",
    "Texas Instruments": "TXN",
    "Analog Devices": "ADI",
    "Lam Research": "LRCX",
    "Applied Materials": "AMAT",
    "KLA": "KLAC",
    "ASML": "ASML",
    "Cisco": "CSCO",
    "Adobe": "ADBE",
    "Netflix": "NFLX",
    "Costco": "COST",
    "Palantir": "PLTR",
}

SP500_SAMPLE = {
    "JPMorgan": "JPM",
    "Bank of America": "BAC",
    "Exxon Mobil": "XOM",
    "Chevron": "CVX",
    "Johnson & Johnson": "JNJ",
    "UnitedHealth": "UNH",
    "Procter & Gamble": "PG",
    "Coca-Cola": "KO",
    "Walmart": "WMT",
    "Home Depot": "HD",
    "Disney": "DIS",
    "Visa": "V",
    "Mastercard": "MA",
    "Goldman Sachs": "GS",
    "Caterpillar": "CAT",
    "Boeing": "BA",
    "GE Aerospace": "GE",
    "NextEra Energy": "NEE",
    "Linde": "LIN",
    "Eli Lilly": "LLY",
}

INDEX_UNIVERSES = {
    "코스피200 (대표종목)": KOSPI200_SAMPLE,
    "나스닥100 (대표종목)": NASDAQ100_SAMPLE,
    "S&P500 (대표종목)": SP500_SAMPLE,
}


def _normalize_date_index(index):
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    return index.strftime("%Y-%m-%d")


def scan_index_correlations(hynix_df, period="1y", top_n=8):
    hynix_returns = hynix_df.set_index("Date")["Close"].pct_change().dropna()
    hynix_returns.index = pd.to_datetime(hynix_returns.index).strftime("%Y-%m-%d")

    results = {}
    for index_name, universe in INDEX_UNIVERSES.items():
        tickers = list(universe.values())
        name_by_ticker = {v: k for k, v in universe.items()}

        try:
            data = yf.download(tickers, period=period, progress=False, auto_adjust=True)["Close"]
        except Exception:
            results[index_name] = []
            continue

        if isinstance(data, pd.Series):
            data = data.to_frame(tickers[0])
        data.index = _normalize_date_index(data.index)

        correlations = []
        for ticker in tickers:
            if ticker not in data.columns:
                continue
            other_returns = data[ticker].dropna().pct_change().dropna()
            joined = pd.concat([hynix_returns, other_returns], axis=1, join="inner")
            joined.columns = ["hynix", "other"]
            if len(joined) < 30:
                continue
            corr = joined["hynix"].corr(joined["other"])
            if pd.isna(corr):
                continue
            correlations.append({"name": name_by_ticker[ticker], "ticker": ticker, "correlation": float(corr)})

        correlations.sort(key=lambda c: abs(c["correlation"]), reverse=True)
        results[index_name] = correlations[:top_n]

    return results


if __name__ == "__main__":
    from fetch_price import fetch_price_history

    df = fetch_price_history()
    scan = scan_index_correlations(df)
    for index_name, top in scan.items():
        print(f"\n{index_name}:")
        for item in top:
            print(f"  {item['name']} ({item['ticker']}): {item['correlation']:+.2f}")
