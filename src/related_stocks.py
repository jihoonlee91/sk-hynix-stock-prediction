import pandas as pd
import yfinance as yf

# SK하이닉스 주가 변동이 (1) 증시 전반의 흐름인지, (2) 반도체 섹터 전반의 흐름인지,
# (3) 메모리 업황 동조인지, (4) SK하이닉스 고유 이슈인지 구분하기 위해 세 계층으로
# 비교군을 나눈다. 계층이 다른 비교군과 상관관계를 나란히 보면 "오늘 하이닉스가
# 오른 게 시장이 다 올라서인지, 반도체만 오른 건지, 하이닉스만 오른 건지" 판별할 수 있다.
MARKET_INDICES = {
    "코스피": "^KS11",
    "S&P500": "^GSPC",
    "나스닥": "^IXIC",
}
SECTOR_INDEX = {
    "필라델피아반도체지수(SOX)": "^SOX",
}
PEER_STOCKS = {
    "삼성전자": "005930.KS",
    "Micron": "MU",
    "SanDisk": "SNDK",
}

RELATED_TICKERS = {**MARKET_INDICES, **SECTOR_INDEX, **PEER_STOCKS}

CATEGORY_OF = {
    **{name: "시장 전체" for name in MARKET_INDICES},
    **{name: "반도체 섹터" for name in SECTOR_INDEX},
    **{name: "메모리 동종업계" for name in PEER_STOCKS},
}


def fetch_related_prices(period="2y"):
    prices = {}
    for name, ticker in RELATED_TICKERS.items():
        history = yf.Ticker(ticker).history(period=period)
        if history.empty:
            continue
        history = history.reset_index()
        history["Date"] = history["Date"].dt.strftime("%Y-%m-%d")
        prices[name] = history[["Date", "Close"]]
    return prices


def compute_correlations(hynix_df, related_prices):
    """일별 수익률(변화율) 기준 상관계수. 원가격 상관보다 추세/변동 동조성을 더 잘 보여준다."""
    hynix_returns = hynix_df.set_index("Date")["Close"].pct_change().dropna()

    correlations = {}
    for name, df in related_prices.items():
        other_returns = df.set_index("Date")["Close"].pct_change().dropna()
        joined = pd.concat([hynix_returns, other_returns], axis=1, join="inner")
        joined.columns = ["hynix", "other"]
        if len(joined) < 10:
            continue
        correlations[name] = joined["hynix"].corr(joined["other"])

    return correlations


def diagnose_move(hynix_df, related_prices, recent_days=5):
    """최근 며칠간의 수익률을 계층별(시장/섹터/동종업계)로 비교해, 하이닉스의
    움직임이 어느 층위에서 기인했는지 짧은 진단 문구를 만든다.

    예: 시장지수는 거의 안 움직였는데 하이닉스만 크게 움직였다면 "종목 고유 이슈"로,
    시장지수도 비슷한 방향/크기로 움직였다면 "시장 전체 흐름에 동조"로 판단한다.
    """
    hynix_returns = hynix_df.set_index("Date")["Close"].pct_change().dropna()
    hynix_recent = hynix_returns.tail(recent_days)
    hynix_cum = (1 + hynix_recent).prod() - 1

    category_cum = {}
    for name, df in related_prices.items():
        other_returns = df.set_index("Date")["Close"].pct_change().dropna()
        joined = pd.concat([hynix_returns, other_returns], axis=1, join="inner").tail(recent_days)
        if joined.empty:
            continue
        other_cum = (1 + joined.iloc[:, 1]).prod() - 1
        category = CATEGORY_OF.get(name, "기타")
        category_cum.setdefault(category, []).append(other_cum)

    category_avg = {cat: sum(vals) / len(vals) for cat, vals in category_cum.items()}

    return {
        "hynix_cumulative_return": hynix_cum,
        "category_cumulative_return": category_avg,
        "relative_to_market": hynix_cum - category_avg.get("시장 전체", 0.0),
        "relative_to_sector": hynix_cum - category_avg.get("반도체 섹터", 0.0),
        "relative_to_peers": hynix_cum - category_avg.get("메모리 동종업계", 0.0),
    }


if __name__ == "__main__":
    from fetch_price import fetch_price_history

    hynix_df = fetch_price_history()
    related = fetch_related_prices()
    corrs = compute_correlations(hynix_df, related)
    for name, corr in corrs.items():
        print(f"{name} ({CATEGORY_OF.get(name, '기타')}): {corr:+.2f}")

    diagnosis = diagnose_move(hynix_df, related)
    print("\n최근 5거래일 누적수익률 비교:")
    print(f"  SK하이닉스: {diagnosis['hynix_cumulative_return']:+.2%}")
    for cat, ret in diagnosis["category_cumulative_return"].items():
        print(f"  {cat} 평균: {ret:+.2%}")
