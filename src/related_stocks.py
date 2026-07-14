import pandas as pd
import yfinance as yf

# 메모리 반도체 업황을 함께 좌우하는 대표 종목들 (상관관계 참고용)
RELATED_TICKERS = {
    "삼성전자": "005930.KS",
    "Micron": "MU",
    "SanDisk": "SNDK",
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


if __name__ == "__main__":
    from fetch_price import fetch_price_history

    hynix_df = fetch_price_history()
    related = fetch_related_prices()
    corrs = compute_correlations(hynix_df, related)
    for name, corr in corrs.items():
        print(f"{name}: {corr:+.2f}")
