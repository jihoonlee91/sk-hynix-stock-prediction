import yfinance as yf

TICKER = "000660.KS"  # SK하이닉스


def fetch_price_history(period="2y"):
    ticker = yf.Ticker(TICKER)
    history = ticker.history(period=period)
    history = history.reset_index()
    history["Date"] = history["Date"].dt.strftime("%Y-%m-%d")
    return history[["Date", "Close"]]


if __name__ == "__main__":
    df = fetch_price_history()
    print(df.tail())
    print(f"rows: {len(df)}")
