import numpy as np
import pandas as pd

# 실무에서 가장 널리 쓰이는 표준 기술적 지표 세트(추세/모멘텀/변동성/과매수·과매도).
# 하나하나가 이미 수십 년간 검증된 공개 공식이며, 여기서는 그대로 계산만 한다.
FEATURE_COLUMNS = [
    "ma5_ratio",
    "ma20_ratio",
    "momentum_5",
    "momentum_20",
    "volatility_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "bollinger_pct",
]


def compute_indicators(close_prices):
    """종가 리스트로부터 표준 기술적 지표를 계산해 DataFrame으로 반환한다."""
    df = pd.DataFrame({"Close": pd.Series(close_prices, dtype=float)})
    close = df["Close"]

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    df["ma5_ratio"] = close / ma5 - 1  # 단기 이동평균 대비 이격도
    df["ma20_ratio"] = close / ma20 - 1  # 중기 이동평균 대비 이격도
    df["momentum_5"] = close.pct_change(5)
    df["momentum_20"] = close.pct_change(20)
    df["volatility_20"] = close.pct_change().rolling(20).std()

    # RSI(14): Wilder(1978)의 표준 정의
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD(12, 26, 9): Appel의 표준 정의
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # 볼린저 밴드(20, 2sigma) 내 위치: -1(하단) ~ +1(상단) 근방
    std20 = close.rolling(20).std()
    df["bollinger_pct"] = (close - ma20) / (2 * std20)

    return df


if __name__ == "__main__":
    from fetch_price import fetch_price_history

    prices = fetch_price_history()["Close"].tolist()
    df = compute_indicators(prices)
    print(df[FEATURE_COLUMNS].tail())
