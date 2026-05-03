from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def compute_macd(series: pd.Series) -> pd.Series:
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()

    return ema_12 - ema_26


def add_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])

    frames = []

    for ticker, group in prices.groupby("ticker"):
        group = group.copy()
        group["return_1d"] = group["close"].pct_change()

        group["momentum_1m"] = group["close"].pct_change(21)
        group["momentum_3m"] = group["close"].pct_change(63)
        group["momentum_6m"] = group["close"].pct_change(126)

        group["rsi_14"] = compute_rsi(group["close"], 14)
        group["macd"] = compute_macd(group["close"])
        group["volatility_30d"] = group["return_1d"].rolling(30).std()

        group["forward_return_30d"] = group["close"].shift(-21) / group["close"] - 1

        frames.append(group)

    return pd.concat(frames, ignore_index=True)


def merge_fundamentals(
    price_features: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    fundamentals = fundamentals.copy()
    fundamentals["date"] = pd.to_datetime(fundamentals["date"])

    price_features = price_features.copy()
    price_features["date"] = pd.to_datetime(price_features["date"])

    merged_frames = []

    for ticker, price_group in price_features.groupby("ticker"):
        fundamental_group = fundamentals[fundamentals["ticker"] == ticker].sort_values("date")
        price_group = price_group.sort_values("date")

        merged = pd.merge_asof(
            price_group,
            fundamental_group,
            on="date",
            by="ticker",
            direction="backward",
        )

        merged_frames.append(merged)

    merged = pd.concat(merged_frames, ignore_index=True)

    sector_median_pe = (
        merged.groupby(["date", "sector"])["pe_ratio"]
        .transform("median")
    )

    merged["sector_relative_pe"] = merged["pe_ratio"] / sector_median_pe

    return merged


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["target"] = (
        df.groupby("date")["forward_return_30d"]
        .transform(lambda x: x >= x.quantile(0.70))
        .astype(int)
    )

    return df


def build_feature_dataset() -> pd.DataFrame:
    prices = pd.read_csv(DATA_DIR / "sample_prices.csv")
    fundamentals = pd.read_csv(DATA_DIR / "sample_fundamentals.csv")

    price_features = add_price_features(prices)
    dataset = merge_fundamentals(price_features, fundamentals)
    dataset = create_target(dataset)

    feature_cols = get_feature_columns()

    dataset = dataset.dropna(
        subset=feature_cols + ["target", "forward_return_30d"]
    ).reset_index(drop=True)

    return dataset


def get_feature_columns() -> list[str]:
    return [
        "momentum_1m",
        "momentum_3m",
        "momentum_6m",
        "rsi_14",
        "macd",
        "pe_ratio",
        "sector_relative_pe",
        "roe",
        "revenue_growth",
        "volatility_30d",
    ]


if __name__ == "__main__":
    dataset = build_feature_dataset()

    print("Feature dataset built successfully")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")
    print(f"Tickers: {dataset['ticker'].nunique()}")
    print(f"Date range: {dataset['date'].min()} -> {dataset['date'].max()}")

    print("\nFeature columns:")
    for col in get_feature_columns():
        print(f"- {col}")