from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")


def load_universe() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "universe.csv")


def load_prices() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "sample_prices.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_fundamentals() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "sample_fundamentals.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_company_info(ticker: str) -> dict:
    ticker = ticker.upper()
    universe = load_universe()

    row = universe[universe["ticker"] == ticker]

    if row.empty:
        available = ", ".join(universe["ticker"].tolist())
        raise ValueError(f"Unknown ticker: {ticker}. Available: {available}")

    return row.iloc[0].to_dict()


def get_latest_fundamentals(ticker: str) -> dict:
    ticker = ticker.upper()
    fundamentals = load_fundamentals()

    rows = fundamentals[fundamentals["ticker"] == ticker].sort_values("date")

    if rows.empty:
        raise ValueError(f"No fundamentals found for ticker: {ticker}")

    return rows.iloc[-1].to_dict()


def get_latest_price(ticker: str) -> dict:
    ticker = ticker.upper()
    prices = load_prices()

    rows = prices[prices["ticker"] == ticker].sort_values("date")

    if rows.empty:
        raise ValueError(f"No prices found for ticker: {ticker}")

    return rows.iloc[-1].to_dict()


def get_stock_snapshot(ticker: str) -> dict:
    ticker = ticker.upper()

    company = get_company_info(ticker)
    fundamentals = get_latest_fundamentals(ticker)
    price = get_latest_price(ticker)

    return {
        "ticker": ticker,
        "company_name": company["company_name"],
        "sector": company["sector"],
        "country": company["country"],
        "currency": company["currency"],
        "latest_close": float(price["close"]),
        "price_date": str(price["date"].date()),
        "pe_ratio": float(fundamentals["pe_ratio"]),
        "roe": float(fundamentals["roe"]),
        "revenue_growth": float(fundamentals["revenue_growth"]),
        "net_margin": float(fundamentals["net_margin"]),
        "fundamentals_date": str(fundamentals["date"].date()),
    }