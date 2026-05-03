from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
DATA_DIR = Path("data")


COMPANIES = [
    {
        "ticker": "ASML",
        "company_name": "ASML Holding",
        "sector": "Technology",
        "country": "Netherlands",
        "currency": "EUR",
    },
    {
        "ticker": "SAP",
        "company_name": "SAP SE",
        "sector": "Technology",
        "country": "Germany",
        "currency": "EUR",
    },
    {
        "ticker": "NOVO",
        "company_name": "Novo Nordisk",
        "sector": "Healthcare",
        "country": "Denmark",
        "currency": "DKK",
    },
    {
        "ticker": "NESN",
        "company_name": "Nestle",
        "sector": "Consumer Defensive",
        "country": "Switzerland",
        "currency": "CHF",
    },
    {
        "ticker": "AIR",
        "company_name": "Airbus",
        "sector": "Industrials",
        "country": "France",
        "currency": "EUR",
    },
    {
        "ticker": "MC",
        "company_name": "LVMH",
        "sector": "Consumer Cyclical",
        "country": "France",
        "currency": "EUR",
    },
    {
        "ticker": "SIE",
        "company_name": "Siemens",
        "sector": "Industrials",
        "country": "Germany",
        "currency": "EUR",
    },
    {
        "ticker": "TTE",
        "company_name": "TotalEnergies",
        "sector": "Energy",
        "country": "France",
        "currency": "EUR",
    },
    {
        "ticker": "SU",
        "company_name": "Schneider Electric",
        "sector": "Industrials",
        "country": "France",
        "currency": "EUR",
    },
    {
        "ticker": "CAP",
        "company_name": "Capgemini",
        "sector": "Technology",
        "country": "France",
        "currency": "EUR",
    },
]


SECTOR_PROFILES = {
    "Technology": {
        "pe": 28.0,
        "roe": 0.24,
        "revenue_growth": 0.11,
        "net_margin": 0.18,
        "drift": 0.00035,
        "volatility": 0.018,
    },
    "Healthcare": {
        "pe": 31.0,
        "roe": 0.32,
        "revenue_growth": 0.13,
        "net_margin": 0.25,
        "drift": 0.00040,
        "volatility": 0.016,
    },
    "Consumer Defensive": {
        "pe": 22.0,
        "roe": 0.18,
        "revenue_growth": 0.04,
        "net_margin": 0.13,
        "drift": 0.00015,
        "volatility": 0.010,
    },
    "Industrials": {
        "pe": 21.0,
        "roe": 0.17,
        "revenue_growth": 0.07,
        "net_margin": 0.11,
        "drift": 0.00025,
        "volatility": 0.014,
    },
    "Consumer Cyclical": {
        "pe": 26.0,
        "roe": 0.20,
        "revenue_growth": 0.08,
        "net_margin": 0.16,
        "drift": 0.00022,
        "volatility": 0.017,
    },
    "Energy": {
        "pe": 11.0,
        "roe": 0.15,
        "revenue_growth": 0.03,
        "net_margin": 0.10,
        "drift": 0.00010,
        "volatility": 0.020,
    },
}


def generate_universe() -> pd.DataFrame:
    return pd.DataFrame(COMPANIES)


def generate_prices(universe: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.bdate_range("2022-01-03", "2024-12-31")

    rows = []

    for _, company in universe.iterrows():
        ticker = company["ticker"]
        sector = company["sector"]
        profile = SECTOR_PROFILES[sector]

        price = rng.uniform(40, 700)
        company_quality = rng.normal(0.0, 0.00015)

        for date in dates:
            daily_return = rng.normal(
                profile["drift"] + company_quality,
                profile["volatility"],
            )

            close = max(2.0, price * (1.0 + daily_return))
            open_price = price * (1.0 + rng.normal(0, 0.004))
            high = max(open_price, close) * (1.0 + abs(rng.normal(0, 0.006)))
            low = min(open_price, close) * (1.0 - abs(rng.normal(0, 0.006)))
            volume = int(rng.integers(300_000, 5_000_000))

            rows.append(
                {
                    "ticker": ticker,
                    "date": date.date().isoformat(),
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close, 2),
                    "volume": volume,
                }
            )

            price = close

    return pd.DataFrame(rows)


def generate_fundamentals(universe: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 1)
    dates = pd.date_range("2022-03-31", "2024-12-31", freq="QE")

    rows = []

    for _, company in universe.iterrows():
        ticker = company["ticker"]
        sector = company["sector"]
        profile = SECTOR_PROFILES[sector]

        company_quality = rng.normal(0.0, 0.03)

        for date in dates:
            pe_ratio = max(4.0, rng.normal(profile["pe"], profile["pe"] * 0.15))
            roe = max(0.01, rng.normal(profile["roe"] + company_quality, 0.04))
            revenue_growth = rng.normal(profile["revenue_growth"] + company_quality / 2, 0.04)
            net_margin = max(0.01, rng.normal(profile["net_margin"] + company_quality / 2, 0.03))

            rows.append(
                {
                    "ticker": ticker,
                    "date": date.date().isoformat(),
                    "sector": sector,
                    "pe_ratio": round(pe_ratio, 2),
                    "roe": round(roe, 4),
                    "revenue_growth": round(revenue_growth, 4),
                    "net_margin": round(net_margin, 4),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    universe = generate_universe()
    prices = generate_prices(universe)
    fundamentals = generate_fundamentals(universe)

    universe.to_csv(DATA_DIR / "universe.csv", index=False)
    prices.to_csv(DATA_DIR / "sample_prices.csv", index=False)
    fundamentals.to_csv(DATA_DIR / "sample_fundamentals.csv", index=False)

    print("Generated:")
    print(f"- {DATA_DIR / 'universe.csv'}")
    print(f"- {DATA_DIR / 'sample_prices.csv'}")
    print(f"- {DATA_DIR / 'sample_fundamentals.csv'}")
    print(f"Rows prices: {len(prices)}")
    print(f"Rows fundamentals: {len(fundamentals)}")


if __name__ == "__main__":
    main()