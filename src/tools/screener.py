import pandas as pd

from src.tools.market_data import load_universe
from src.tools.scorer import score_stocks


def screen_stocks(
    sector: str = "all",
    max_pe: float = 30.0,
    min_roe: float = 0.0,
) -> str:
    universe = load_universe()

    tickers = universe["ticker"].tolist()
    scored = score_stocks(tickers)

    if sector != "all":
        scored = scored[scored["sector"].str.lower() == sector.lower()]

    scored = scored[
        (scored["pe_ratio"] <= max_pe)
        & (scored["roe"] >= min_roe)
    ]

    if scored.empty:
        return "No stocks matched the screening criteria."

    return format_screening_results(scored)


def format_screening_results(df: pd.DataFrame) -> str:
    output = ["## Screening Results", ""]

    output.append(
        "| Ticker | Sector | P/E | ROE | Revenue Growth | ML Score |"
    )
    output.append(
        "|---|---|---:|---:|---:|---:|"
    )

    for _, row in df.iterrows():
        output.append(
            f"| {row['ticker']} "
            f"| {row['sector']} "
            f"| {row['pe_ratio']:.2f} "
            f"| {row['roe']:.2%} "
            f"| {row['revenue_growth']:.2%} "
            f"| {row['ml_score']:.3f} |"
        )

    return "\n".join(output)


if __name__ == "__main__":
    print(screen_stocks(sector="all", max_pe=30.0))