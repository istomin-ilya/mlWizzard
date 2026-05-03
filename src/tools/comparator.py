from src.tools.analyzer import analyze_stock


def compare_stocks(ticker_a: str, ticker_b: str) -> str:
    ticker_a = ticker_a.upper()
    ticker_b = ticker_b.upper()

    analysis_a = analyze_stock(ticker_a)
    analysis_b = analyze_stock(ticker_b)

    return "\n\n".join(
        [
            f"## Raw Comparison Data: {ticker_a} vs {ticker_b}",
            "",
            f"### {ticker_a}",
            analysis_a,
            "",
            f"### {ticker_b}",
            analysis_b,
        ]
    )


if __name__ == "__main__":
    print(compare_stocks("SAP", "ASML"))