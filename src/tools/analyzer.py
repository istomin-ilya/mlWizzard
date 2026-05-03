from src.tools.market_data import get_stock_snapshot
from src.tools.scorer import score_stock
from src.rag.retrieval import search_knowledge_base


def analyze_stock(ticker: str) -> str:
    ticker = ticker.upper()

    snapshot = get_stock_snapshot(ticker)
    score = score_stock(ticker)

    try:
        rag = search_knowledge_base(
            query=f"{ticker} business strengths risks growth outlook",
            ticker=ticker,
            n_results=3,
        )
    except Exception as exc:
        rag = f"RAG unavailable: {exc}"

    output = [
        f"## Raw Analysis Data: {ticker}",
        "",
        "### Company Snapshot",
        f"Company: {snapshot['company_name']}",
        f"Sector: {snapshot['sector']}",
        f"Country: {snapshot['country']}",
        f"Latest close: {snapshot['latest_close']:.2f} {snapshot['currency']}",
        f"P/E: {snapshot['pe_ratio']:.2f}",
        f"ROE: {snapshot['roe']:.2%}",
        f"Revenue growth: {snapshot['revenue_growth']:.2%}",
        f"Net margin: {snapshot['net_margin']:.2%}",
        "",
        "### ML Score",
        f"Score: {score['ml_score']:.3f}",
        f"Momentum 1m: {score['momentum_1m']:.2%}",
        f"Momentum 3m: {score['momentum_3m']:.2%}",
        f"Momentum 6m: {score['momentum_6m']:.2%}",
        f"RSI 14: {score['rsi_14']:.2f}",
        f"Volatility 30d: {score['volatility_30d']:.2%}",
        "",
        "### Retrieved Evidence",
        rag,
    ]

    return "\n".join(output)


if __name__ == "__main__":
    print(analyze_stock("ASML"))