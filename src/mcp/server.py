from fastmcp import FastMCP

from src.rag.retrieval import search_knowledge_base
from src.tools.analyzer import analyze_stock
from src.tools.backtest import run_backtest
from src.tools.comparator import compare_stocks
from src.tools.market_data import load_universe
from src.tools.screener import screen_stocks


mcp = FastMCP("ML Wizard")


@mcp.tool()
def mcp_analyze_stock(ticker: str) -> str:
    """Run full educational stock analysis using local data, ML score, and RAG evidence."""
    return analyze_stock(ticker)


@mcp.tool()
def mcp_screen_stocks(
    sector: str = "all",
    max_pe: float = 30.0,
    min_roe: float = 0.0,
) -> str:
    """Screen European stocks by sector, P/E, ROE, and ML score."""
    return screen_stocks(
        sector=sector,
        max_pe=max_pe,
        min_roe=min_roe,
    )


@mcp.tool()
def mcp_compare_stocks(ticker_a: str, ticker_b: str) -> str:
    """Compare two European stocks using local data, ML score, and RAG evidence."""
    return compare_stocks(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
    )


@mcp.tool()
def mcp_run_backtest(strategy: str = "top_ml_score") -> str:
    """Run educational backtest on synthetic local stock data."""
    return run_backtest(strategy=strategy)


@mcp.tool()
def mcp_search_knowledge_base(
    query: str,
    ticker: str,
    n_results: int = 3,
) -> str:
    """Search synthetic company briefs using ChromaDB semantic retrieval."""
    return search_knowledge_base(
        query=query,
        ticker=ticker,
        n_results=n_results,
    )


@mcp.resource("stocks://universe")
def get_universe() -> str:
    """Return the local European stock universe."""
    universe = load_universe()

    return universe.to_csv(index=False)


@mcp.resource("stocks://methodology")
def get_methodology() -> str:
    """Return project methodology."""
    return """
ML Wizard methodology

Data:
- local synthetic stock prices
- local synthetic fundamentals
- synthetic educational company briefs

ML:
- LightGBM binary classifier
- target: top 30% by 30-day forward return
- features: momentum, RSI, MACD, P/E, sector-relative P/E, ROE, revenue growth, volatility

RAG:
- local text files
- chunking
- OpenAI embeddings
- ChromaDB vector store
- ticker-filtered semantic retrieval

Backtest:
- monthly rebalancing
- top 3 stocks by ML score
- benchmark: equal-weight local universe

Limitations:
- synthetic data
- small universe
- no transaction costs
- no slippage
- educational use only
- not financial advice
""".strip()


@mcp.resource("stocks://model-card")
def get_model_card() -> str:
    """Return the educational model card."""
    return """
Model Card

Model:
- LightGBM binary classifier

Purpose:
- educational stock scoring

Target:
- 1 if stock belongs to top 30% by 30-day forward return
- 0 otherwise

Features:
- momentum_1m
- momentum_3m
- momentum_6m
- rsi_14
- macd
- pe_ratio
- sector_relative_pe
- roe
- revenue_growth
- volatility_30d

Current evaluation:
- AUC: about 0.50
- Precision@5: about 0.60

Important:
- model quality is close to random on the synthetic dataset
- model is not suitable for trading
- output is not financial advice
""".strip()


@mcp.prompt()
def equity_analysis_prompt(ticker: str) -> str:
    """Prompt template for educational equity analysis."""
    return f"""
Analyze {ticker} using the available ML Wizard tools.

Required structure:
- educational thesis
- financial metrics
- ML score
- evidence from documents
- risks
- educational verdict
- disclaimer

Do not provide financial advice.
Do not use BUY, SELL, or STRONG BUY.
""".strip()


@mcp.prompt()
def risk_review_prompt(ticker: str) -> str:
    """Prompt template for risk-focused company review."""
    return f"""
Review the main risks for {ticker}.

Use available company data and RAG evidence.

Focus on:
- business risk
- valuation risk
- market risk
- regulatory or geopolitical risk
- data limitations

Return concise educational analysis only.
""".strip()


if __name__ == "__main__":
    mcp.run()