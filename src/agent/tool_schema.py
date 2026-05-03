import json

from src.tools.analyzer import analyze_stock
from src.tools.backtest import run_backtest
from src.tools.comparator import compare_stocks
from src.tools.screener import screen_stocks
from src.rag.retrieval import search_knowledge_base


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_stock",
            "description": "Run full educational analysis for one European stock using local data, ML score, and RAG evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker, for example ASML, SAP, NOVO, AIR.",
                    }
                },
                "required": ["ticker"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "screen_stocks",
            "description": "Screen European stocks by sector, valuation, and profitability. Returns a Markdown table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sector": {
                        "type": "string",
                        "description": "Sector filter. Use all if no sector is specified.",
                        "default": "all",
                    },
                    "max_pe": {
                        "type": "number",
                        "description": "Maximum P/E ratio.",
                        "default": 30.0,
                    },
                    "min_roe": {
                        "type": "number",
                        "description": "Minimum ROE as decimal. Example: 0.15 means 15%.",
                        "default": 0.0,
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_stocks",
            "description": "Compare two European stocks using local data, ML score, and RAG evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_a": {
                        "type": "string",
                        "description": "First stock ticker.",
                    },
                    "ticker_b": {
                        "type": "string",
                        "description": "Second stock ticker.",
                    },
                },
                "required": ["ticker_a", "ticker_b"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_backtest",
            "description": "Run educational monthly backtest on synthetic local data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "description": "Backtest strategy name.",
                        "default": "top_ml_score",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search company synthetic briefs using RAG by ticker and query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic search query.",
                    },
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results.",
                        "default": 3,
                    },
                },
                "required": ["query", "ticker"],
                "additionalProperties": False,
            },
        },
    },
]


def execute_tool(name: str, arguments_json: str) -> str:
    try:
        arguments = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return f"Invalid JSON arguments for tool: {name}"

    try:
        if name == "analyze_stock":
            return analyze_stock(**arguments)

        if name == "screen_stocks":
            return screen_stocks(**arguments)

        if name == "compare_stocks":
            return compare_stocks(**arguments)

        if name == "run_backtest":
            return run_backtest(**arguments)

        if name == "search_knowledge_base":
            return search_knowledge_base(**arguments)

        return f"Unknown tool: {name}"

    except Exception as exc:
        return f"Tool error in {name}: {exc}"