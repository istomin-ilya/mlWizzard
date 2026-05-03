# ML Wizard

Educational AI agent for European stock analysis.

Current version includes:
- synthetic European stock dataset;
- LightGBM scoring model;
- model evaluation;
- stock screener;
- educational backtest;
- RAG over synthetic company briefs;
- OpenAI tool-calling agent;
- FastMCP server with tools, resources, and prompts.

> This project is for academic use only. It is not financial advice.

---

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

**Windows:**
```bash
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

Create `.env`:

```bash
cp .env.example .env
```

Required variables:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
CHROMA_PATH=./data/chroma
MODEL_PATH=./models/lightgbm_ranker.pkl
MAX_AGENT_STEPS=5
```

---

## Data

Generate local data:

```bash
uv run python -m src.ml.generate_sample_data
```

Creates:
- `data/universe.csv`
- `data/sample_prices.csv`
- `data/sample_fundamentals.csv`

**Universe:**

| Ticker | Company |
|---|---|
| ASML | ASML Holding |
| SAP | SAP SE |
| NOVO | Novo Nordisk |
| NESN | Nestle |
| AIR | Airbus |
| MC | LVMH |
| SIE | Siemens |
| TTE | TotalEnergies |
| SU | Schneider Electric |
| CAP | Capgemini |

The dataset is synthetic and reproducible.

---

## ML

Build features:

```bash
uv run python -m src.ml.features
```

Train model:

```bash
uv run python -m src.ml.train
```

Evaluate:

```bash
uv run python -m src.ml.evaluate
```

**Model:** LightGBM binary classifier

**Target:**
- `1` if stock is in top 30% by 30-day forward return
- `0` otherwise

**Features:**
- momentum 1m / 3m / 6m;
- RSI;
- MACD;
- P/E;
- sector-relative P/E;
- ROE;
- revenue growth;
- volatility.

**Current evaluation:**

| Metric | Value |
|---|---|
| AUC | 0.5003 |
| Precision@5 | 0.6000 |
| Top bucket avg. 30d return | 3.0636% |
| Random baseline avg. 30d return | 2.1857% |
| Difference | 0.8779% |

> The model is educational. It is not suitable for real trading.

---

## Screener

```bash
uv run python -m src.tools.screener
```

**Default filters:**
- sector = all
- max_pe = 30.0
- min_roe = 0.0

---

## Backtest

```bash
uv run python -m src.tools.backtest
```

**Current result:**

| Metric | Value |
|---|---|
| Total return | 0.66% |
| Benchmark total return | 7.80% |
| Max drawdown | -18.75% |
| Sharpe ratio | 0.14 |

**Assumptions:**
- synthetic data;
- monthly rebalancing;
- top 3 stocks by ML score;
- no transaction costs;
- no slippage;
- no liquidity constraints.

---

## RAG

Build ChromaDB index:

```bash
uv run python -m src.rag.ingestion
```

Test retrieval:

```bash
uv run python -m src.rag.retrieval
```

**Corpus:**

```
data/reports/
├── ASML_2025.txt
├── SAP_2025.txt
├── NOVO_2025.txt
├── NESN_2025.txt
├── AIR_2025.txt
├── MC_2025.txt
├── SIE_2025.txt
├── TTE_2025.txt
├── SU_2025.txt
└── CAP_2025.txt
```

The corpus contains synthetic educational company briefs.

---

## Agent CLI

```bash
uv run python -m src.cli "Analyze ASML"
uv run python -m src.cli "Find European technology stocks with P/E below 30"
uv run python -m src.cli "Compare SAP and ASML"
uv run python -m src.cli "Run backtest"
```

The agent uses OpenAI tool calling to select project tools and generate structured educational answers.

**Implemented tools:**
- `analyze_stock`
- `screen_stocks`
- `compare_stocks`
- `run_backtest`
- `search_knowledge_base`

---

## MCP Server

Run server:

```bash
uv run python -m src.mcp.server
```

Test with MCP Inspector:

```bash
npx @modelcontextprotocol/inspector uv run python -m src.mcp.server
```

**Tools:**
- `mcp_analyze_stock`
- `mcp_screen_stocks`
- `mcp_compare_stocks`
- `mcp_run_backtest`
- `mcp_search_knowledge_base`

**Resources:**
- `stocks://universe`
- `stocks://methodology`
- `stocks://model-card`

**Prompts:**
- `equity_analysis_prompt`
- `risk_review_prompt`

---

## Full Pipeline

```bash
uv run python -m src.ml.generate_sample_data
uv run python -m src.ml.features
uv run python -m src.ml.train
uv run python -m src.ml.evaluate
uv run python -m src.tools.screener
uv run python -m src.tools.backtest
uv run python -m src.rag.ingestion
uv run python -m src.rag.retrieval
uv run python -m src.cli "Analyze ASML"
```

---

## Project Structure

```
ml-wizard/
├── README.md
├── .gitignore
├── .env.example
├── requirements.txt
├── src/
│   ├── cli.py
│   ├── config.py
│   ├── agent/
│   │   ├── prompts.py
│   │   ├── tool_schema.py
│   │   └── orchestrator.py
│   ├── mcp/
│   │   └── server.py
│   ├── ml/
│   │   ├── generate_sample_data.py
│   │   ├── features.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── rag/
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── ingestion.py
│   │   └── retrieval.py
│   └── tools/
│       ├── analyzer.py
│       ├── backtest.py
│       ├── comparator.py
│       ├── market_data.py
│       ├── scorer.py
│       └── screener.py
├── data/
│   ├── universe.csv
│   ├── sample_prices.csv
│   ├── sample_fundamentals.csv
│   └── reports/
├── models/
│   └── lightgbm_ranker.pkl
└── docs/
    └── screenshots/
```

---

## Limitations

- synthetic data;
- small stock universe;
- educational company briefs;
- weak ML predictive quality;
- no transaction costs in backtest;
- no real investment recommendations.

---

## Author

Ilya Istomin