# ML Wizard

Educational ML foundation for European stock analysis.

Current version includes:
- local stock universe;
- synthetic price and fundamental data;
- feature engineering;
- LightGBM scoring model;
- model evaluation;
- stock screener;
- educational backtest.

> This project is for academic use only. It is not financial advice.

---

## Setup

Create virtual environment and install dependencies with `uv`:

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

Run commands through the activated environment:

```bash
python -m src.ml.generate_sample_data
python -m src.ml.train
```

Alternative without manual activation:

```bash
uv run python -m src.ml.generate_sample_data
uv run python -m src.ml.train
```

---

## Generate Data

```bash
python -m src.ml.generate_sample_data
```

Creates:
- `data/universe.csv`
- `data/sample_prices.csv`
- `data/sample_fundamentals.csv`

---

## Build Features

```bash
python -m src.ml.features
```

**Features:**
- momentum 1m / 3m / 6m;
- RSI;
- MACD;
- P/E;
- sector-relative P/E;
- ROE;
- revenue growth;
- volatility.

**Target:**
- `1` if stock is in top 30% by 30-day forward return
- `0` otherwise

---

## Train Model

```bash
python -m src.ml.train
```

Creates:
- `models/lightgbm_ranker.pkl`

**Model:** LightGBM binary classifier

---

## Evaluate Model

```bash
python -m src.ml.evaluate
```

**Current result:**

| Metric | Value |
|---|---|
| AUC | 0.5003 |
| Precision@5 | 0.6000 |
| Top bucket avg. 30d return | 3.0636% |
| Random baseline avg. 30d return | 2.1857% |
| Difference | 0.8779% |

> The model is educational. Current AUC is close to random.

---

## Run Screener

```bash
python -m src.tools.screener
```

**Default filters:**
- sector = all
- max_pe = 30.0
- min_roe = 0.0

---

## Run Backtest

```bash
python -m src.tools.backtest
```

**Current result:**

| Metric | Value |
|---|---|
| Total return | 0.66% |
| Benchmark total return | 7.80% |
| Max drawdown | -18.75% |
| Sharpe ratio | 0.14 |

**Backtest assumptions:**
- synthetic data;
- monthly rebalancing;
- top 3 stocks by model score;
- no transaction costs;
- no slippage;
- no liquidity constraints.

---

## Full Pipeline

```bash
python -m src.ml.generate_sample_data
python -m src.ml.features
python -m src.ml.train
python -m src.ml.evaluate
python -m src.tools.screener
python -m src.tools.backtest
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
│   ├── config.py
│   ├── ml/
│   │   ├── generate_sample_data.py
│   │   ├── features.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── tools/
│       ├── market_data.py
│       ├── scorer.py
│       ├── screener.py
│       └── backtest.py
├── data/
│   ├── universe.csv
│   ├── sample_prices.csv
│   └── sample_fundamentals.csv
└── models/
    └── lightgbm_ranker.pkl
```

---

## Notes

- The dataset is synthetic and reproducible.
- The model is not suitable for real trading or investment decisions.

---

## Author

Ilya Istomin