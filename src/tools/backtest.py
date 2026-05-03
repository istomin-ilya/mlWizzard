from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.ml.features import build_feature_dataset


MODEL_PATH = Path("models/lightgbm_ranker.pkl")


def calculate_max_drawdown(returns: pd.Series) -> float:
    equity_curve = (1 + returns).cumprod()
    rolling_peak = equity_curve.cummax()
    drawdown = equity_curve / rolling_peak - 1

    return float(drawdown.min())


def run_backtest(strategy: str = "top_ml_score", top_n: int = 3) -> str:
    if strategy != "top_ml_score":
        return f"Unsupported strategy: {strategy}. Available: top_ml_score"

    if not MODEL_PATH.exists():
        return f"Model not found at {MODEL_PATH}. Run: python -m src.ml.train"

    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    feature_cols = artifact["feature_columns"]
    split_date = pd.to_datetime(artifact["split_date"])

    df = build_feature_dataset()
    df["date"] = pd.to_datetime(df["date"])

    # Use only out-of-sample period after the training split.
    df = df[df["date"] > split_date].copy()

    if df.empty:
        return "Backtest could not be calculated: no out-of-sample rows."

    df["score"] = model.predict_proba(df[feature_cols])[:, 1]
    df["month"] = df["date"].dt.to_period("M")

    monthly_results = []

    for month, group in df.groupby("month"):
        # Use only the last available observation per ticker in each month.
        rebalance_frame = (
            group.sort_values("date")
            .groupby("ticker")
            .tail(1)
            .copy()
        )

        if len(rebalance_frame) < top_n:
            continue

        selected = rebalance_frame.sort_values("score", ascending=False).head(top_n)

        strategy_return = selected["forward_return_30d"].mean()
        benchmark_return = rebalance_frame["forward_return_30d"].mean()

        if pd.isna(strategy_return) or pd.isna(benchmark_return):
            continue

        monthly_results.append(
            {
                "month": str(month),
                "selected_tickers": ", ".join(selected["ticker"].tolist()),
                "strategy_return": float(strategy_return),
                "benchmark_return": float(benchmark_return),
            }
        )

    result = pd.DataFrame(monthly_results)

    if result.empty:
        return "Backtest could not be calculated."

    total_return = (1 + result["strategy_return"]).prod() - 1
    benchmark_total_return = (1 + result["benchmark_return"]).prod() - 1

    monthly_volatility = result["strategy_return"].std()
    sharpe = (
        result["strategy_return"].mean() / monthly_volatility * np.sqrt(12)
        if monthly_volatility and monthly_volatility > 0
        else 0.0
    )

    max_drawdown = calculate_max_drawdown(result["strategy_return"])

    output = [
        "## Backtest Results",
        "",
        f"Strategy: `{strategy}`",
        f"Top N selected monthly: {top_n}",
        f"Out-of-sample periods: {len(result)}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total return | {total_return:.2%} |",
        f"| Benchmark total return | {benchmark_total_return:.2%} |",
        f"| Max drawdown | {max_drawdown:.2%} |",
        f"| Sharpe ratio | {sharpe:.2f} |",
        "",
        "## Monthly Selections",
        "",
        "| Month | Selected tickers | Strategy return | Benchmark return |",
        "|---|---|---:|---:|",
    ]

    for _, row in result.tail(8).iterrows():
        output.append(
            f"| {row['month']} "
            f"| {row['selected_tickers']} "
            f"| {row['strategy_return']:.2%} "
            f"| {row['benchmark_return']:.2%} |"
        )

    output.extend(
        [
            "",
            "Educational backtest only. The simulation uses synthetic local data, monthly rebalancing, no transaction costs, no slippage and no liquidity constraints.",
        ]
    )

    return "\n".join(output)


if __name__ == "__main__":
    print(run_backtest())