from pathlib import Path

import joblib
import pandas as pd

from src.ml.features import build_feature_dataset


MODEL_PATH = Path("models/lightgbm_ranker.pkl")


def load_model_artifact() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python -m src.ml.train"
        )

    return joblib.load(MODEL_PATH)


def score_stocks(tickers: list[str]) -> pd.DataFrame:
    artifact = load_model_artifact()

    model = artifact["model"]
    feature_cols = artifact["feature_columns"]

    df = build_feature_dataset()
    df["date"] = pd.to_datetime(df["date"])

    latest_rows = (
        df.sort_values("date")
        .groupby("ticker")
        .tail(1)
        .copy()
    )

    tickers = [ticker.upper() for ticker in tickers]
    latest_rows = latest_rows[latest_rows["ticker"].isin(tickers)]

    if latest_rows.empty:
        raise ValueError("No matching tickers found for scoring.")

    latest_rows["ml_score"] = model.predict_proba(latest_rows[feature_cols])[:, 1]

    result_cols = [
        "ticker",
        "date",
        "sector",
        "pe_ratio",
        "sector_relative_pe",
        "roe",
        "revenue_growth",
        "momentum_1m",
        "momentum_3m",
        "momentum_6m",
        "rsi_14",
        "volatility_30d",
        "ml_score",
    ]

    return (
        latest_rows[result_cols]
        .sort_values("ml_score", ascending=False)
        .reset_index(drop=True)
    )


def score_stock(ticker: str) -> dict:
    scored = score_stocks([ticker])

    if scored.empty:
        raise ValueError(f"No score found for ticker: {ticker}")

    row = scored.iloc[0].to_dict()

    return row