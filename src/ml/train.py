from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.ml.features import build_feature_dataset, get_feature_columns


MODEL_PATH = Path("models/lightgbm_ranker.pkl")


def train_model() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = build_feature_dataset()
    df["date"] = pd.to_datetime(df["date"])

    feature_cols = get_feature_columns()

    split_date = df["date"].quantile(0.75)

    train_df = df[df["date"] <= split_date]
    test_df = df[df["date"] > split_date]

    X_train = train_df[feature_cols]
    y_train = train_df["target"]

    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=7,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1,
        force_col_wise=True,
    )

    model.fit(X_train, y_train)

    predictions = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, predictions)

    artifact = {
        "model": model,
        "feature_columns": feature_cols,
        "split_date": str(split_date.date()),
        "auc": float(auc),
    }

    joblib.dump(artifact, MODEL_PATH)

    print("Model trained successfully")
    print(f"Saved to: {MODEL_PATH}")
    print(f"Rows train: {len(train_df)}")
    print(f"Rows test: {len(test_df)}")
    print(f"AUC: {auc:.4f}")


if __name__ == "__main__":
    train_model()