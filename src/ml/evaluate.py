from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.ml.features import build_feature_dataset


MODEL_PATH = Path("models/lightgbm_ranker.pkl")


def precision_at_k(y_true, scores, k: int = 5) -> float:
    temp = pd.DataFrame({"target": y_true, "score": scores})
    top_k = temp.sort_values("score", ascending=False).head(k)

    return float(top_k["target"].mean())


def evaluate_model() -> None:
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    feature_cols = artifact["feature_columns"]
    split_date = pd.to_datetime(artifact["split_date"])

    df = build_feature_dataset()
    df["date"] = pd.to_datetime(df["date"])

    test_df = df[df["date"] > split_date].copy()

    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    test_df["score"] = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, test_df["score"])
    p_at_5 = precision_at_k(y_test, test_df["score"], k=5)

    top_bucket = test_df.sort_values("score", ascending=False).head(50)
    random_baseline = test_df.sample(50, random_state=42)

    avg_return_top = top_bucket["forward_return_30d"].mean()
    avg_return_baseline = random_baseline["forward_return_30d"].mean()

    print("Model evaluation")
    print(f"AUC: {auc:.4f}")
    print(f"Precision@5: {p_at_5:.4f}")
    print(f"Average 30d return top bucket: {avg_return_top:.4%}")
    print(f"Average 30d return random baseline: {avg_return_baseline:.4%}")
    print(f"Difference: {(avg_return_top - avg_return_baseline):.4%}")


if __name__ == "__main__":
    evaluate_model()