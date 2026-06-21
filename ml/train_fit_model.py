import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

DATA_PATH = "data/fit_scoring_data.csv"
MODEL_OUT_PATH = "ml/models/fit_model.joblib"


def train_and_evaluate():
    df = pd.read_csv(DATA_PATH)
    X = df[["skill_match_pct", "experience_diff", "location_match"]]
    y = df["is_good_fit"]

    # Handle class imbalance: weight = (negative count) / (positive count)
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, precs, recs, f1s = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

        print(f"Fold {fold}: acc={acc:.3f} precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")

    print("\n--- Average across 5 folds ---")
    print(f"Accuracy:  {sum(accs)/5:.3f}")
    print(f"Precision: {sum(precs)/5:.3f}")
    print(f"Recall:    {sum(recs)/5:.3f}")
    print(f"F1 Score:  {sum(f1s)/5:.3f}")

    # Train final model on ALL data for deployment
    final_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_OUT_PATH)
    print(f"\nFinal model trained on full dataset and saved to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    train_and_evaluate()