import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib

DATA_PATH = "data/salary_data.csv"
MODEL_OUT_PATH = "ml/models/salary_model.joblib"


def train_and_evaluate():
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c != "salary"]
    X = df[feature_cols]
    y = df["salary"]

    # 5-fold CV (regular KFold since this is regression, not classification)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    maes, r2s, rmses = [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)

        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

        print(f"Fold {fold}: MAE=₹{mae:,.0f}  RMSE=₹{rmse:,.0f}  R²={r2:.3f}")

    print("\n--- Average across 5 folds ---")
    print(f"MAE:  ₹{sum(maes)/5:,.0f}")
    print(f"RMSE: ₹{sum(rmses)/5:,.0f}")
    print(f"R²:   {sum(r2s)/5:.3f}")

    # Train final model on all data
    final_model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_OUT_PATH)

    # Save the feature column order - we'll need this at inference time
    joblib.dump(feature_cols, "ml/models/salary_feature_columns.joblib")

    print(f"\nFinal model saved to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    train_and_evaluate() 