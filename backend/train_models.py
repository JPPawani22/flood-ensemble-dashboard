"""Train and compare bagging, boosting, and hybrid stacking regressors."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from backend.preprocessing import prepare_features, build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "sri_lanka_flood_risk_dataset_25000.csv"
MODEL_DIR = ROOT / "backend" / "models"
OUTPUT_DIR = ROOT / "output"


def regression_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {"R2": float(r2_score(actual, predicted)),
            "MSE": float(mse),
            "RMSE": float(np.sqrt(mse))}


def make_models(seed=42):
    return {
        "Random Forest (Bagging)": RandomForestRegressor(n_estimators=180, max_depth=16,
                                                          min_samples_leaf=2, n_jobs=-1,
                                                          random_state=seed),
        "XGBoost (Boosting)": XGBRegressor(n_estimators=350, learning_rate=0.05, max_depth=6,
                                            subsample=0.8, colsample_bytree=0.8,
                                            objective="reg:squarederror", n_jobs=-1, random_state=seed),
        "LightGBM (Boosting)": LGBMRegressor(n_estimators=350, learning_rate=0.05,
                                               num_leaves=31, verbosity=-1, n_jobs=-1,
                                               random_state=seed),
        "CatBoost (Boosting)": CatBoostRegressor(iterations=350, learning_rate=0.05,
                                                   depth=6, verbose=0, random_seed=seed),
    }


def train(data_path=DEFAULT_DATA, target="flood_risk_score", drop_columns=None, test_size=0.2, seed=42):
    data = data_path.copy() if isinstance(data_path, pd.DataFrame) else pd.read_csv(data_path)
    X, y, numeric, categorical, dropped = prepare_features(data, target, drop_columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    templates = make_models(seed)
    fitted, predictions, results = {}, {}, []
    for name, estimator in templates.items():
        pipe = Pipeline([("preprocessor", build_preprocessor(numeric, categorical)),
                         ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name], predictions[name] = pipe, pred
        results.append({"Model": name, **regression_metrics(y_test, pred)})

    # The ANN learns from out-of-fold predictions of diverse bagging/boosting models.
    base = [(key, templates[key]) for key in ("Random Forest (Bagging)",
                                               "XGBoost (Boosting)",
                                               "LightGBM (Boosting)")]
    stack = StackingRegressor(estimators=base,
        final_estimator=MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=350,
                                     early_stopping=True, random_state=seed),
        cv=5, n_jobs=-1)
    stack_pipe = Pipeline([("preprocessor", build_preprocessor(numeric, categorical)),
                           ("model", stack)])
    stack_pipe.fit(X_train, y_train)
    predictions["RF + XGB + LightGBM + ANN Stack"] = stack_pipe.predict(X_test)
    fitted["RF + XGB + LightGBM + ANN Stack"] = stack_pipe
    results.append({"Model": "RF + XGB + LightGBM + ANN Stack",
                    **regression_metrics(y_test, predictions["RF + XGB + LightGBM + ANN Stack"])})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(results).sort_values("RMSE")
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    # Persist a complete run so dashboard selection and explanations use matching data.
    joblib.dump({"models": fitted, "comparison": comparison,
                 "feature_columns": X.columns.tolist(), "target": target,
                 "numeric_columns": numeric, "categorical_columns": categorical,
                 "dropped_columns": dropped, "test_features": X_test,
                 "test_actual": y_test, "test_predictions": predictions,
                 "split_seed": seed, "test_size": test_size},
                MODEL_DIR / "training_bundle.pkl", compress=3)
    # Canonical artifact name makes the stacking method explicit. Keep the
    # earlier filename as an identical compatibility copy for existing users.
    stacking_path = MODEL_DIR / "stacking_model.pkl"
    joblib.dump(stack_pipe, stacking_path, compress=3)
    (MODEL_DIR / "hybrid_stack_model.pkl").write_bytes(stacking_path.read_bytes())
    metadata = {"dataset": Path(data_path).name if not isinstance(data_path, pd.DataFrame) else "uploaded CSV",
                "rows_used": len(X), "features_used": X.shape[1],
                "target": target, "numeric_features": numeric, "categorical_features": categorical,
                "excluded_columns": dropped, "test_size": test_size, "random_state": seed}
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Input CSV path")
    parser.add_argument("--target", default="flood_risk_score", help="Numeric regression target column")
    parser.add_argument("--drop", nargs="*", default=[], help="Additional columns to exclude")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.target, args.drop, args.test_size, args.seed)
