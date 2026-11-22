"""Reusable preprocessing for arbitrary tabular regression datasets."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Dataset-specific fields that reveal the target or are identifiers. Extend this
# list when adapting the workflow to another dataset.
FLOOD_LEAKAGE_COLUMNS = {
    "flood_occurrence_current_event", "current_event", "inundation_area_sqm",
    "is_good_to_live", "reason_not_good_to_live", "is_synthetic",
    "issynthetic", "isgoodtolive", "reasonnotgoodtolive",
}
IDENTIFIER_COLUMNS = {
    "record_id", "recordid", "id", "place_name", "placename",
    "generation_date", "date_generated",
}


def prepare_features(data, target_column, drop_columns=None):
    """Clean a frame and return X, y, feature lists, and removed columns.

    Object columns are treated as categorical; numeric and boolean columns as
    numeric. Identifiers and known flood-target leakage fields are excluded.
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' is not in the CSV.")
    frame = data.copy().drop_duplicates().replace([np.inf, -np.inf], np.nan)
    frame = frame.loc[frame[target_column].notna()].copy()
    y = pd.to_numeric(frame[target_column], errors="coerce")
    valid = y.notna()
    frame, y = frame.loc[valid], y.loc[valid]
    excluded = IDENTIFIER_COLUMNS | FLOOD_LEAKAGE_COLUMNS | set(drop_columns or [])
    # Always exclude other known target/leakage fields when the flood target is used.
    if target_column == "flood_risk_score":
        excluded |= {"flood_occurrence", "flood_occurrence_current_event", "inundation_area_sqm"}
    drop = [c for c in frame.columns if c != target_column and c.lower() in excluded]
    X = frame.drop(columns=[target_column, *drop], errors="ignore")
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after excluding target and identifiers.")
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype("int8")
    # Parse numeric-looking strings while leaving genuine categories intact.
    for col in X.select_dtypes(include=["object", "string"]).columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().mean() >= 0.9:
            X[col] = converted
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    if not numeric:
        raise ValueError("At least one numeric predictor is required for this regression workflow.")
    return X, y.astype(float), numeric, categorical, drop


def build_preprocessor(numeric_cols, categorical_cols):
    """Fit-time imputation, scaling, and safe one-hot encoding."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, numeric_cols),
                              ("cat", categorical, categorical_cols)],
                             remainder="drop")
