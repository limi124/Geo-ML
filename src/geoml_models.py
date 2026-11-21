"""Reusable model factories and evaluation helpers for the Geo-ML tutorial."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import geoml_data as gd

FEATURES = list(gd.FEATURE_FILES)
FEATURES_CN = [gd.FEATURE_LABELS[k] for k in FEATURES]
SEED = 42
N_FOLDS = 5


def metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


def mean_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DummyRegressor(strategy="mean")),
    ])


def linear_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])


def ridge_model(alpha: float = 1.0) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def tree_model(max_depth: int | None = 4) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=5,
                                        random_state=SEED)),
    ])


def forest_model(n_estimators: int = 400, min_samples_leaf: int = 3) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=0.8,
            n_jobs=4,
            random_state=SEED,
        )),
    ])


def xgb_model(
    n_estimators: int = 300,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=4,
        random_state=SEED,
    )


def get_xy(df: pd.DataFrame, features: list[str] | None = None):
    features = FEATURES if features is None else features
    return df[features].copy(), df["lst"].to_numpy(dtype=float)


def cv_scores(
    factory: Callable[[], object],
    train_df: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate a model on the fixed spatial folds using only training-region rows."""
    features = FEATURES if features is None else features
    rows = []
    for fold in range(N_FOLDS):
        fit = train_df[train_df["fold"] != fold]
        val = train_df[train_df["fold"] == fold]
        model = factory()
        model.fit(fit[features], fit["lst"])
        pred = model.predict(val[features])
        rows.append({"fold": fold, "n": len(val), **metrics(val["lst"], pred)})
    return pd.DataFrame(rows)


def fit_and_test(
    factory: Callable[[], object],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str] | None = None,
):
    """Fit once on the whole training region and evaluate the independent test block."""
    features = FEATURES if features is None else features
    model = factory()
    model.fit(train_df[features], train_df["lst"])
    pred = model.predict(test_df[features])
    return model, pred, metrics(test_df["lst"], pred)


def summarize_cv(cv: pd.DataFrame, name: str) -> dict:
    return {
        "模型": name,
        "MAE_折均": cv["MAE"].mean(),
        "MAE_折标准差": cv["MAE"].std(ddof=1),
        "RMSE_折均": cv["RMSE"].mean(),
        "RMSE_折标准差": cv["RMSE"].std(ddof=1),
        "R2_折均": cv["R2"].mean(),
        "R2_折标准差": cv["R2"].std(ddof=1),
    }
