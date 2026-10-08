import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        mean_absolute_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
        root_mean_squared_error,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from dsforbiz_finnish_rail_punctuality.extract import extract_months

    return (
        Lasso,
        LinearRegression,
        RandomForestRegressor,
        Ridge,
        StandardScaler,
        accuracy_score,
        extract_months,
        make_pipeline,
        mean_absolute_error,
        np,
        os,
        pd,
        plt,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
        root_mean_squared_error,
    )


@app.cell
def _(extract_months, os, pd):
    os.makedirs("outputs/it3", exist_ok=True)
    _combined_path = "data/extracted_combined.csv"
    if os.path.exists(_combined_path):
        _df_raw = pd.read_csv(_combined_path)
    else:
        _all_months = [f"{y}_{m:02d}" for y in range(2018, 2025) for m in range(1, 13)]
        _df_raw = extract_months(_all_months, output_path=_combined_path, max_workers=6)

    _arrived = _df_raw[_df_raw["is_late"].notnull()].copy()
    _arrived["year"] = _arrived["departure_date"].str[:4].astype(int)
    _arrived["month"] = _arrived["departure_date"].str[5:7].astype(int)
    _arrived["is_winter"] = _arrived["month"].isin([12, 1, 2, 3]).astype(int)
    _arrived["is_weekend"] = _arrived["day_of_week"].isin([5, 6]).astype(int)

    _top_corrs = _arrived["corridor"].value_counts().head(10).index
    _arrived["corridor_clean"] = _arrived["corridor"].where(_arrived["corridor"].isin(_top_corrs), "Other")

    _arrived["dest_temp_imp"] = _arrived["dest_temp"].fillna(_arrived["dest_temp"].median())
    _arrived["dest_snow_imp"] = _arrived["dest_snow_depth"].fillna(0.0)
    _arrived["dest_wind_imp"] = _arrived["dest_wind_speed"].fillna(_arrived["dest_wind_speed"].median())
    _arrived["dest_precip_imp"] = _arrived["dest_precip_intensity"].fillna(0.0)

    _arrived["is_extreme_cold"] = (_arrived["dest_temp_imp"] < -15).astype(int)
    _arrived["is_freezing"] = (
        (_arrived["dest_temp_imp"] >= -2) & (_arrived["dest_temp_imp"] <= 2)
    ).astype(int)
    _arrived["is_heavy_snow"] = (_arrived["dest_snow_imp"] > 20).astype(int)
    _arrived["temp_x_wind"] = _arrived["dest_temp_imp"] * _arrived["dest_wind_imp"]
    _arrived["snow_x_wind"] = _arrived["dest_snow_imp"] * _arrived["dest_wind_imp"]

    _encoded = pd.get_dummies(_arrived, columns=["corridor_clean", "train_type"], drop_first=True)

    _route_cols = [c for c in _encoded.columns if c.startswith(("corridor_clean_", "train_type_"))]
    feat_a = ["departure_hour", "day_of_week", "month", "is_winter", "is_weekend"] + _route_cols
    weather_cols = [
        "dest_temp_imp",
        "dest_snow_imp",
        "dest_wind_imp",
        "dest_precip_imp",
        "is_extreme_cold",
        "is_freezing",
        "is_heavy_snow",
        "temp_x_wind",
        "snow_x_wind",
    ]
    feat_b = feat_a + weather_cols

    train_df = _encoded[_encoded["year"] < 2024]
    test_df = _encoded[_encoded["year"] == 2024]

    y_train = train_df["delay_minutes"]
    y_test = test_df["delay_minutes"]
    y_test_late = test_df["is_late"].astype(int)
    return (
        feat_a,
        feat_b,
        test_df,
        train_df,
        weather_cols,
        y_test,
        y_test_late,
        y_train,
    )


@app.cell
def _(
    Lasso,
    Ridge,
    StandardScaler,
    feat_b,
    make_pipeline,
    mean_absolute_error,
    np,
    pd,
    root_mean_squared_error,
    train_df,
):
    _fit = train_df[train_df["year"] < 2023]
    _val = train_df[train_df["year"] == 2023]

    _rows = []
    for _alpha in np.logspace(-1, 6, 8):
        _ridge = make_pipeline(StandardScaler(), Ridge(alpha=_alpha))
        _ridge.fit(_fit[feat_b], _fit["delay_minutes"])
        _pred = _ridge.predict(_val[feat_b])
        _rows.append(
            {
                "Model": "Ridge",
                "alpha": _alpha,
                "Val MAE": round(float(mean_absolute_error(_val["delay_minutes"], _pred)), 4),
                "Val RMSE": round(float(root_mean_squared_error(_val["delay_minutes"], _pred)), 4),
            }
        )

    _lasso_coefs = {}
    for _alpha in np.logspace(-3, 1, 9):
        _lasso = make_pipeline(StandardScaler(), Lasso(alpha=_alpha))
        _lasso.fit(_fit[feat_b], _fit["delay_minutes"])
        _pred = _lasso.predict(_val[feat_b])
        _lasso_coefs[_alpha] = _lasso[-1].coef_
        _rows.append(
            {
                "Model": "Lasso",
                "alpha": _alpha,
                "Val MAE": round(float(mean_absolute_error(_val["delay_minutes"], _pred)), 4),
                "Val RMSE": round(float(root_mean_squared_error(_val["delay_minutes"], _pred)), 4),
            }
        )

    alpha_df = pd.DataFrame(_rows)
    alpha_df.to_csv("outputs/it3/alpha_validation.csv", index=False)

    best_ridge_alpha = float(alpha_df[alpha_df["Model"] == "Ridge"].sort_values("Val RMSE").iloc[0]["alpha"])
    best_lasso_alpha = float(alpha_df[alpha_df["Model"] == "Lasso"].sort_values("Val RMSE").iloc[0]["alpha"])
    lasso_path = pd.DataFrame(_lasso_coefs, index=feat_b)
    alpha_df
    return best_lasso_alpha, best_ridge_alpha, lasso_path


@app.cell
def _(
    Lasso,
    LinearRegression,
    RandomForestRegressor,
    Ridge,
    StandardScaler,
    accuracy_score,
    best_lasso_alpha,
    best_ridge_alpha,
    feat_a,
    feat_b,
    make_pipeline,
    mean_absolute_error,
    np,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
    test_df,
    train_df,
    y_test,
    y_test_late,
    y_train,
):
    def _score(model_name, feature_set, pred):
        # Official 5-min punctuality threshold applied to predicted minutes
        _late_pred = (pred >= 5.0).astype(int)
        return {
            "Model": model_name,
            "Feature Set": feature_set,
            "MAE": round(float(mean_absolute_error(y_test, pred)), 4),
            "RMSE": round(float(root_mean_squared_error(y_test, pred)), 4),
            "R2": round(float(r2_score(y_test, pred)), 4),
            "ROC-AUC": round(float(roc_auc_score(y_test_late, pred)), 4),
            "Accuracy": round(float(accuracy_score(y_test_late, _late_pred)), 4),
            "Precision": round(float(precision_score(y_test_late, _late_pred, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_test_late, _late_pred, zero_division=0)), 4),
        }

    reg_results = [
        _score("Baseline (Train Mean)", "None", np.full(len(y_test), y_train.mean())),
    ]
    test_preds = {}

    _models = [
        ("Linear Regression (OLS)", make_pipeline(StandardScaler(), LinearRegression())),
        (f"Ridge (alpha={best_ridge_alpha:g})", make_pipeline(StandardScaler(), Ridge(alpha=best_ridge_alpha))),
        (f"Lasso (alpha={best_lasso_alpha:g})", make_pipeline(StandardScaler(), Lasso(alpha=best_lasso_alpha))),
        (
            "Random Forest Regressor (depth 8)",
            RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
        ),
    ]

    for _model_name, _model in _models:
        for _f_label, _f_cols in [("Set A (Calendar + Route)", feat_a), ("Set B (+ Weather)", feat_b)]:
            _reg = _model
            _reg.fit(train_df[_f_cols], y_train)
            _pred = _reg.predict(test_df[_f_cols])
            test_preds[(_model_name, _f_label)] = _pred
            reg_results.append(_score(_model_name, _f_label, _pred))
    return reg_results, test_preds


@app.cell
def _(pd, reg_results):
    _results_df = pd.DataFrame(reg_results)
    _results_df.to_csv("outputs/it3/model_comparison_results.csv", index=False)
    _results_df
    return


@app.cell
def _(np, plt, test_preds, y_test):
    _rng = np.random.default_rng(42)
    _idx = _rng.choice(len(y_test), size=min(20000, len(y_test)), replace=False)
    _y = y_test.to_numpy()[_idx]

    _panels = [
        ("Linear Regression (OLS)", "OLS (Set B)"),
        ("Random Forest Regressor (depth 8)", "Random Forest (Set B)"),
    ]

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for _ax, (_model_name, _title) in zip(_axes, _panels):
        _fitted = test_preds[(_model_name, "Set B (+ Weather)")][_idx]
        _ax.scatter(_fitted, _y - _fitted, s=4, alpha=0.15, color="teal")
        _ax.axhline(0, color="red", linestyle="--", linewidth=1)
        _ax.set_title(f"Residuals vs Fitted: {_title}")
        _ax.set_xlabel("Fitted delay (min)")
        _ax.grid(True, alpha=0.3)
    _axes[0].set_ylabel("Residual e = y - y_hat (min)")
    _fig.savefig("outputs/it3/regression_residuals.png", bbox_inches="tight")
    _fig
    return


@app.cell
def _(lasso_path, np, plt, weather_cols):
    _fig, _ax = plt.subplots(figsize=(9, 5))
    _alphas = np.array(lasso_path.columns, dtype=float)
    for _feat, _coefs in lasso_path.iterrows():
        if _feat in weather_cols:
            _ax.plot(_alphas, _coefs.to_numpy(), linewidth=2, label=_feat)
        else:
            _ax.plot(_alphas, _coefs.to_numpy(), color="lightgray", linewidth=1)
    _ax.set_xscale("log")
    _ax.axhline(0, color="black", linewidth=0.8)
    _ax.set_xlabel("Lasso alpha (log scale)")
    _ax.set_ylabel("Standardized coefficient (min)")
    _ax.set_title("Lasso Coefficient Path (Set B, weather features coloured)")
    _ax.legend(loc="upper right", fontsize=8)
    _ax.grid(True, alpha=0.3)
    _fig.savefig("outputs/it3/lasso_coefficient_path.png", bbox_inches="tight")
    _fig
    return


if __name__ == "__main__":
    app.run()
