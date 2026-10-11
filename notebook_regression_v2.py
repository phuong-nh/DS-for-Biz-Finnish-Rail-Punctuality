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
    from sklearn.linear_model import HuberRegressor, Lasso, LinearRegression, Ridge
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

    from dsforbiz_finnish_rail_punctuality.extract_v2 import extract_months

    return (
        HuberRegressor,
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
    os.makedirs("outputs/it4", exist_ok=True)
    _combined_path = "data/extracted_v2_combined.csv"
    if os.path.exists(_combined_path):
        _df_raw = pd.read_csv(_combined_path)
    else:
        _all_months = [f"{y}_{m:02d}" for y in range(2018, 2025) for m in range(1, 13)]
        _df_raw = extract_months(_all_months, output_path=_combined_path, max_workers=6)

    _ran = _df_raw[~_df_raw["cancelled"]]
    _reached = _ran[_ran["reached_destination"].isin([True]) & _ran["delay_minutes"].notna()]
    _arrived = _reached[_reached["timetable_type"] == "REGULAR"].copy()
    print("scheduled:", len(_df_raw), "| cancelled:", int(_df_raw["cancelled"].sum()))
    print("not reached destination:", len(_ran) - len(_reached), "| adhoc dropped:", len(_reached) - len(_arrived))
    print("regular arrived used:", len(_arrived))
    _arrived["year"] = _arrived["departure_date"].str[:4].astype(int)
    _arrived["month"] = _arrived["departure_date"].str[5:7].astype(int)
    _arrived["is_winter"] = _arrived["month"].isin([12, 1, 2, 3]).astype(int)
    _arrived["is_weekend"] = _arrived["day_of_week"].isin([5, 6]).astype(int)

    _top_corrs = _arrived["corridor"].value_counts().head(10).index
    _arrived["corridor_clean"] = _arrived["corridor"].where(_arrived["corridor"].isin(_top_corrs), "Other")

    _arrived["dest_temp_imp"] = _arrived["dest_temp"].fillna(_arrived["dest_temp"].median())
    _arrived["dest_snow_imp"] = _arrived["dest_snow_depth"].fillna(0.0)
    _arrived["dest_wind_imp"] = _arrived["dest_wind"].fillna(_arrived["dest_wind"].median())
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
    feat_a = ["departure_hour_local", "day_of_week", "month", "is_winter", "is_weekend"] + _route_cols
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
def _(pd, test_df, train_df):
    _pcts = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
    _desc = pd.DataFrame(
        {
            "train (2018-2023)": train_df["delay_minutes"].describe(percentiles=_pcts),
            "test (2024)": test_df["delay_minutes"].describe(percentiles=_pcts),
        }
    ).round(2)
    _desc.to_csv("outputs/it4/delay_describe.csv")
    _desc
    return


@app.cell
def _(pd, train_df):
    _rows = []
    for _cut in [60, 120, 180, 240, 360]:
        _rows.append({"cutoff": f"> {_cut} min", "n_train": int((train_df["delay_minutes"] > _cut).sum())})
    for _cut in [-10, -15, -20, -30, -60]:
        _rows.append({"cutoff": f"< {_cut} min", "n_train": int((train_df["delay_minutes"] < _cut).sum())})
    _counts = pd.DataFrame(_rows)
    _counts["share_%"] = (100 * _counts["n_train"] / len(train_df)).round(3)
    _counts.to_csv("outputs/it4/delay_cutoff_counts.csv", index=False)
    _counts
    return


@app.cell
def _(pd, train_df):
    _cols = ["departure_date", "train_number", "corridor", "departure_hour_local", "delay_minutes", "dest_temp"]
    _extremes = pd.concat(
        [
            train_df.nlargest(10, "delay_minutes")[_cols],
            train_df.nsmallest(10, "delay_minutes")[_cols],
        ]
    )
    _extremes.to_csv("outputs/it4/delay_extreme_rows.csv", index=False)
    _extremes
    return


@app.cell
def _(pd, train_df):
    # about the 0.1% / 99.9% percentiles of the training years, see delay_describe.csv
    clip_low, clip_high = -15, 120
    y_train_clipped = train_df["delay_minutes"].clip(lower=clip_low, upper=clip_high)

    _low = train_df[train_df["delay_minutes"] < clip_low]
    _high = train_df[train_df["delay_minutes"] > clip_high]
    _report = pd.DataFrame(
        {
            "side": [f"< {clip_low} min", f"> {clip_high} min"],
            "n_rows_clipped": [len(_low), len(_high)],
            "share_of_train_%": [round(100 * len(_low) / len(train_df), 3), round(100 * len(_high) / len(train_df), 3)],
            "mean_delay_before": [round(_low["delay_minutes"].mean(), 1), round(_high["delay_minutes"].mean(), 1)],
            "n_distinct_trains": [_low["train_number"].nunique(), _high["train_number"].nunique()],
            "n_distinct_days": [_low["departure_date"].nunique(), _high["departure_date"].nunique()],
        }
    )
    _report.to_csv("outputs/it4/clipping_report.csv", index=False)

    _details = pd.concat(
        [
            _low["train_number"].value_counts().head(5).rename_axis("value").reset_index().assign(what="early side: top train numbers"),
            _high["departure_date"].value_counts().head(5).rename_axis("value").reset_index().assign(what="late side: top days"),
            _high["train_number"].value_counts().head(5).rename_axis("value").reset_index().assign(what="late side: top train numbers"),
        ]
    )
    _details.to_csv("outputs/it4/clipping_details.csv", index=False)
    print("train target mean raw:", round(train_df["delay_minutes"].mean(), 3), "clipped:", round(y_train_clipped.mean(), 3))
    print("train target std raw:", round(train_df["delay_minutes"].std(), 3), "clipped:", round(y_train_clipped.std(), 3))
    print(_details)
    _report
    return (y_train_clipped,)


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
    alpha_df.to_csv("outputs/it4/alpha_validation.csv", index=False)

    best_ridge_alpha = float(alpha_df[alpha_df["Model"] == "Ridge"].sort_values("Val RMSE").iloc[0]["alpha"])
    best_lasso_alpha = float(alpha_df[alpha_df["Model"] == "Lasso"].sort_values("Val RMSE").iloc[0]["alpha"])
    lasso_path = pd.DataFrame(_lasso_coefs, index=feat_b)
    alpha_df
    return best_lasso_alpha, best_ridge_alpha, lasso_path


@app.cell
def _(
    HuberRegressor,
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
    y_train_clipped,
):
    def _score(model_name, feature_set, target, pred):
        # Official 5-min punctuality threshold applied to predicted minutes
        _late_pred = (pred >= 5.0).astype(int)
        return {
            "Model": model_name,
            "Feature Set": feature_set,
            "Train Target": target,
            "MAE": round(float(mean_absolute_error(y_test, pred)), 4),
            "RMSE": round(float(root_mean_squared_error(y_test, pred)), 4),
            "R2": round(float(r2_score(y_test, pred)), 4),
            "ROC-AUC": round(float(roc_auc_score(y_test_late, pred)), 4),
            "Accuracy": round(float(accuracy_score(y_test_late, _late_pred)), 4),
            "Precision": round(float(precision_score(y_test_late, _late_pred, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_test_late, _late_pred, zero_division=0)), 4),
        }

    reg_results = [
        _score("Baseline (Train Mean)", "None", "raw", np.full(len(y_test), y_train.mean())),
        _score("Baseline (Train Mean)", "None", "clipped", np.full(len(y_test), y_train_clipped.mean())),
    ]
    test_preds = {}

    _models = [
        ("Linear Regression (OLS)", make_pipeline(StandardScaler(), LinearRegression())),
        (f"Ridge (alpha={best_ridge_alpha:g})", make_pipeline(StandardScaler(), Ridge(alpha=best_ridge_alpha))),
        (f"Lasso (alpha={best_lasso_alpha:g})", make_pipeline(StandardScaler(), Lasso(alpha=best_lasso_alpha))),
        ("Huber Regression (robust)", make_pipeline(StandardScaler(), HuberRegressor(max_iter=200))),
        (
            "Random Forest Regressor (depth 8)",
            RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
        ),
    ]

    for _model_name, _model in _models:
        for _f_label, _f_cols in [("Set A (Calendar + Route)", feat_a), ("Set B (+ Weather)", feat_b)]:
            for _target, _y in [("raw", y_train), ("clipped", y_train_clipped)]:
                _reg = _model
                _reg.fit(train_df[_f_cols], _y)
                _pred = _reg.predict(test_df[_f_cols])
                test_preds[(_model_name, _f_label, _target)] = _pred
                reg_results.append(_score(_model_name, _f_label, _target, _pred))
    return reg_results, test_preds


@app.cell
def _(pd, reg_results):
    _results_df = pd.DataFrame(reg_results)
    _results_df.to_csv("outputs/it4/model_comparison_results.csv", index=False)

    _wide = _results_df[_results_df["Feature Set"] != "None"].pivot_table(
        index=["Model", "Feature Set"], columns="Train Target", values=["MAE", "RMSE", "R2", "ROC-AUC"]
    )
    _effect = pd.DataFrame(
        {m + " (clipped - raw)": (_wide[(m, "clipped")] - _wide[(m, "raw")]).round(4) for m in ["MAE", "RMSE", "R2", "ROC-AUC"]}
    )
    _effect.to_csv("outputs/it4/clipping_effect.csv")
    _results_df


@app.cell
def _(np, plt, test_preds, y_test):
    _rng = np.random.default_rng(42)
    _idx = _rng.choice(len(y_test), size=min(20000, len(y_test)), replace=False)
    _y = y_test.to_numpy()[_idx]

    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    for _row, (_model_name, _short) in enumerate([("Linear Regression (OLS)", "OLS"), ("Random Forest Regressor (depth 8)", "RF")]):
        for _col, _target in enumerate(["raw", "clipped"]):
            _ax = _axes[_row, _col]
            _fitted = test_preds[(_model_name, "Set B (+ Weather)", _target)][_idx]
            _ax.scatter(_fitted, _y - _fitted, s=4, alpha=0.15, color="teal")
            _ax.axhline(0, color="red", linestyle="--", linewidth=1)
            _ax.set_title(f"Residuals vs Fitted: {_short} (Set B, {_target} train target)")
            _ax.set_xlabel("Fitted delay (min)")
            _ax.grid(True, alpha=0.3)
        _axes[_row, 0].set_ylabel("Residual e = y - y_hat (min)")
    _fig.savefig("outputs/it4/regression_residuals.png", bbox_inches="tight")
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
    _fig.savefig("outputs/it4/lasso_coefficient_path.png", bbox_inches="tight")
    _fig
    return


@app.cell
def _(np, plt, test_preds, y_test):
    _rng = np.random.default_rng(42)
    _idx = _rng.choice(len(y_test), size=min(20000, len(y_test)), replace=False)
    _y = y_test.to_numpy()[_idx]

    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    for _row, (_model_name, _short) in enumerate([("Linear Regression (OLS)", "OLS"), ("Random Forest Regressor (depth 8)", "RF")]):
        for _col, _target in enumerate(["raw", "clipped"]):
            _ax = _axes[_row, _col]
            _fitted = test_preds[(_model_name, "Set B (+ Weather)", _target)][_idx]
            _ax.scatter(_fitted, _y, s=4, alpha=0.15, color="teal")
            _lims = [min(_fitted.min(), 0), max(_fitted.max(), 30)]
            _ax.plot(_lims, _lims, color="red", linestyle="--", linewidth=1, label="y = y_hat")
            _ax.set_title(f"Actual vs Fitted: {_short} (Set B, {_target} train target)")
            _ax.set_xlabel("Fitted delay (min)")
            _ax.legend(loc="upper left")
            _ax.grid(True, alpha=0.3)
        _axes[_row, 0].set_ylabel("Actual delay (min)")
    _fig.savefig("outputs/it4/regression_actual_vs_fitted.png", bbox_inches="tight")
    _fig
    return


if __name__ == "__main__":
    app.run()
