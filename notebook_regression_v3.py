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

    from dsforbiz_finnish_rail_punctuality.extract_v2 import extract_months

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
    os.makedirs("outputs/it4b", exist_ok=True)
    _combined_path = "data/extracted_v2_combined.csv"
    if os.path.exists(_combined_path):
        _df_raw = pd.read_csv(_combined_path)
    else:
        _all_months = [f"{y}_{m:02d}" for y in range(2018, 2025) for m in range(1, 13)]
        _df_raw = extract_months(_all_months, output_path=_combined_path, max_workers=6)

    _ran = _df_raw[~_df_raw["cancelled"]]
    _reached = _ran[_ran["reached_destination"].isin([True]) & _ran["delay_minutes"].notna()]
    _arrived = _reached[_reached["timetable_type"] == "REGULAR"].copy()
    print("regular arrived used:", len(_arrived))

    _arrived["year"] = _arrived["departure_date"].str[:4].astype(int)
    _arrived["month"] = _arrived["departure_date"].str[5:7].astype(int)
    _arrived["is_winter"] = _arrived["month"].isin([12, 1, 2, 3]).astype(int)
    _arrived["is_weekend"] = _arrived["day_of_week"].isin([5, 6]).astype(int)
    # hours 0-4 have very few trains, merge them
    _arrived["hour_clean"] = _arrived["departure_hour_local"].clip(lower=4)
    _arrived["month_cat"] = _arrived["month"]
    _arrived["dow_cat"] = _arrived["day_of_week"].astype(int)
    _arrived["hour_cat"] = _arrived["hour_clean"].astype(int)

    # only use training years for anything learned from the data (corridors, medians)
    _is_train = _arrived["year"] < 2024
    _top_corrs = _arrived.loc[_is_train, "corridor"].value_counts().head(10).index
    _arrived["corridor_clean"] = _arrived["corridor"].where(_arrived["corridor"].isin(_top_corrs), "Other")

    for _p in ["origin", "dest"]:
        _core = [f"{_p}_temp", f"{_p}_snow_depth", f"{_p}_wind", f"{_p}_precip_intensity"]
        _arrived[f"{_p}_wx_missing"] = _arrived[_core].isna().any(axis=1).astype(int)
        for _c in _core:
            _arrived[_c + "_imp"] = _arrived[_c].fillna(_arrived.loc[_is_train, _c].median())
        _arrived[f"{_p}_is_extreme_cold"] = (_arrived[f"{_p}_temp_imp"] < -15).astype(int)
        _arrived[f"{_p}_is_freezing"] = _arrived[f"{_p}_temp_imp"].between(-2, 2).astype(int)
        _arrived[f"{_p}_is_heavy_snow"] = (_arrived[f"{_p}_snow_depth_imp"] > 20).astype(int)
        _arrived[f"{_p}_temp_x_wind"] = _arrived[f"{_p}_temp_imp"] * _arrived[f"{_p}_wind_imp"]
        _arrived[f"{_p}_snow_x_wind"] = _arrived[f"{_p}_snow_depth_imp"] * _arrived[f"{_p}_wind_imp"]

    _route_raw = ["route_min_temp", "route_max_snow_depth", "route_max_wind", "route_max_gust", "route_max_precip_intensity", "route_min_visibility"]
    for _c in _route_raw:
        _arrived[_c + "_imp"] = _arrived[_c].fillna(_arrived.loc[_is_train, _c].median())
    _arrived["route_is_extreme_cold"] = (_arrived["route_min_temp_imp"] < -15).astype(int)
    _arrived["route_is_heavy_snow"] = (_arrived["route_max_snow_depth_imp"] > 20).astype(int)

    _encoded = pd.get_dummies(
        _arrived,
        columns=["month_cat", "dow_cat", "hour_cat", "corridor_clean", "train_type"],
        drop_first=True,
    )

    _route_cols = [c for c in _encoded.columns if c.startswith(("corridor_clean_", "train_type_"))]
    _calendar_onehot = [c for c in _encoded.columns if c.startswith(("month_cat_", "dow_cat_", "hour_cat_"))]
    _journey = ["scheduled_duration_min", "n_commercial_stops"]

    # one-hot calendar only for linear models, is_winter / is_weekend would be collinear with the dummies
    feat_a_tree = ["departure_hour_local", "day_of_week", "month", "is_winter", "is_weekend"] + _journey + _route_cols
    feat_a_lin = _calendar_onehot + _journey + _route_cols
    _wx_names = ["temp_imp", "snow_depth_imp", "wind_imp", "precip_intensity_imp", "is_extreme_cold", "is_freezing", "is_heavy_snow", "temp_x_wind", "snow_x_wind", "wx_missing"]
    _origin_wx = [f"origin_{n}" for n in _wx_names]
    _dest_wx = [f"dest_{n}" for n in _wx_names]
    _route_wx = [c + "_imp" for c in _route_raw] + ["route_is_extreme_cold", "route_is_heavy_snow"]
    weather_groups = {"origin": _origin_wx, "dest": _dest_wx, "route": _route_wx}

    _set_names = {
        "A": "Set A (Calendar + Route + Journey)",
        "B": "Set B (+ Destination Weather)",
        "C": "Set C (+ Origin Weather)",
        "D": "Set D (+ All Weather)",
    }
    _extra = {"A": [], "B": _dest_wx, "C": _origin_wx, "D": _origin_wx + _dest_wx + _route_wx}
    feature_sets = {
        "lin": [(_set_names[k], feat_a_lin + _extra[k]) for k in _extra],
        "tree": [(_set_names[k], feat_a_tree + _extra[k]) for k in _extra],
    }
    feat_d_lin = feat_a_lin + _extra["D"]
    print("features: A tree", len(feat_a_tree), "| A linear", len(feat_a_lin), "| D linear", len(feat_d_lin))

    train_df = _encoded[_encoded["year"] < 2024]
    test_df = _encoded[_encoded["year"] == 2024]

    # clip bounds from notebook_regression_v2 (outputs/it4/clipping_report.csv)
    y_train = train_df["delay_minutes"].clip(lower=-15, upper=120)
    y_test = test_df["delay_minutes"]
    y_test_late = test_df["is_late"].astype(int)
    return (
        feat_d_lin,
        feature_sets,
        test_df,
        train_df,
        weather_groups,
        y_test,
        y_test_late,
        y_train,
    )


@app.cell
def _(
    Lasso,
    Ridge,
    StandardScaler,
    feat_d_lin,
    make_pipeline,
    mean_absolute_error,
    np,
    pd,
    root_mean_squared_error,
    train_df,
    y_train,
):
    _is_fit = train_df["year"] < 2023
    _fit, _val = train_df[_is_fit], train_df[~_is_fit]
    _y_fit, _y_val = y_train[_is_fit], train_df.loc[~_is_fit, "delay_minutes"]

    _rows = []
    for _alpha in np.logspace(-1, 6, 8):
        _ridge = make_pipeline(StandardScaler(), Ridge(alpha=_alpha))
        _ridge.fit(_fit[feat_d_lin], _y_fit)
        _pred = _ridge.predict(_val[feat_d_lin])
        _rows.append(
            {
                "Model": "Ridge",
                "alpha": _alpha,
                "Val MAE": round(float(mean_absolute_error(_y_val, _pred)), 4),
                "Val RMSE": round(float(root_mean_squared_error(_y_val, _pred)), 4),
            }
        )

    _lasso_coefs = {}
    for _alpha in np.logspace(-3, 1, 9):
        _lasso = make_pipeline(StandardScaler(), Lasso(alpha=_alpha))
        _lasso.fit(_fit[feat_d_lin], _y_fit)
        _pred = _lasso.predict(_val[feat_d_lin])
        _lasso_coefs[_alpha] = _lasso[-1].coef_
        _rows.append(
            {
                "Model": "Lasso",
                "alpha": _alpha,
                "Val MAE": round(float(mean_absolute_error(_y_val, _pred)), 4),
                "Val RMSE": round(float(root_mean_squared_error(_y_val, _pred)), 4),
            }
        )

    alpha_df = pd.DataFrame(_rows)
    alpha_df.to_csv("outputs/it4b/alpha_validation.csv", index=False)

    best_ridge_alpha = float(alpha_df[alpha_df["Model"] == "Ridge"].sort_values("Val RMSE").iloc[0]["alpha"])
    best_lasso_alpha = float(alpha_df[alpha_df["Model"] == "Lasso"].sort_values("Val RMSE").iloc[0]["alpha"])
    lasso_path = pd.DataFrame(_lasso_coefs, index=feat_d_lin)
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
    feature_sets,
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
        ("Linear Regression (OLS)", make_pipeline(StandardScaler(), LinearRegression()), "lin"),
        (f"Ridge (alpha={best_ridge_alpha:g})", make_pipeline(StandardScaler(), Ridge(alpha=best_ridge_alpha)), "lin"),
        (f"Lasso (alpha={best_lasso_alpha:g})", make_pipeline(StandardScaler(), Lasso(alpha=best_lasso_alpha)), "lin"),
        (
            "Random Forest Regressor (depth 8)",
            RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
            "tree",
        ),
    ]

    for _model_name, _model, _kind in _models:
        for _f_label, _f_cols in feature_sets[_kind]:
            _reg = _model
            _reg.fit(train_df[_f_cols], y_train)
            _pred = _reg.predict(test_df[_f_cols])
            test_preds[(_model_name, _f_label)] = _pred
            reg_results.append(_score(_model_name, _f_label, _pred))
    return reg_results, test_preds


@app.cell
def _(pd, reg_results):
    _results_df = pd.DataFrame(reg_results)
    _results_df.to_csv("outputs/it4b/model_comparison_results.csv", index=False)

    _rows = []
    for _model_name, _g in _results_df[_results_df["Feature Set"] != "None"].groupby("Model", sort=False):
        _base = _g[_g["Feature Set"].str.startswith("Set A")].iloc[0]
        for _, _r in _g[~_g["Feature Set"].str.startswith("Set A")].iterrows():
            _rows.append(
                {
                    "Model": _model_name,
                    "Compared to Set A": _r["Feature Set"],
                    "dAUC": round(_r["ROC-AUC"] - _base["ROC-AUC"], 4),
                    "dMAE": round(_r["MAE"] - _base["MAE"], 4),
                    "dRMSE": round(_r["RMSE"] - _base["RMSE"], 4),
                    "dR2": round(_r["R2"] - _base["R2"], 4),
                }
            )
    pd.DataFrame(_rows).to_csv("outputs/it4b/weather_gain_summary.csv", index=False)
    _results_df
    return


@app.cell
def _(np, plt, test_preds, y_test):
    _rng = np.random.default_rng(42)
    _idx = _rng.choice(len(y_test), size=min(20000, len(y_test)), replace=False)
    _y = y_test.to_numpy()[_idx]

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for _ax, (_model_name, _title) in zip(
        _axes, [("Linear Regression (OLS)", "OLS (Set D)"), ("Random Forest Regressor (depth 8)", "Random Forest (Set D)")]
    ):
        _fitted = test_preds[(_model_name, "Set D (+ All Weather)")][_idx]
        _ax.scatter(_fitted, _y - _fitted, s=4, alpha=0.15, color="teal")
        _ax.axhline(0, color="red", linestyle="--", linewidth=1)
        _ax.set_title(f"Residuals vs Fitted: {_title}")
        _ax.set_xlabel("Fitted delay (min)")
        _ax.grid(True, alpha=0.3)
    _axes[0].set_ylabel("Residual e = y - y_hat (min)")
    _fig.savefig("outputs/it4b/regression_residuals.png", bbox_inches="tight")
    _fig
    return


@app.cell
def _(lasso_path, np, pd, plt, weather_groups):
    _colors = {"origin": "royalblue", "dest": "seagreen", "route": "firebrick"}
    _fig, _ax = plt.subplots(figsize=(9, 5))
    _alphas = np.array(lasso_path.columns, dtype=float)
    for _feat, _coefs in lasso_path.iterrows():
        _group = [g for g, cols in weather_groups.items() if _feat in cols]
        if _group:
            _ax.plot(_alphas, _coefs.to_numpy(), linewidth=1.5, color=_colors[_group[0]])
        elif _feat in ["scheduled_duration_min", "n_commercial_stops"]:
            _ax.plot(_alphas, _coefs.to_numpy(), linewidth=2, color="black")
        else:
            _ax.plot(_alphas, _coefs.to_numpy(), color="lightgray", linewidth=1)
    for _g, _c in list(_colors.items()) + [("journey", "black"), ("calendar / route", "lightgray")]:
        _ax.plot([], [], color=_c, label=_g)
    _ax.set_xscale("log")
    _ax.axhline(0, color="black", linewidth=0.8)
    _ax.set_xlabel("Lasso alpha (log scale)")
    _ax.set_ylabel("Standardized coefficient (min)")
    _ax.set_title("Lasso Coefficient Path (Set D linear)")
    _ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    _ax.grid(True, alpha=0.3)
    _fig.savefig("outputs/it4b/lasso_coefficient_path.png", bbox_inches="tight")

    _survive = {}
    for _feat, _coefs in lasso_path.iterrows():
        _nonzero = [a for a, c in zip(_alphas, _coefs) if c != 0]
        _survive[_feat] = max(_nonzero) if _nonzero else 0.0
    _survive = pd.Series(_survive, name="largest_alpha_nonzero").sort_values(ascending=False)
    _survive.to_csv("outputs/it4b/lasso_survival.csv")
    _fig
    return


@app.cell
def _(plt, train_df, weather_groups):
    _cols = [c for c in weather_groups["origin"] + weather_groups["dest"] + weather_groups["route"] if c.endswith("_imp")]
    _corr = train_df[_cols].corr().round(2)
    _corr.to_csv("outputs/it4b/weather_correlation.csv")

    _fig, _ax = plt.subplots(figsize=(10, 8))
    _im = _ax.imshow(_corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    _ax.set_xticks(range(len(_cols)), _cols, rotation=90, fontsize=8)
    _ax.set_yticks(range(len(_cols)), _cols, fontsize=8)
    _fig.colorbar(_im)
    _ax.set_title("Correlation of Weather Features (training years)")
    _fig.savefig("outputs/it4b/weather_correlation.png", bbox_inches="tight")
    _fig
    return


if __name__ == "__main__":
    app.run()
