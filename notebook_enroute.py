import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        mean_absolute_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
        roc_curve,
        root_mean_squared_error,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        LinearRegression,
        LogisticRegression,
        StandardScaler,
        accuracy_score,
        make_pipeline,
        mean_absolute_error,
        os,
        pd,
        plt,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
        roc_curve,
        root_mean_squared_error,
    )


@app.cell
def _(os, pd):
    os.makedirs("outputs/enroute", exist_ok=True)
    _stops = pd.read_csv("data/extracted_stops_combined.csv")
    _df = _stops[_stops["timetable_type"] == "REGULAR"].copy()
    print("regular checkpoint rows:", len(_df))

    _df["year"] = _df["departure_date"].str[:4].astype(int)
    _df["month"] = _df["departure_date"].str[5:7].astype(int)
    _df["is_winter"] = _df["month"].isin([12, 1, 2, 3]).astype(int)
    _df["is_weekend"] = _df["day_of_week"].isin([5, 6]).astype(int)
    _df["hour_cat"] = _df["departure_hour_local"].clip(lower=4).astype(int)
    _df["month_cat"] = _df["month"]
    _df["dow_cat"] = _df["day_of_week"].astype(int)

    # at the origin there is no previous stop, so the change is just 0
    _df["prev_stop_delay"] = _df["prev_stop_delay"].fillna(_df["current_delay"])
    _df["delay_change"] = _df["current_delay"] - _df["prev_stop_delay"]

    _is_train = _df["year"] < 2024
    _top_corrs = _df.loc[_is_train, "corridor"].value_counts().head(10).index
    _df["corridor_clean"] = _df["corridor"].where(_df["corridor"].isin(_top_corrs), "Other")

    _now = ["now_temp", "now_wind", "now_gust", "now_precip_intensity", "now_snow_depth", "now_visibility"]
    _ahead = [
        "ahead_min_temp",
        "ahead_max_snow_depth",
        "ahead_max_wind",
        "ahead_max_gust",
        "ahead_max_precip_intensity",
        "ahead_min_visibility",
    ]
    _df["now_wx_missing"] = _df[_now].isna().any(axis=1).astype(int)
    for _c in _now + _ahead:
        _df[_c] = _df[_c].fillna(_df.loc[_is_train, _c].median())
    _df["now_is_extreme_cold"] = (_df["now_temp"] < -15).astype(int)
    _df["now_is_heavy_snow"] = (_df["now_snow_depth"] > 20).astype(int)
    _df["ahead_is_extreme_cold"] = (_df["ahead_min_temp"] < -15).astype(int)
    _df["ahead_is_heavy_snow"] = (_df["ahead_max_snow_depth"] > 20).astype(int)

    df = pd.get_dummies(_df, columns=["month_cat", "dow_cat", "hour_cat", "corridor_clean", "train_type"], drop_first=True)

    _route = [c for c in df.columns if c.startswith(("corridor_clean_", "train_type_"))]
    _cal_onehot = [c for c in df.columns if c.startswith(("month_cat_", "dow_cat_", "hour_cat_"))]
    _trip = ["scheduled_duration_min", "n_commercial_stops", "remaining_min", "remaining_stops", "progress"]
    weather_cols = _now + _ahead + [
        "now_wx_missing",
        "now_is_extreme_cold",
        "now_is_heavy_snow",
        "ahead_is_extreme_cold",
        "ahead_is_heavy_snow",
    ]

    # E0 = current delay only, E1 = + calendar / route / trip, E2 = + weather now and ahead
    _e0 = ["current_delay"]
    _e1_lin = _e0 + ["delay_change"] + _cal_onehot + _trip + _route
    _e1_tree = _e0 + ["delay_change", "departure_hour_local", "day_of_week", "month", "is_winter", "is_weekend"] + _trip + _route
    sets = {
        "lin": {"E0": _e0, "E1": _e1_lin, "E2": _e1_lin + weather_cols},
        "tree": {"E0": _e0, "E1": _e1_tree, "E2": _e1_tree + weather_cols},
    }
    print("features E1 lin / tree:", len(_e1_lin), len(_e1_tree), "| weather:", len(weather_cols))
    return df, sets


@app.cell
def _(
    HistGradientBoostingClassifier,
    LogisticRegression,
    StandardScaler,
    accuracy_score,
    df,
    make_pipeline,
    pd,
    precision_score,
    recall_score,
    roc_auc_score,
    sets,
):
    # hyperparameters reused from notebook_modeling_v3 (tuned on 2023)
    _models = {
        "Logistic Regression (Balanced)": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
            "lin",
        ),
        "Gradient Boosting (HistGB, Balanced)": (
            HistGradientBoostingClassifier(
                learning_rate=0.1, max_iter=500, max_leaf_nodes=31, class_weight="balanced", early_stopping=False, random_state=42
            ),
            "tree",
        ),
    }

    _rows = []
    clf_probs = {}
    for _cp, _g in df.groupby("checkpoint"):
        _train, _test = _g[_g["year"] < 2024], _g[_g["year"] == 2024]
        for _name, (_model, _kind) in _models.items():
            for _set, _cols in sets[_kind].items():
                _model.fit(_train[_cols], _train["is_late"])
                _probs = _model.predict_proba(_test[_cols])[:, 1]
                _preds = (_probs >= 0.5).astype(int)
                clf_probs[(_cp, _name, _set)] = _probs
                _rows.append(
                    {
                        "checkpoint_%": _cp,
                        "Model": _name,
                        "Set": _set,
                        "ROC-AUC": round(float(roc_auc_score(_test["is_late"], _probs)), 4),
                        "Accuracy": round(float(accuracy_score(_test["is_late"], _preds)), 4),
                        "Precision": round(float(precision_score(_test["is_late"], _preds, zero_division=0)), 4),
                        "Recall": round(float(recall_score(_test["is_late"], _preds, zero_division=0)), 4),
                    }
                )
        print("done checkpoint", _cp)
    clf_results = pd.DataFrame(_rows)
    clf_results.to_csv("outputs/enroute/classification_by_checkpoint.csv", index=False)
    clf_results
    return clf_probs, clf_results


@app.cell
def _(
    HistGradientBoostingRegressor,
    LinearRegression,
    StandardScaler,
    df,
    make_pipeline,
    mean_absolute_error,
    pd,
    r2_score,
    root_mean_squared_error,
    sets,
):
    _models = {
        "Linear Regression (OLS)": (make_pipeline(StandardScaler(), LinearRegression()), "lin"),
        "Gradient Boosting (HistGB)": (
            HistGradientBoostingRegressor(learning_rate=0.05, max_iter=200, max_leaf_nodes=31, early_stopping=False, random_state=42),
            "tree",
        ),
    }

    def _metrics(cp, name, set_name, y, pred):
        return {
            "checkpoint_%": cp,
            "Model": name,
            "Set": set_name,
            "MAE": round(float(mean_absolute_error(y, pred)), 4),
            "RMSE": round(float(root_mean_squared_error(y, pred)), 4),
            "R2": round(float(r2_score(y, pred)), 4),
        }

    _rows = []
    for _cp, _g in df.groupby("checkpoint"):
        _train, _test = _g[_g["year"] < 2024], _g[_g["year"] == 2024]
        _y_test = _test["final_delay"]
        # simplest rule: the train stays exactly as late as it is now
        _rows.append(_metrics(_cp, "Persistence (final = current)", "-", _y_test, _test["current_delay"]))
        # predict the delay still to come and add it to the current delay, so a train already 150 min late is not capped by the clipping (clip bounds from notebook_regression_v2)
        _y_train = _train["delay_to_go"].clip(lower=-15, upper=120)
        for _name, (_model, _kind) in _models.items():
            for _set, _cols in sets[_kind].items():
                _model.fit(_train[_cols], _y_train)
                _pred_final = _test["current_delay"].to_numpy() + _model.predict(_test[_cols])
                _rows.append(_metrics(_cp, _name, _set, _y_test, _pred_final))
        print("done checkpoint", _cp)
    reg_results = pd.DataFrame(_rows)
    reg_results.to_csv("outputs/enroute/regression_by_checkpoint.csv", index=False)
    reg_results
    return (reg_results,)


@app.cell
def _(clf_results, pd, reg_results):
    _rows = []
    for (_cp, _model), _g in clf_results.groupby(["checkpoint_%", "Model"]):
        _r = _g.set_index("Set")
        _rows.append(
            {
                "checkpoint_%": _cp,
                "Model": _model,
                "dAUC E1-E0 (context)": round(_r.loc["E1", "ROC-AUC"] - _r.loc["E0", "ROC-AUC"], 4),
                "dAUC E2-E1 (weather)": round(_r.loc["E2", "ROC-AUC"] - _r.loc["E1", "ROC-AUC"], 4),
            }
        )
    _clf_gain = pd.DataFrame(_rows)
    _clf_gain.to_csv("outputs/enroute/gain_classification.csv", index=False)

    _rows = []
    for (_cp, _model), _g in reg_results[reg_results["Set"] != "-"].groupby(["checkpoint_%", "Model"]):
        _r = _g.set_index("Set")
        _rows.append(
            {
                "checkpoint_%": _cp,
                "Model": _model,
                "dMAE E1-E0 (context)": round(_r.loc["E1", "MAE"] - _r.loc["E0", "MAE"], 4),
                "dMAE E2-E1 (weather)": round(_r.loc["E2", "MAE"] - _r.loc["E1", "MAE"], 4),
                "dR2 E1-E0 (context)": round(_r.loc["E1", "R2"] - _r.loc["E0", "R2"], 4),
                "dR2 E2-E1 (weather)": round(_r.loc["E2", "R2"] - _r.loc["E1", "R2"], 4),
            }
        )
    pd.DataFrame(_rows).to_csv("outputs/enroute/gain_regression.csv", index=False)
    _clf_gain
    return


@app.cell
def _(clf_results, plt, reg_results):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.5))
    _colors = {"E0": "gray", "E1": "royalblue", "E2": "firebrick"}
    for _name, _style in [("Gradient Boosting (HistGB, Balanced)", "-"), ("Logistic Regression (Balanced)", "--")]:
        for _set, _color in _colors.items():
            _r = clf_results[(clf_results["Model"] == _name) & (clf_results["Set"] == _set)]
            _ax_label = f"{_set} {'HistGB' if 'Gradient' in _name else 'LR'}"
            _axes[0].plot(_r["checkpoint_%"], _r["ROC-AUC"], _style, color=_color, marker="o", label=_ax_label)
    # best pre-departure classifier from notebook_modeling_v3 (outputs/it5/test_results.csv, Set D)
    _axes[0].axhline(0.7007, color="black", linestyle=":", label="pre-departure best (0.701)")
    _axes[0].set_title("Late-arrival AUC on 2024 vs progress through the trip")
    _axes[0].set_xlabel("checkpoint (% of scheduled trip)")
    _axes[0].set_ylabel("ROC-AUC")
    _axes[0].legend(fontsize=8)
    _axes[0].grid(True, alpha=0.3)

    _p = reg_results[reg_results["Model"] == "Persistence (final = current)"]
    _axes[1].plot(_p["checkpoint_%"], _p["MAE"], ":", color="black", marker="o", label="persistence")
    for _set, _color in _colors.items():
        _r = reg_results[(reg_results["Model"] == "Gradient Boosting (HistGB)") & (reg_results["Set"] == _set)]
        _axes[1].plot(_r["checkpoint_%"], _r["MAE"], "-", color=_color, marker="o", label=f"{_set} HistGB")
    _axes[1].set_title("Final-delay MAE on 2024 vs progress (lower is better)")
    _axes[1].set_xlabel("checkpoint (% of scheduled trip)")
    _axes[1].set_ylabel("MAE (min)")
    _axes[1].legend(fontsize=8)
    _axes[1].grid(True, alpha=0.3)
    _fig.savefig("outputs/enroute/metrics_by_checkpoint.png", bbox_inches="tight")
    _fig
    return


@app.cell
def _(clf_probs, df, plt, roc_auc_score, roc_curve):
    _fig, _axes = plt.subplots(1, 4, figsize=(20, 4.8), sharey=True)
    for _ax, (_cp, _g) in zip(_axes, df.groupby("checkpoint")):
        _y = _g.loc[_g["year"] == 2024, "is_late"]
        for _set, _color in [("E0", "gray"), ("E1", "royalblue"), ("E2", "firebrick")]:
            _probs = clf_probs[(_cp, "Gradient Boosting (HistGB, Balanced)", _set)]
            _fpr, _tpr, _ = roc_curve(_y, _probs)
            _ax.plot(_fpr, _tpr, color=_color, label=f"{_set} (AUC {roc_auc_score(_y, _probs):.3f})")
        _ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        _ax.set_title(f"HistGB ROC at {_cp}% of the trip")
        _ax.set_xlabel("False positive rate")
        _ax.legend(loc="lower right", fontsize=8)
        _ax.grid(True, alpha=0.3)
    _axes[0].set_ylabel("True positive rate")
    _fig.savefig("outputs/enroute/roc_by_checkpoint.png", bbox_inches="tight")
    _fig
    return


@app.cell
def _(clf_probs, df, pd):
    _parts = []
    for _cp, _g in df.groupby("checkpoint"):
        _t = _g[_g["year"] == 2024]
        _part = pd.DataFrame({"checkpoint_%": _cp, "departure_date": _t["departure_date"].to_numpy(), "is_late": _t["is_late"].to_numpy()})
        for _set in ["E0", "E1", "E2"]:
            _part[f"prob_{_set}"] = clf_probs[(_cp, "Gradient Boosting (HistGB, Balanced)", _set)]
        _parts.append(_part)
    pd.concat(_parts).to_csv("outputs/enroute/test_probabilities.csv", index=False)
    return


if __name__ == "__main__":
    app.run()
