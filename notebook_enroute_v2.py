import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import (
        mean_absolute_error,
        r2_score,
        roc_auc_score,
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
        make_pipeline,
        mean_absolute_error,
        os,
        pd,
        plt,
        r2_score,
        roc_auc_score,
        root_mean_squared_error,
    )


@app.cell
def _(os, pd):
    os.makedirs("outputs/enroute_v2", exist_ok=True)
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

    # some trains are late again and again: late rate of the same train number, only from 2018-2022
    # so it does not leak into the 2023 validation or the 2024 test, NaN if fewer than 30 trips
    _hist = _df[(_df["checkpoint"] == 0) & (_df["year"] < 2023)].groupby("train_number")["is_late"].agg(["mean", "size"])
    _df["train_hist_late_rate"] = _df["train_number"].map(_hist["mean"].where(_hist["size"] >= 30))
    _df["train_hist_missing"] = _df["train_hist_late_rate"].isna().astype(int)
    _df["train_hist_late_rate"] = _df["train_hist_late_rate"].fillna(_df.loc[_df["year"] < 2023, "is_late"].mean())
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

    # E0 = current delay only, E1 = + calendar / route / trip / train history, E2 = + weather now and ahead
    _e0 = ["current_delay"]
    _hist_cols = ["train_hist_late_rate", "train_hist_missing"]
    _e1_lin = _e0 + ["delay_change"] + _cal_onehot + _trip + _route + _hist_cols
    _e1_tree = _e0 + ["delay_change", "departure_hour_local", "day_of_week", "month", "is_winter", "is_weekend"] + _trip + _route + _hist_cols
    sets = {
        "lin": {"E0": _e0, "E1": _e1_lin, "E2": _e1_lin + weather_cols},
        "tree": {"E0": _e0, "E1": _e1_tree, "E2": _e1_tree + weather_cols},
    }
    print("features E1 lin / tree:", len(_e1_lin), len(_e1_tree), "| weather:", len(weather_cols))
    return df, sets


@app.cell
def _(HistGradientBoostingClassifier, df, pd, roc_auc_score, sets):
    # tune once on the 50% checkpoint (fit 2018-2022, validate 2023), then use for every checkpoint
    _g = df[df["checkpoint"] == 50]
    _fit, _val = _g[_g["year"] < 2023], _g[_g["year"] == 2023]
    _cols = sets["tree"]["E2"]
    _rows = []
    for _lr in [0.05, 0.1]:
        for _iters in [200, 500]:
            for _leaves in [31, 63]:
                _m = HistGradientBoostingClassifier(
                    learning_rate=_lr, max_iter=_iters, max_leaf_nodes=_leaves, class_weight="balanced", early_stopping=False, random_state=42
                )
                _m.fit(_fit[_cols], _fit["is_late"])
                _rows.append(
                    {
                        "learning_rate": _lr,
                        "max_iter": _iters,
                        "max_leaf_nodes": _leaves,
                        "val AUC": round(float(roc_auc_score(_val["is_late"], _m.predict_proba(_val[_cols])[:, 1])), 4),
                    }
                )
    _tuning = pd.DataFrame(_rows).sort_values("val AUC", ascending=False)
    _tuning.to_csv("outputs/enroute_v2/hgb_tuning.csv", index=False)
    best_hgb = _tuning.iloc[0]
    print("best:", dict(best_hgb))
    _tuning
    return (best_hgb,)


@app.cell
def _(
    HistGradientBoostingClassifier,
    LogisticRegression,
    StandardScaler,
    best_hgb,
    df,
    make_pipeline,
    pd,
    roc_auc_score,
    sets,
):
    def _hgb():
        return HistGradientBoostingClassifier(
            learning_rate=float(best_hgb["learning_rate"]),
            max_iter=int(best_hgb["max_iter"]),
            max_leaf_nodes=int(best_hgb["max_leaf_nodes"]),
            class_weight="balanced",
            early_stopping=False,
            random_state=42,
        )

    _models = {
        "Logistic Regression (Balanced)": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
            "lin",
        ),
        "Gradient Boosting (HistGB, tuned)": (_hgb(), "tree"),
    }
    _rows = []
    for _cp, _g in df.groupby("checkpoint"):
        _train, _test = _g[_g["year"] < 2024], _g[_g["year"] == 2024]
        for _name, (_model, _kind) in _models.items():
            for _set, _cols in sets[_kind].items():
                _model.fit(_train[_cols], _train["is_late"])
                _auc = roc_auc_score(_test["is_late"], _model.predict_proba(_test[_cols])[:, 1])
                _rows.append({"checkpoint_%": _cp, "Model": _name, "Set": _set, "ROC-AUC": round(float(_auc), 4)})
        print("done checkpoint", _cp)

    # one model for all checkpoints, the checkpoint itself is a feature
    _cols = sets["tree"]["E2"] + ["checkpoint"]
    _train, _test = df[df["year"] < 2024], df[df["year"] == 2024]
    _pooled = _hgb()
    _pooled.fit(_train[_cols], _train["is_late"])
    _probs = _pooled.predict_proba(_test[_cols])[:, 1]
    for _cp in sorted(df["checkpoint"].unique()):
        _k = (_test["checkpoint"] == _cp).to_numpy()
        _auc = roc_auc_score(_test["is_late"].to_numpy()[_k], _probs[_k])
        _rows.append({"checkpoint_%": _cp, "Model": "Gradient Boosting (HistGB, tuned, pooled)", "Set": "E2", "ROC-AUC": round(float(_auc), 4)})

    clf_results = pd.DataFrame(_rows)
    clf_results.to_csv("outputs/enroute_v2/classification_by_checkpoint.csv", index=False)
    clf_results
    return (clf_results,)


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
        "Linear Regression (OLS)": (make_pipeline(StandardScaler(), LinearRegression()), "lin", ["E1", "E2"]),
        "Gradient Boosting (HistGB, squared loss)": (
            HistGradientBoostingRegressor(learning_rate=0.05, max_iter=200, max_leaf_nodes=31, early_stopping=False, random_state=42),
            "tree",
            ["E0", "E1", "E2"],
        ),
        # robust loss, like Huber in notebook_regression_v2, should help MAE
        "Gradient Boosting (HistGB, absolute loss)": (
            HistGradientBoostingRegressor(
                loss="absolute_error", learning_rate=0.05, max_iter=200, max_leaf_nodes=31, early_stopping=False, random_state=42
            ),
            "tree",
            ["E1", "E2"],
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
        _rows.append(_metrics(_cp, "Persistence (final = current)", "-", _y_test, _test["current_delay"]))
        _y_train = _train["delay_to_go"].clip(lower=-15, upper=120)
        for _name, (_model, _kind, _set_names) in _models.items():
            for _set in _set_names:
                _cols = sets[_kind][_set]
                _model.fit(_train[_cols], _y_train)
                _pred_final = _test["current_delay"].to_numpy() + _model.predict(_test[_cols])
                _rows.append(_metrics(_cp, _name, _set, _y_test, _pred_final))
        print("done checkpoint", _cp)
    reg_results = pd.DataFrame(_rows)
    reg_results.to_csv("outputs/enroute_v2/regression_by_checkpoint.csv", index=False)
    reg_results
    return (reg_results,)


@app.cell
def _(clf_results, pd, reg_results):
    # compare with the first en-route notebook (outputs/enroute)
    _v1_clf = pd.read_csv("outputs/enroute/classification_by_checkpoint.csv")
    _v1 = _v1_clf[(_v1_clf["Model"].str.startswith("Gradient")) & (_v1_clf["Set"] == "E2")].set_index("checkpoint_%")["ROC-AUC"]
    _rows = []
    for _cp in _v1.index:
        _c = clf_results[clf_results["checkpoint_%"] == _cp].set_index(["Model", "Set"])["ROC-AUC"]
        _rows.append(
            {
                "checkpoint_%": _cp,
                "v1 HistGB E2": _v1.loc[_cp],
                "v2 HistGB tuned E1": _c.loc[("Gradient Boosting (HistGB, tuned)", "E1")],
                "v2 HistGB tuned E2": _c.loc[("Gradient Boosting (HistGB, tuned)", "E2")],
                "v2 pooled E2": _c.loc[("Gradient Boosting (HistGB, tuned, pooled)", "E2")],
                "weather gain v2 (E2-E1)": round(
                    _c.loc[("Gradient Boosting (HistGB, tuned)", "E2")] - _c.loc[("Gradient Boosting (HistGB, tuned)", "E1")], 4
                ),
            }
        )
    _cmp = pd.DataFrame(_rows)
    _cmp["v2 - v1"] = (_cmp["v2 HistGB tuned E2"] - _cmp["v1 HistGB E2"]).round(4)
    _cmp.to_csv("outputs/enroute_v2/comparison_with_v1.csv", index=False)

    _r = reg_results.pivot_table(index="checkpoint_%", columns=["Model", "Set"], values="MAE")
    _r.columns = [f"{m} {s}" for m, s in _r.columns]
    _r.round(4).to_csv("outputs/enroute_v2/mae_by_checkpoint_wide.csv")
    _cmp
    return


@app.cell
def _(clf_results, plt, reg_results):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for _name, _set, _color, _style in [
        ("Gradient Boosting (HistGB, tuned)", "E0", "gray", "-"),
        ("Gradient Boosting (HistGB, tuned)", "E1", "royalblue", "-"),
        ("Gradient Boosting (HistGB, tuned)", "E2", "firebrick", "-"),
        ("Gradient Boosting (HistGB, tuned, pooled)", "E2", "darkorange", "--"),
        ("Logistic Regression (Balanced)", "E2", "firebrick", ":"),
    ]:
        _r = clf_results[(clf_results["Model"] == _name) & (clf_results["Set"] == _set)]
        _axes[0].plot(_r["checkpoint_%"], _r["ROC-AUC"], _style, color=_color, marker="o", label=f"{_name.replace('Gradient Boosting ', 'HGB ')} {_set}")
    _axes[0].axhline(0.7007, color="black", linestyle=":", label="pre-departure best (0.701)")
    _axes[0].set_title("Late-arrival AUC on 2024 vs progress (v2)")
    _axes[0].set_xlabel("checkpoint (% of scheduled trip)")
    _axes[0].set_ylabel("ROC-AUC")
    _axes[0].legend(fontsize=7)
    _axes[0].grid(True, alpha=0.3)

    for _name, _set, _color, _style in [
        ("Persistence (final = current)", "-", "black", ":"),
        ("Gradient Boosting (HistGB, squared loss)", "E2", "firebrick", "-"),
        ("Gradient Boosting (HistGB, absolute loss)", "E2", "darkgreen", "-"),
        ("Linear Regression (OLS)", "E2", "royalblue", "--"),
    ]:
        _r = reg_results[(reg_results["Model"] == _name) & (reg_results["Set"] == _set)]
        _axes[1].plot(_r["checkpoint_%"], _r["MAE"], _style, color=_color, marker="o", label=f"{_name} {_set}")
    _axes[1].set_title("Final-delay MAE on 2024 vs progress (v2, lower is better)")
    _axes[1].set_xlabel("checkpoint (% of scheduled trip)")
    _axes[1].set_ylabel("MAE (min)")
    _axes[1].legend(fontsize=7)
    _axes[1].grid(True, alpha=0.3)
    _fig.savefig("outputs/enroute_v2/metrics_by_checkpoint.png", bbox_inches="tight")
    _fig
    return


if __name__ == "__main__":
    app.run()
