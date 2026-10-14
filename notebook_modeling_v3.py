import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

    from dsforbiz_finnish_rail_punctuality.extract_v2 import extract_months

    return (
        DecisionTreeClassifier,
        LogisticRegression,
        RandomForestClassifier,
        StandardScaler,
        accuracy_score,
        export_text,
        extract_months,
        make_pipeline,
        os,
        pd,
        plot_tree,
        plt,
        precision_score,
        recall_score,
        roc_auc_score,
    )


@app.cell
def _(extract_months, os, pd):
    os.makedirs("outputs/it5", exist_ok=True)
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

    # 2018-2022 fit, 2023 validation (pick the model), 2024 test only once at the end
    fit_df = _encoded[_encoded["year"] < 2023]
    val_df = _encoded[_encoded["year"] == 2023]
    train_df = _encoded[_encoded["year"] < 2024]
    test_df = _encoded[_encoded["year"] == 2024]
    print("fit:", len(fit_df), "| val:", len(val_df), "| train (fit+val):", len(train_df), "| test:", len(test_df))
    print("late rate fit / val / test:", round(fit_df["is_late"].mean(), 4), round(val_df["is_late"].mean(), 4), round(test_df["is_late"].mean(), 4))
    return feature_sets, fit_df, test_df, train_df, val_df


@app.cell
def _(
    DecisionTreeClassifier,
    RandomForestClassifier,
    feature_sets,
    fit_df,
    pd,
    roc_auc_score,
    val_df,
):
    _cols = feature_sets["tree"][-1][1]
    _y_fit = fit_df["is_late"].astype(int)
    _y_val = val_df["is_late"].astype(int)

    _rows = []
    for _depth in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
        for _leaf in [1, 200]:
            _dt = DecisionTreeClassifier(max_depth=_depth, min_samples_leaf=_leaf, class_weight="balanced", random_state=42)
            _dt.fit(fit_df[_cols], _y_fit)
            _rows.append(
                {
                    "model": "Decision Tree",
                    "max_depth": _depth,
                    "min_samples_leaf": _leaf,
                    "train AUC": round(float(roc_auc_score(_y_fit, _dt.predict_proba(fit_df[_cols])[:, 1])), 4),
                    "val AUC": round(float(roc_auc_score(_y_val, _dt.predict_proba(val_df[_cols])[:, 1])), 4),
                }
            )
    # 50 trees instead of 100 to keep the tuning time ok
    for _depth in [4, 6, 8, 10, 12, 16]:
        for _leaf in [1, 200]:
            _rf = RandomForestClassifier(
                n_estimators=50, max_depth=_depth, min_samples_leaf=_leaf, class_weight="balanced", random_state=42, n_jobs=-1
            )
            _rf.fit(fit_df[_cols], _y_fit)
            _rows.append(
                {
                    "model": "Random Forest",
                    "max_depth": _depth,
                    "min_samples_leaf": _leaf,
                    "train AUC": round(float(roc_auc_score(_y_fit, _rf.predict_proba(fit_df[_cols])[:, 1])), 4),
                    "val AUC": round(float(roc_auc_score(_y_val, _rf.predict_proba(val_df[_cols])[:, 1])), 4),
                }
            )
    tuning = pd.DataFrame(_rows)
    tuning.to_csv("outputs/it5/tree_tuning.csv", index=False)

    best_dt = tuning[tuning["model"] == "Decision Tree"].sort_values("val AUC", ascending=False).iloc[0]
    best_rf = tuning[tuning["model"] == "Random Forest"].sort_values("val AUC", ascending=False).iloc[0]
    print("best DT:", dict(best_dt))
    print("best RF:", dict(best_rf))
    tuning
    return best_dt, best_rf, tuning


@app.cell
def _(plt, tuning):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for _ax, _model in zip(_axes, ["Decision Tree", "Random Forest"]):
        _t = tuning[tuning["model"] == _model]
        for _leaf, _style in [(1, "-"), (200, "--")]:
            _tl = _t[_t["min_samples_leaf"] == _leaf]
            _ax.plot(_tl["max_depth"], _tl["train AUC"], _style, color="steelblue", marker="o", label=f"train (2018-22), leaf={_leaf}")
            _ax.plot(_tl["max_depth"], _tl["val AUC"], _style, color="darkorange", marker="o", label=f"validation (2023), leaf={_leaf}")
        _ax.set_title(f"{_model}: train vs validation AUC (Set D)")
        _ax.set_xlabel("max_depth")
        _ax.grid(True, alpha=0.3)
    _axes[0].set_ylabel("ROC-AUC")
    _axes[1].legend(loc="upper left", fontsize=8)
    _fig.savefig("outputs/it5/bias_variance_depth.png", bbox_inches="tight")
    _fig
    return


@app.cell
def _(
    DecisionTreeClassifier,
    LogisticRegression,
    RandomForestClassifier,
    StandardScaler,
    best_dt,
    best_rf,
    make_pipeline,
):
    _dt_depth, _dt_leaf = int(best_dt["max_depth"]), int(best_dt["min_samples_leaf"])
    _rf_depth, _rf_leaf = int(best_rf["max_depth"]), int(best_rf["min_samples_leaf"])
    models = {
        "Logistic Regression (Balanced)": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
            "lin",
        ),
        f"Decision Tree (depth {_dt_depth}, leaf {_dt_leaf}, Balanced)": (
            DecisionTreeClassifier(max_depth=_dt_depth, min_samples_leaf=_dt_leaf, class_weight="balanced", random_state=42),
            "tree",
        ),
        f"Random Forest (depth {_rf_depth}, leaf {_rf_leaf}, Balanced)": (
            RandomForestClassifier(
                n_estimators=100,
                max_depth=_rf_depth,
                min_samples_leaf=_rf_leaf,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "tree",
        ),
    }
    return (models,)


@app.cell
def _(accuracy_score, precision_score, recall_score, roc_auc_score):
    def score(model_name, feature_set, y_true, probs):
        _preds = (probs >= 0.5).astype(int)
        return {
            "Model": model_name,
            "Feature Set": feature_set,
            "ROC-AUC": round(float(roc_auc_score(y_true, probs)), 4),
            "Accuracy": round(float(accuracy_score(y_true, _preds)), 4),
            "Precision": round(float(precision_score(y_true, _preds, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_true, _preds, zero_division=0)), 4),
        }

    return (score,)


@app.cell
def _(feature_sets, fit_df, models, pd, score, val_df):
    _rows = []
    for _name, (_model, _kind) in models.items():
        for _f_label, _f_cols in feature_sets[_kind]:
            _model.fit(fit_df[_f_cols], fit_df["is_late"].astype(int))
            _probs = _model.predict_proba(val_df[_f_cols])[:, 1]
            _rows.append(score(_name, _f_label, val_df["is_late"].astype(int), _probs))
    val_results = pd.DataFrame(_rows)
    val_results.to_csv("outputs/it5/validation_results.csv", index=False)

    best_model_name = val_results.groupby("Model")["ROC-AUC"].mean().idxmax()
    print("chosen on 2023 validation (mean AUC over sets A-D):", best_model_name)
    val_results
    return (best_model_name,)


@app.cell
def _(best_model_name, feature_sets, models, pd, score, test_df, train_df):
    _model, _kind = models[best_model_name]
    _rows = []
    for _f_label, _f_cols in feature_sets[_kind]:
        _model.fit(train_df[_f_cols], train_df["is_late"].astype(int))
        _probs = _model.predict_proba(test_df[_f_cols])[:, 1]
        _rows.append(score(best_model_name, _f_label, test_df["is_late"].astype(int), _probs))
    test_results = pd.DataFrame(_rows)
    test_results.to_csv("outputs/it5/test_results.csv", index=False)
    test_results
    return


@app.cell
def _(
    DecisionTreeClassifier,
    export_text,
    feature_sets,
    pd,
    plot_tree,
    plt,
    roc_auc_score,
    test_df,
    train_df,
):
    # small unweighted tree so each leaf shows the real share of late trains, Set C = known at departure
    _label, _cols = feature_sets["tree"][2]
    _tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=42)
    _tree.fit(train_df[_cols], train_df["is_late"].astype(int))
    _auc = roc_auc_score(test_df["is_late"].astype(int), _tree.predict_proba(test_df[_cols])[:, 1])
    pd.DataFrame([{"tree": "depth 3, leaf 200, unweighted", "Feature Set": _label, "test ROC-AUC": round(float(_auc), 4)}]).to_csv(
        "outputs/it5/tree_rules_summary.csv", index=False
    )

    with open("outputs/it5/tree_rules.txt", "w") as _f:
        _f.write(export_text(_tree, feature_names=list(_cols), show_weights=True))

    _fig, _ax = plt.subplots(figsize=(22, 8))
    plot_tree(
        _tree,
        feature_names=list(_cols),
        class_names=["on time", "late"],
        filled=True,
        proportion=True,
        impurity=False,
        rounded=True,
        fontsize=9,
        ax=_ax,
    )
    _ax.set_title(f"Depth-3 decision tree, {_label}, trained 2018-2023 (test AUC {_auc:.3f})")
    _fig.savefig("outputs/it5/tree_rules.png", bbox_inches="tight", dpi=120)
    _fig
    return


if __name__ == "__main__":
    app.run()
