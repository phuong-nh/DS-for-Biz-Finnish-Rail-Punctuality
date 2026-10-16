import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from dsforbiz_finnish_rail_punctuality.extract_v2 import extract_months

    return (
        HistGradientBoostingClassifier,
        LogisticRegression,
        StandardScaler,
        extract_months,
        make_pipeline,
        os,
        pd,
        plt,
        roc_auc_score,
    )


@app.cell
def _(extract_months, os, pd):
    os.makedirs("outputs/it6", exist_ok=True)
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

    # cause codes are logged after the delay happens, so they are only used as targets here, never as features
    groups = ["infra", "operator", "traffic"]
    for _g in groups:
        _encoded[f"late_{_g}"] = ((_encoded["is_late"] == 1) & (_encoded["cause_dominant"] == _g)).astype(int)

    train_df = _encoded[_encoded["year"] < 2024]
    test_df = _encoded[_encoded["year"] == 2024]
    all_df = _encoded
    return all_df, feature_sets, groups, test_df, train_df


@app.cell
def _(all_df, groups, pd, plt):
    _late = all_df[all_df["is_late"] == 1]
    _dom_year = (100 * pd.crosstab(_late["year"], _late["cause_dominant"], normalize="index")).round(2)
    _dom_year.to_csv("outputs/it6/cause_dominant_by_year.csv")

    _rates = (100 * all_df.groupby("year")[["is_late"] + [f"late_{g}" for g in groups]].mean()).round(2)
    _rates.to_csv("outputs/it6/cause_rates_by_year.csv")

    _top = all_df["corridor"].value_counts().head(10).index
    _corr = _late[_late["corridor"].isin(_top)]
    _by_corr = (100 * pd.crosstab(_corr["corridor"], _corr["cause_dominant"], normalize="index")).round(2)
    _by_corr["late_rate_%"] = (100 * all_df[all_df["corridor"].isin(_top)].groupby("corridor")["is_late"].mean()).round(2)
    _by_corr.to_csv("outputs/it6/cause_by_corridor.csv")

    # how concentrated is each group on a few train numbers? (systematic problems should be concentrated)
    _train_years = all_df[all_df["year"] < 2024]
    _rows = []
    for _target in ["is_late"] + [f"late_{g}" for g in groups]:
        _counts = _train_years.groupby("train_number")[_target].sum().sort_values(ascending=False)
        _top10 = _counts.head(max(1, len(_counts) // 10)).sum()
        _trips = _train_years.groupby("train_number").size()
        _rows.append(
            {
                "target": _target,
                "events_2018_2023": int(_counts.sum()),
                "share_from_top10pct_train_numbers_%": round(100 * _top10 / _counts.sum(), 2),
                "share_of_trips_in_those_trains_%": round(100 * _trips.loc[_counts.head(len(_counts) // 10).index].sum() / _trips.sum(), 2),
            }
        )
    _conc = pd.DataFrame(_rows)
    _conc.to_csv("outputs/it6/cause_concentration.csv", index=False)

    _fig, _ax = plt.subplots(figsize=(9, 4.5))
    for _g, _color in zip(groups, ["firebrick", "darkorange", "steelblue"]):
        _ax.plot(_rates.index, _rates[f"late_{_g}"], marker="o", color=_color, label=f"late, mainly {_g}")
    _ax.set_ylabel("% of all regular trains")
    _ax.set_title("Late trains by main cause group and year")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _fig.savefig("outputs/it6/cause_rates_by_year.png", bbox_inches="tight")
    print(_conc)
    _rates
    return


@app.cell
def _(
    HistGradientBoostingClassifier,
    LogisticRegression,
    StandardScaler,
    feature_sets,
    groups,
    make_pipeline,
    pd,
    roc_auc_score,
    test_df,
    train_df,
):
    # same settings as notebook_modeling_v3, no new tuning
    _models = {
        "Gradient Boosting (HistGB, Balanced)": (
            HistGradientBoostingClassifier(
                learning_rate=0.1, max_iter=500, max_leaf_nodes=31, class_weight="balanced", early_stopping=False, random_state=42
            ),
            "tree",
        ),
        "Logistic Regression (Balanced)": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
            "lin",
        ),
    }
    _rows = []
    for _target in ["is_late"] + [f"late_{g}" for g in groups]:
        for _name, (_model, _kind) in _models.items():
            for _label, _cols in [feature_sets[_kind][i] for i in [0, 2, 3]]:
                _model.fit(train_df[_cols], train_df[_target])
                _probs = _model.predict_proba(test_df[_cols])[:, 1]
                _rows.append(
                    {
                        "target": _target,
                        "Model": _name,
                        "Feature Set": _label,
                        "positive_rate_train_%": round(100 * train_df[_target].mean(), 2),
                        "positive_rate_test_%": round(100 * test_df[_target].mean(), 2),
                        "ROC-AUC": round(float(roc_auc_score(test_df[_target], _probs)), 4),
                    }
                )
        print("done", _target)
    cause_models = pd.DataFrame(_rows)
    cause_models.to_csv("outputs/it6/cause_models.csv", index=False)
    cause_models
    return (cause_models,)


@app.cell
def _(cause_models, pd, plt):
    _rows = []
    for (_target, _model), _g in cause_models.groupby(["target", "Model"], sort=False):
        _auc = dict(zip(_g["Feature Set"].str[:5], _g["ROC-AUC"]))
        _rows.append(
            {
                "target": _target,
                "Model": _model,
                "systematic (AUC A - 0.5)": round(_auc["Set A"] - 0.5, 4),
                "origin weather (AUC C - A)": round(_auc["Set C"] - _auc["Set A"], 4),
                "all weather (AUC D - A)": round(_auc["Set D"] - _auc["Set A"], 4),
                "unexplained (1 - AUC D)": round(1 - _auc["Set D"], 4),
            }
        )
    _dec = pd.DataFrame(_rows)
    _dec.to_csv("outputs/it6/cause_decomposition.csv", index=False)

    _h = _dec[_dec["Model"].str.startswith("Gradient")].set_index("target")
    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    _ax.bar(_h.index, _h["systematic (AUC A - 0.5)"], color="gray", label="calendar + route + trip (AUC A - 0.5)")
    _ax.bar(_h.index, _h["all weather (AUC D - A)"], bottom=_h["systematic (AUC A - 0.5)"], color="firebrick", label="added by weather (AUC D - A)")
    _ax.set_ylabel("AUC above random (0.5)")
    _ax.set_title("How predictable is each kind of lateness? (HistGB, 2024 test)")
    _ax.legend()
    _ax.grid(True, axis="y", alpha=0.3)
    _fig.savefig("outputs/it6/cause_decomposition.png", bbox_inches="tight")
    _dec
    return


if __name__ == "__main__":
    app.run()
