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
    os.makedirs("outputs/rq", exist_ok=True)
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
    encoded = _encoded
    return encoded, feature_sets, weather_groups


@app.cell
def _(
    HistGradientBoostingClassifier,
    LogisticRegression,
    StandardScaler,
    encoded,
    feature_sets,
    make_pipeline,
    pd,
    roc_auc_score,
):
    # train on all years before Y, test on Y (contiguous blocks, course 3.8)
    # note: medians / top corridors in the prep above come from 2018-2023, a small leak for Y < 2024
    _rows = []
    for _year in [2021, 2022, 2023, 2024]:
        _train = encoded[encoded["year"] < _year]
        _test = encoded[encoded["year"] == _year]
        for _name, _model, _kind in [
            (
                "Gradient Boosting (HistGB, Balanced)",
                HistGradientBoostingClassifier(
                    learning_rate=0.1, max_iter=500, max_leaf_nodes=31, class_weight="balanced", early_stopping=False, random_state=42
                ),
                "tree",
            ),
            ("Logistic Regression (Balanced)", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")), "lin"),
        ]:
            for _label, _cols in [feature_sets[_kind][i] for i in [0, 2, 3]]:
                _model.fit(_train[_cols], _train["is_late"].astype(int))
                _auc = roc_auc_score(_test["is_late"].astype(int), _model.predict_proba(_test[_cols])[:, 1])
                _rows.append(
                    {
                        "test year": _year,
                        "train years": f"2018-{_year - 1}",
                        "test late rate %": round(100 * _test["is_late"].mean(), 2),
                        "Model": _name,
                        "Feature Set": _label,
                        "ROC-AUC": round(float(_auc), 4),
                    }
                )
        print("done", _year)
    rolling = pd.DataFrame(_rows)
    rolling.to_csv("outputs/rq/rolling_years.csv", index=False)
    rolling
    return (rolling,)


@app.cell
def _(pd, plt, rolling):
    _rows = []
    for (_year, _model), _g in rolling.groupby(["test year", "Model"]):
        _auc = dict(zip(_g["Feature Set"].str[:5], _g["ROC-AUC"]))
        _rows.append(
            {
                "test year": _year,
                "Model": _model,
                "test late rate %": _g["test late rate %"].iloc[0],
                "AUC Set A": _auc["Set A"],
                "dAUC C-A (origin weather)": round(_auc["Set C"] - _auc["Set A"], 4),
                "dAUC D-A (all weather)": round(_auc["Set D"] - _auc["Set A"], 4),
            }
        )
    _gain = pd.DataFrame(_rows)
    _gain.to_csv("outputs/rq/rolling_weather_gain.csv", index=False)

    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    for _model, _style in [("Gradient Boosting (HistGB, Balanced)", "-"), ("Logistic Regression (Balanced)", "--")]:
        _m = _gain[_gain["Model"] == _model]
        _short = "HistGB" if "Gradient" in _model else "LR"
        _ax.plot(_m["test year"], _m["dAUC C-A (origin weather)"], _style, marker="o", color="royalblue", label=f"{_short}: origin weather (C - A)")
        _ax.plot(_m["test year"], _m["dAUC D-A (all weather)"], _style, marker="o", color="firebrick", label=f"{_short}: all weather (D - A)")
    _ax.axhline(0, color="black", linewidth=1)
    _ax.set_xticks([2021, 2022, 2023, 2024])
    _ax.set_xlabel("test year (trained on all earlier years)")
    _ax.set_ylabel("AUC gain from weather")
    _ax.set_title("Weather gain in every test year")
    _ax.legend(fontsize=8)
    _ax.grid(True, alpha=0.3)
    _fig.savefig("outputs/rq/rolling_weather_gain.png", bbox_inches="tight")
    _gain
    return


@app.cell
def _(
    LogisticRegression,
    StandardScaler,
    encoded,
    feature_sets,
    make_pipeline,
    pd,
    roc_auc_score,
    weather_groups,
):
    # forward selection over weather groups (course 5.4 wrapper), picked on 2023 validation
    _all_wx = weather_groups["origin"] + weather_groups["dest"] + weather_groups["route"]
    _groups = {}
    for _c in _all_wx:
        if "snow" in _c:
            _key = "snow"
        elif "temp" in _c or "cold" in _c or "freezing" in _c:
            _key = "temperature"
        elif "precip" in _c:
            _key = "precipitation"
        elif "visibility" in _c:
            _key = "visibility"
        elif "wind" in _c or "gust" in _c:
            _key = "wind"
        else:
            _key = "missing flags"
        _groups.setdefault(_key, []).append(_c)

    _fit = encoded[encoded["year"] < 2023]
    _fit = _fit.sample(n=min(200000, len(_fit)), random_state=42)  # sample to keep it quick
    _val = encoded[encoded["year"] == 2023]
    _base = feature_sets["lin"][0][1]

    def _val_auc(cols):
        _m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        _m.fit(_fit[cols], _fit["is_late"].astype(int))
        return roc_auc_score(_val["is_late"].astype(int), _m.predict_proba(_val[cols])[:, 1])

    _chosen = []
    _current = _val_auc(_base)
    _rows = [{"step": 0, "added group": "(Set A only)", "n weather features": 0, "val AUC": round(float(_current), 4), "gain": None}]
    _left = list(_groups)
    while _left:
        _scores = {g: _val_auc(_base + [c for k in _chosen + [g] for c in _groups[k]]) for g in _left}
        _best = max(_scores, key=lambda g: _scores[g])
        _chosen.append(_best)
        _left.remove(_best)
        _rows.append(
            {
                "step": len(_chosen),
                "added group": _best,
                "n weather features": sum(len(_groups[k]) for k in _chosen),
                "val AUC": round(float(_scores[_best]), 4),
                "gain": round(float(_scores[_best] - _current), 4),
            }
        )
        _current = _scores[_best]
        print("added", _best, round(float(_current), 4))
    _forward = pd.DataFrame(_rows)
    _forward.to_csv("outputs/rq/forward_selection_weather_groups.csv", index=False)
    pd.DataFrame([{"group": k, "features": ", ".join(v)} for k, v in _groups.items()]).to_csv(
        "outputs/rq/weather_group_definitions.csv", index=False
    )
    _forward
    return


if __name__ == "__main__":
    app.run()
