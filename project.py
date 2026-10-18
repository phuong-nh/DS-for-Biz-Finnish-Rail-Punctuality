import os

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from dsforbiz_finnish_rail_punctuality.extract_v2 import extract_months


def main():
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

    # clip bounds from notebook_regression_v2 (outputs/it4/clipping_report.csv)
    y_train = train_df["delay_minutes"].clip(lower=-15, upper=120)
    y_test = test_df["delay_minutes"]
    y_test_late = test_df["is_late"].astype(int)

    _cols = feature_sets["tree"][-1][1]
    _y_fit = fit_df["is_late"].astype(int)
    _y_val = val_df["is_late"].astype(int)

    _rows = []
    for _lr in [0.05, 0.1]:
        for _iters in [200, 500]:
            for _leaves in [31, 63]:
                _hgb = HistGradientBoostingClassifier(
                    learning_rate=_lr,
                    max_iter=_iters,
                    max_leaf_nodes=_leaves,
                    class_weight="balanced",
                    early_stopping=False,
                    random_state=42,
                )
                _hgb.fit(fit_df[_cols], _y_fit)
                _rows.append(
                    {
                        "learning_rate": _lr,
                        "max_iter": _iters,
                        "max_leaf_nodes": _leaves,
                        "train AUC": round(float(roc_auc_score(_y_fit, _hgb.predict_proba(fit_df[_cols])[:, 1])), 4),
                        "val AUC": round(float(roc_auc_score(_y_val, _hgb.predict_proba(val_df[_cols])[:, 1])), 4),
                    }
                )
    hgb_tuning = pd.DataFrame(_rows).sort_values("val AUC", ascending=False)
    best_hgb = hgb_tuning.iloc[0]
    print("best HistGB:", dict(best_hgb))
    print(hgb_tuning)

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

    best_model_name = "Gradient Boosting (HistGB, Balanced)"
    models = {
        "Gradient Boosting (HistGB, Balanced)": (
            HistGradientBoostingClassifier(
                learning_rate=float(best_hgb["learning_rate"]),
                max_iter=int(best_hgb["max_iter"]),
                max_leaf_nodes=int(best_hgb["max_leaf_nodes"]),
                class_weight="balanced",
                early_stopping=False,
                random_state=42,
            ),
            "tree",
        ),
    }

    _model, _kind = models[best_model_name]
    _rows = []
    test_probs = {}
    for _f_label, _f_cols in feature_sets[_kind]:
        _model.fit(train_df[_f_cols], train_df["is_late"].astype(int))
        _probs = _model.predict_proba(test_df[_f_cols])[:, 1]
        test_probs[_f_label] = _probs
        _rows.append(score(best_model_name, _f_label, test_df["is_late"].astype(int), _probs))
    test_results = pd.DataFrame(_rows)

    _probs_df = pd.DataFrame({"departure_date": test_df["departure_date"].to_numpy(), "is_late": test_df["is_late"].astype(int).to_numpy()})
    for _i, _label in enumerate(test_probs):
        _probs_df[f"prob_set_{'ABCD'[_i]}"] = test_probs[_label]
    print(test_results)

    _cols = feature_sets["tree"][-1][1]
    _is_fit = train_df["year"] < 2023
    _y_val = train_df.loc[~_is_fit, "delay_minutes"]

    _rows = []
    for _lr in [0.05, 0.1]:
        for _iters in [200, 500]:
            for _leaves in [31, 63]:
                _hgb = HistGradientBoostingRegressor(
                    learning_rate=_lr, max_iter=_iters, max_leaf_nodes=_leaves, early_stopping=False, random_state=42
                )
                _hgb.fit(train_df.loc[_is_fit, _cols], y_train[_is_fit])
                _pred = _hgb.predict(train_df.loc[~_is_fit, _cols])
                _rows.append(
                    {
                        "learning_rate": _lr,
                        "max_iter": _iters,
                        "max_leaf_nodes": _leaves,
                        "Val MAE": round(float(mean_absolute_error(_y_val, _pred)), 4),
                        "Val RMSE": round(float(root_mean_squared_error(_y_val, _pred)), 4),
                    }
                )
    hgb_df = pd.DataFrame(_rows).sort_values("Val RMSE")
    best_hgb = hgb_df.iloc[0]
    print(hgb_df)

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
        (
            "Gradient Boosting (HistGB)",
            HistGradientBoostingRegressor(
                learning_rate=float(best_hgb["learning_rate"]),
                max_iter=int(best_hgb["max_iter"]),
                max_leaf_nodes=int(best_hgb["max_leaf_nodes"]),
                early_stopping=False,
                random_state=42,
            ),
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
    print(pd.DataFrame(reg_results))

    pre = _probs_df
    train_prior = round(float(train_df["is_late"].mean()), 4)

    def day_bootstrap(frame, col_a, col_b, n_boot=1000, by_day=True, seed=42):
        # resample whole days, trains on the same day are not independent (cascading delays)
        _rng = np.random.default_rng(seed)
        _y = frame["is_late"].to_numpy()
        _a = frame[col_a].to_numpy()
        _b = frame[col_b].to_numpy()
        _point = roc_auc_score(_y, _b) - roc_auc_score(_y, _a)
        if by_day:
            _groups = list(frame.groupby("departure_date").indices.values())
        else:
            _groups = [np.array([i]) for i in range(len(frame))]
        _diffs = []
        for _ in range(n_boot):
            _pick = _rng.integers(0, len(_groups), size=len(_groups))
            _ix = np.concatenate([_groups[i] for i in _pick])
            _diffs.append(roc_auc_score(_y[_ix], _b[_ix]) - roc_auc_score(_y[_ix], _a[_ix]))
        _diffs = np.array(_diffs)
        return {
            "point dAUC": round(float(_point), 4),
            "ci_low": round(float(np.percentile(_diffs, 2.5)), 4),
            "ci_high": round(float(np.percentile(_diffs, 97.5)), 4),
            "share_boot_le_0": round(float((_diffs <= 0).mean()), 4),
        }

    _rows = []
    for _set, _name in [("B", "destination weather"), ("C", "origin weather"), ("D", "all weather")]:
        _rows.append({"comparison": f"pre-departure Set {_set} vs A ({_name})", "resampling": "by day"} | day_bootstrap(pre, "prob_set_A", f"prob_set_{_set}"))
    # same comparison but resampling single trains, to see how much the day dependence matters
    _rows.append({"comparison": "pre-departure Set C vs A (origin weather)", "resampling": "single trains (iid)"} | day_bootstrap(pre, "prob_set_A", "prob_set_C", by_day=False, n_boot=300))
    boot = pd.DataFrame(_rows)
    boot["ci_width"] = (boot["ci_high"] - boot["ci_low"]).round(4)
    print(boot)

    def to_real_prob(p, prior):
        # balanced weights act like a 50/50 prior, move the odds back to the real late rate
        _odds = p / (1 - p) * prior / (1 - prior)
        return _odds / (1 + _odds)

    scores = {}
    for _set in ["A", "C", "D"]:
        scores[f"pre-departure Set {_set}"] = (pre["is_late"].to_numpy(), to_real_prob(pre[f"prob_set_{_set}"].to_numpy(), train_prior))
    _rows = []
    for _name, (_y, _p) in scores.items():
        _balanced = _p / (1 - _p) * (1 - train_prior) / train_prior
        _rows.append(
            {
                "score": _name,
                "actual late rate 2024": round(float(_y.mean()), 4),
                "mean balanced prob": round(float(np.mean(_balanced / (1 + _balanced))), 4),
                "mean corrected prob": round(float(np.mean(_p)), 4),
            }
        )
    _calib = pd.DataFrame(_rows)
    print(_calib)

    _rows = []
    for _r in [2, 3, 5, 7, 10]:
        for _name, (_y, _p) in scores.items():
            _act = _p > 1 / _r
            _tp = int((_act & (_y == 1)).sum())
            _fp = int((_act & (_y == 0)).sum())
            _fn = int((~_act & (_y == 1)).sum())
            _n = len(_y)
            _cost = (_tp + _fp + _r * _fn) / _n
            _never = _r * _y.mean()
            _always = 1.0
            _rows.append(
                {
                    "cost ratio r (miss / action)": _r,
                    "score": _name,
                    "threshold 1/r": round(1 / _r, 3),
                    "share flagged": round(float(_act.mean()), 4),
                    "recall": round(_tp / max(_tp + _fn, 1), 4),
                    "precision": round(_tp / max(_tp + _fp, 1), 4),
                    "cost per train": round(float(_cost), 4),
                    "never act": round(float(_never), 4),
                    "always act": _always,
                    "saving vs best simple rule per 1000 trains": round(float(1000 * (min(_never, _always) - _cost)), 1),
                }
            )
    ev = pd.DataFrame(_rows)
    print(ev)

    # value of weather = cost without weather - cost with weather
    _w = ev.pivot_table(index="cost ratio r (miss / action)", columns="score", values="cost per train")
    _value = pd.DataFrame(
        {
            "pre-departure: Set A - Set C (per 1000 trains)": (1000 * (_w["pre-departure Set A"] - _w["pre-departure Set C"])).round(1),
            "pre-departure: Set A - Set D (per 1000 trains)": (1000 * (_w["pre-departure Set A"] - _w["pre-departure Set D"])).round(1),
        }
    )
    print(_value)


if __name__ == "__main__":
    main()
