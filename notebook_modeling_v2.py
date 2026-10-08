import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    from dsforbiz_finnish_rail_punctuality.extract import extract_months

    return (
        DecisionTreeClassifier,
        LogisticRegression,
        RandomForestClassifier,
        StandardScaler,
        accuracy_score,
        confusion_matrix,
        extract_months,
        make_pipeline,
        np,
        os,
        pd,
        plt,
        precision_score,
        recall_score,
        roc_auc_score,
    )


@app.cell
def _(extract_months, os, pd):
    os.makedirs("outputs/it2", exist_ok=True)
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
    feat_b = (
        feat_a
        + [
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
    )

    train_df = _encoded[_encoded["year"] < 2024]
    test_df = _encoded[_encoded["year"] == 2024]

    y_train = train_df["is_late"].astype(int)
    y_test = test_df["is_late"].astype(int)
    return feat_a, feat_b, test_df, train_df, y_test, y_train


@app.cell
def _(
    DecisionTreeClassifier,
    LogisticRegression,
    RandomForestClassifier,
    StandardScaler,
    accuracy_score,
    feat_a,
    feat_b,
    make_pipeline,
    np,
    precision_score,
    recall_score,
    roc_auc_score,
    test_df,
    train_df,
    y_test,
    y_train,
):
    _baseline_preds = np.zeros_like(y_test)
    results = [
        {
            "Model": "Baseline (Always On-Time)",
            "Feature Set": "None",
            "ROC-AUC": 0.5000,
            "Accuracy": round(float(accuracy_score(y_test, _baseline_preds)), 4),
            "Precision": 0.0000,
            "Recall": 0.0000,
        }
    ]

    _models = [
        (
            "Logistic Regression (Balanced)",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        ),
        (
            "Decision Tree (depth 5, Balanced)",
            DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
        ),
        (
            "Random Forest (depth 8, Balanced)",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    for _model_name, _model in _models:
        for _f_label, _f_cols in [("Set A (Calendar + Route)", feat_a), ("Set B (+ Weather)", feat_b)]:
            _clf = _model
            _clf.fit(train_df[_f_cols], y_train)
            _probs = _clf.predict_proba(test_df[_f_cols])[:, 1]
            _preds = _clf.predict(test_df[_f_cols])

            results.append(
                {
                    "Model": _model_name,
                    "Feature Set": _f_label,
                    "ROC-AUC": round(float(roc_auc_score(y_test, _probs)), 4),
                    "Accuracy": round(float(accuracy_score(y_test, _preds)), 4),
                    "Precision": round(float(precision_score(y_test, _preds, zero_division=0)), 4),
                    "Recall": round(float(recall_score(y_test, _preds, zero_division=0)), 4),
                }
            )
    return (results,)


@app.cell
def _(os, pd, results):
    os.makedirs("outputs/it2", exist_ok=True)
    _results_df = pd.DataFrame(results)
    _results_df.to_csv("outputs/it2/model_comparison_results.csv", index=False)
    _results_df
    return


@app.cell
def _(RandomForestClassifier, feat_b, pd, plt, train_df, y_train):
    _rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    _rf.fit(train_df[feat_b], y_train)

    _fi = pd.Series(_rf.feature_importances_, index=feat_b).sort_values(ascending=True).tail(12)

    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.barh(_fi.index, _fi.values, color="teal", edgecolor="black")
    _ax.set_title("Random Forest Top Feature Importances (Set B, Balanced)")
    _ax.set_xlabel("Importance")
    _ax.grid(True, axis="x", alpha=0.3)
    _fig.savefig("outputs/it2/model_feature_importances.png", bbox_inches="tight")
    _fig
    return


@app.cell
def _(
    RandomForestClassifier,
    confusion_matrix,
    feat_b,
    plt,
    test_df,
    train_df,
    y_test,
    y_train,
):
    _rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    _rf.fit(train_df[feat_b], y_train)
    _preds = _rf.predict(test_df[feat_b])

    _cm = confusion_matrix(y_test, _preds)

    _fig, _ax = plt.subplots(figsize=(5, 4))
    _cax = _ax.matshow(_cm, cmap="Blues", alpha=0.8)
    _fig.colorbar(_cax)

    for _i in range(_cm.shape[0]):
        for _j in range(_cm.shape[1]):
            _ax.text(_j, _i, f"{_cm[_i, _j]:,}", ha="center", va="center", color="black", fontsize=12)

    _ax.set_xticks([0, 1], ["On-Time", "Late (>= 5m)"])
    _ax.set_yticks([0, 1], ["On-Time", "Late (>= 5m)"])
    _ax.set_xlabel("Predicted Label")
    _ax.set_ylabel("True Label")
    _ax.set_title("Confusion Matrix (Random Forest Balanced + Weather)", pad=15)
    _fig.savefig("outputs/it2/model_confusion_matrix.png", bbox_inches="tight")
    _fig
    return


@app.cell
def _(
    RandomForestClassifier,
    feat_b,
    np,
    plt,
    precision_score,
    recall_score,
    test_df,
    train_df,
    y_test,
    y_train,
):
    _rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    _rf.fit(train_df[feat_b], y_train)
    _probs = _rf.predict_proba(test_df[feat_b])[:, 1]

    _thresholds = np.linspace(0.10, 0.60, 26)
    _recalls = []
    _precisions = []
    _costs = []

    _c_fn = 2500
    _c_fp = 350
    _c_tp = 350

    for _th in _thresholds:
        _preds_th = (_probs >= _th).astype(int)
        _rec = float(recall_score(y_test, _preds_th, zero_division=0))
        _prec = float(precision_score(y_test, _preds_th, zero_division=0))
        _recalls.append(_rec)
        _precisions.append(_prec)

        _tp = int(np.sum((y_test == 1) & (_preds_th == 1)))
        _fp = int(np.sum((y_test == 0) & (_preds_th == 1)))
        _fn = int(np.sum((y_test == 1) & (_preds_th == 0)))
        _avg_cost = (_tp * _c_tp + _fp * _c_fp + _fn * _c_fn) / len(y_test)
        _costs.append(_avg_cost)

    _fig, _ax1 = plt.subplots(figsize=(8, 4))
    _ax2 = _ax1.twinx()

    _l1 = _ax1.plot(_thresholds, _recalls, "b-", label="Recall")
    _l2 = _ax1.plot(_thresholds, _precisions, "g--", label="Precision")
    _l3 = _ax2.plot(_thresholds, _costs, "r-.", label="Expected Cost (€/train)")

    _ax1.axvline(0.20, color="gray", linestyle=":", label="Optimal Threshold (0.20)")
    _ax1.set_xlabel("Decision Threshold")
    _ax1.set_ylabel("Metric Score")
    _ax2.set_ylabel("Expected Cost (€/train)", color="red")
    _ax1.set_title("Decision Threshold Trade-off (Expected Value Framework)")

    _lines = _l1 + _l2 + _l3
    _labels = [l.get_label() for l in _lines]
    _ax1.legend(_lines, _labels, loc="upper left")
    _ax1.grid(True, alpha=0.3)

    _fig.savefig("outputs/it2/model_threshold_curve.png", bbox_inches="tight")
    _fig
    return


if __name__ == "__main__":
    app.run()
