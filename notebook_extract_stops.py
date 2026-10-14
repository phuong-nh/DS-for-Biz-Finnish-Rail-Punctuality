import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from dsforbiz_finnish_rail_punctuality.extract_stops import extract_months

    return extract_months, os, pd, roc_auc_score


@app.cell
def _(extract_months, os, pd):
    os.makedirs("outputs/enroute", exist_ok=True)
    _combined_path = "data/extracted_stops_combined.csv"
    if os.path.exists(_combined_path):
        stops = pd.read_csv(_combined_path)
    else:
        _all_months = [f"{y}_{m:02d}" for y in range(2018, 2025) for m in range(1, 13)]
        stops = extract_months(_all_months, output_path=_combined_path, max_workers=6)
    print(stops.shape)
    return (stops,)


@app.cell
def _(pd, roc_auc_score, stops):
    _reg = stops[stops["timetable_type"] == "REGULAR"]
    _rows = []
    for _cp, _g in _reg.groupby("checkpoint"):
        _rows.append(
            {
                "checkpoint_%": _cp,
                "rows": len(_g),
                "trains": _g[["train_number", "departure_date"]].drop_duplicates().shape[0],
                "late_rate_%": round(100 * _g["is_late"].mean(), 2),
                "median_progress": round(_g["progress"].median(), 3),
                "median_remaining_min": round(_g["remaining_min"].median(), 1),
                "mean_current_delay": round(_g["current_delay"].mean(), 2),
                "corr_current_vs_final": round(_g["current_delay"].corr(_g["final_delay"]), 3),
                "AUC_current_delay_only": round(float(roc_auc_score(_g["is_late"], _g["current_delay"])), 4),
            }
        )
    _overview = pd.DataFrame(_rows)
    _overview.to_csv("outputs/enroute/overview_checkpoints.csv", index=False)

    _missing = (100 * _reg.isna().mean()).round(2).rename("missing_%")
    _missing[_missing > 0].to_csv("outputs/enroute/missing_columns.csv")
    print("adhoc rows dropped:", int((stops["timetable_type"] != "REGULAR").sum()))
    _overview
    return


if __name__ == "__main__":
    app.run()
