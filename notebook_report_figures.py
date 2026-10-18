import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import shutil

    import matplotlib.pyplot as plt
    import pandas as pd

    return os, pd, plt, shutil


@app.cell
def _(os, shutil):
    os.makedirs("report/figures", exist_ok=True)
    _copies = {
        "outputs/data_v2/eda_lateness_by_month.png": "fig1_lateness_by_month.png",
        "outputs/data_v2/eda_lateness_by_temp_source.png": "fig2_lateness_by_temperature.png",
        "outputs/it5/bias_variance_depth.png": "fig3_bias_variance.png",
        "outputs/enroute_v2/metrics_by_checkpoint.png": "fig5_enroute.png",
        "outputs/rq/ev_profit_and_sensitivity.png": "fig6_expected_value.png",
        "outputs/it4b/regression_residuals.png": "figA1_residuals.png",
        "outputs/it5/tree_rules.png": "figA2_tree_rules.png",
    }
    for _src, _dst in _copies.items():
        shutil.copy(_src, f"report/figures/{_dst}")
    print("copied", len(_copies), "figures")
    return


@app.cell
def _(pd, plt):
    _boot = pd.read_csv("outputs/rq/bootstrap_weather_gain.csv")
    _boot = _boot[_boot["resampling"] == "by day"].iloc[::-1]
    _labels = [
        c.replace("pre-departure ", "Pre: ").replace(" (weather now + ahead)", "").replace("E2 vs E1", "weather")
        for c in _boot["comparison"]
    ]
    _roll = pd.read_csv("outputs/rq/rolling_weather_gain.csv")

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.2))
    _axes[0].errorbar(
        _boot["point dAUC"],
        range(len(_boot)),
        xerr=[_boot["point dAUC"] - _boot["ci_low"], _boot["ci_high"] - _boot["point dAUC"]],
        fmt="o",
        color="firebrick",
        capsize=4,
    )
    _axes[0].axvline(0, color="black", linewidth=1)
    _axes[0].set_yticks(range(len(_boot)), _labels, fontsize=8)
    _axes[0].set_xlabel("AUC gain from weather, 2024 test (95% day-block bootstrap CI)")
    _axes[0].set_title("(a) Significance on the 2024 test year")
    _axes[0].grid(True, axis="x", alpha=0.3)

    for _model, _style in [("Gradient Boosting (HistGB, Balanced)", "-"), ("Logistic Regression (Balanced)", "--")]:
        _m = _roll[_roll["Model"] == _model]
        _short = "HistGB" if "Gradient" in _model else "LR"
        _axes[1].plot(_m["test year"], _m["dAUC C-A (origin weather)"], _style, marker="o", color="royalblue", label=f"{_short}: origin weather (C - A)")
        _axes[1].plot(_m["test year"], _m["dAUC D-A (all weather)"], _style, marker="o", color="firebrick", label=f"{_short}: all weather (D - A)")
    _axes[1].axhline(0, color="black", linewidth=1)
    _axes[1].set_xticks([2021, 2022, 2023, 2024])
    _axes[1].set_xlabel("test year (trained on all earlier years)")
    _axes[1].set_ylabel("AUC gain from weather")
    _axes[1].set_title("(b) Robustness across test years")
    _axes[1].legend(fontsize=7)
    _axes[1].grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig.savefig("report/figures/fig4_weather_gain.png", bbox_inches="tight", dpi=150)
    _fig
    return


if __name__ == "__main__":
    app.run()
