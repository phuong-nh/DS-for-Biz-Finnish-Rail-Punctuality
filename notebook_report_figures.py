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
def _(os, pd, plt):
    os.makedirs("report/figures/panels", exist_ok=True)
    _boot = pd.read_csv("outputs/rq/bootstrap_weather_gain.csv")
    _boot = _boot[_boot["resampling"] == "by day"].iloc[::-1]
    # short labels so they stay readable at half page width
    _short = {
        "pre-departure Set B vs A (destination weather)": "B: destination",
        "pre-departure Set C vs A (origin weather)": "C: origin",
        "pre-departure Set D vs A (all weather)": "D: all weather",
        "en-route 0%: E2 vs E1 (weather now + ahead)": "En-route 0%",
        "en-route 25%: E2 vs E1 (weather now + ahead)": "En-route 25%",
        "en-route 50%: E2 vs E1 (weather now + ahead)": "En-route 50%",
        "en-route 75%: E2 vs E1 (weather now + ahead)": "En-route 75%",
    }
    _labels = [_short[c] for c in _boot["comparison"]]
    _roll = pd.read_csv("outputs/rq/rolling_weather_gain.csv")

    with plt.rc_context({"font.size": 13, "axes.titlesize": 13, "legend.fontsize": 10.5}):
        # (a) bootstrap intervals on 2024
        _fa, _a = plt.subplots(figsize=(5.2, 4.2))
        _a.errorbar(
            _boot["point dAUC"],
            range(len(_boot)),
            xerr=[_boot["point dAUC"] - _boot["ci_low"], _boot["ci_high"] - _boot["point dAUC"]],
            fmt="o",
            color="firebrick",
            capsize=4,
            markersize=6,
        )
        _a.axvline(0, color="black", linewidth=1)
        _a.axhline(3.5, color="gray", linewidth=0.8, linestyle=":")
        _a.set_yticks(range(len(_boot)), _labels)
        _a.set_xlabel("AUC gain from weather (2024)")
        _a.grid(True, axis="x", alpha=0.3)
        _fa.savefig("report/figures/panels/weather_gain_bootstrap.pdf", bbox_inches="tight")
        _fa.savefig("report/figures/panels/weather_gain_bootstrap.png", bbox_inches="tight", dpi=200)

        # (b) expanding window by test year
        _fb, _b = plt.subplots(figsize=(5.2, 4.2))
        for _model, _style in [("Gradient Boosting (HistGB, Balanced)", "-"), ("Logistic Regression (Balanced)", "--")]:
            _m = _roll[_roll["Model"] == _model]
            _name = "GB" if "Gradient" in _model else "LR"
            _b.plot(_m["test year"], _m["dAUC C-A (origin weather)"], _style, marker="o", color="royalblue", label=f"{_name}: C $-$ A")
            _b.plot(_m["test year"], _m["dAUC D-A (all weather)"], _style, marker="o", color="firebrick", label=f"{_name}: D $-$ A")
        _b.axhline(0, color="black", linewidth=1)
        _b.set_xticks([2021, 2022, 2023, 2024])
        _b.set_xlabel("test year")
        _b.set_ylabel("AUC gain from weather")
        _b.legend(ncol=2, loc="lower right")
        _b.grid(True, alpha=0.3)
        _fb.savefig("report/figures/panels/weather_gain_rolling.pdf", bbox_inches="tight")
        _fb.savefig("report/figures/panels/weather_gain_rolling.png", bbox_inches="tight", dpi=200)
    plt.close(_fa)
    _fb
    return


@app.cell
def _(pd):
    _raw = pd.read_csv("data/extracted_v2_combined.csv")
    _ran = _raw[~_raw["cancelled"]]
    arrived = _ran[_ran["reached_destination"].isin([True]) & _ran["delay_minutes"].notna() & (_ran["timetable_type"] == "REGULAR")].copy()
    arrived["is_early"] = (arrived["delay_minutes"] < 0).astype(int)

    _vars = [
        ("Target", "Arrival delay at final station (min)", "delay_minutes"),
        ("Target", "Late, >= 5 min (share)", "is_late"),
        ("Target", "Early, < 0 min (share)", "is_early"),
        ("Journey", "Scheduled duration (min)", "scheduled_duration_min"),
        ("Journey", "Commercial stops", "n_commercial_stops"),
        ("Calendar", "Departure hour (local)", "departure_hour_local"),
        ("Weather, origin", "Temperature (°C)", "origin_temp"),
        ("Weather, origin", "Snow depth (cm)", "origin_snow_depth"),
        ("Weather, origin", "Wind speed (m/s)", "origin_wind"),
        ("Weather, origin", "Precipitation intensity (mm/h)", "origin_precip_intensity"),
        ("Weather, destination", "Temperature (°C)", "dest_temp"),
        ("Weather, destination", "Snow depth (cm)", "dest_snow_depth"),
        ("Weather, route", "Minimum temperature (°C)", "route_min_temp"),
        ("Weather, route", "Maximum snow depth (cm)", "route_max_snow_depth"),
        ("Weather, route", "Maximum gust (m/s)", "route_max_gust"),
    ]
    _rows = []
    for _group, _label, _col in _vars:
        _x = arrived[_col]
        _rows.append(
            {
                "Group": _group,
                "Variable": _label,
                "N": int(_x.notna().sum()),
                "Mean": round(float(_x.mean()), 2),
                "SD": round(float(_x.std()), 2),
                "Min": round(float(_x.min()), 1),
                "P5": round(float(_x.quantile(0.05)), 1),
                "Median": round(float(_x.median()), 1),
                "P95": round(float(_x.quantile(0.95)), 1),
                "Max": round(float(_x.max()), 1),
                "Missing %": round(float(100 * _x.isna().mean()), 1),
            }
        )
    summary_table = pd.DataFrame(_rows)
    summary_table.to_csv("report/figures/table_summary_stats.csv", index=False)

    # markdown version to paste into the report
    _lines = ["| " + " | ".join(summary_table.columns) + " |", "|" + "---|" * len(summary_table.columns)]
    for _, _r in summary_table.iterrows():
        _lines.append("| " + " | ".join(f"{v:,}" if isinstance(v, int) else str(v) for v in _r.values) + " |")
    with open("report/figures/table_summary_stats.md", "w") as _f:
        _f.write("\n".join(_lines) + "\n")
    print("regular arrivals:", len(arrived))
    summary_table
    return arrived, summary_table


@app.cell
def _(arrived, os, pd, plt):
    os.makedirs("report/figures/panels", exist_ok=True)

    def _delay_hist(ax):
        ax.hist(arrived["delay_minutes"].clip(-10, 60), bins=70, color="steelblue", edgecolor="white")
        ax.axvline(5, color="red", linestyle="--", label="late threshold (5 min)")
        ax.set_xlabel("delay at final station (min), clipped to -10...60")
        ax.set_ylabel("trains")
        ax.legend(fontsize=8)

    def _by_year(ax):
        _arr = arrived.assign(year=arrived["departure_date"].str[:4].astype(int))
        _year = _arr.groupby("year")["is_late"].mean() * 100
        _bars = ax.bar(_year.index.astype(str), _year.values, color=["steelblue"] * 6 + ["firebrick"])
        for _b in _bars:
            ax.annotate(f"{_b.get_height():.1f}", (_b.get_x() + _b.get_width() / 2, _b.get_height()), ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("year (red = test year)")
        ax.set_ylabel("% late (>= 5 min)")

    def _by_month(ax):
        _m = pd.read_csv("outputs/data_v2/eda_lateness_by_month.csv")
        _m["month"] = _m["year_month"].str[5:7].astype(int)
        _g = _m.groupby("month")["is_late"].agg(["mean", "min", "max"])
        ax.fill_between(_g.index, _g["min"], _g["max"], color="steelblue", alpha=0.2, label="min-max over years")
        ax.plot(_g.index, _g["mean"], marker="o", color="steelblue", label="mean over years")
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("month")
        ax.set_ylabel("% late (>= 5 min)")
        ax.legend(fontsize=8)

    def _by_duration(ax):
        _d = pd.read_csv("outputs/data_v2/eda_lateness_by_duration.csv")
        ax.bar(_d["dur_bin"], _d["late_rate"], color="olivedrab")
        ax.set_xlabel("scheduled trip duration")
        ax.set_ylabel("% late (>= 5 min)")

    def _by_temp(ax):
        _t = pd.read_csv("outputs/data_v2/eda_lateness_by_temp_source.csv")
        _x = range(len(_t))
        for _k, (_col, _color) in enumerate([("origin (departure)", "skyblue"), ("destination (arrival)", "teal"), ("route minimum", "navy")]):
            ax.bar([i + (_k - 1) * 0.27 for i in _x], _t[_col], width=0.27, color=_color, label=_col)
        ax.set_xticks(list(_x), _t["temp_bin"], fontsize=8)
        ax.set_xlabel("temperature band")
        ax.set_ylabel("% late (>= 5 min)")
        ax.legend(fontsize=8)

    def _cause_by_temp(ax):
        _c = pd.read_csv("outputs/data_v2/eda_cause_minutes_by_temp.csv")
        _bottom = pd.Series(0.0, index=_c.index)
        for _grp, _color in zip(["infra", "operator", "traffic", "other"], ["firebrick", "darkorange", "steelblue", "gray"]):
            ax.bar(_c["temp_bin"], _c[_grp], bottom=_bottom, color=_color, label=_grp)
            _bottom = _bottom + _c[_grp]
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("route minimum temperature")
        ax.set_ylabel("% of delay minutes (late trains)")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)

    _panels = [
        ("delay_distribution", "(a) Arrival delay", _delay_hist),
        ("late_by_year", "(b) Late rate by year", _by_year),
        ("late_by_month", "(c) Late rate by calendar month", _by_month),
        ("late_by_duration", "(d) Late rate by trip duration", _by_duration),
        ("late_by_temperature", "(e) Late rate by temperature", _by_temp),
        ("cause_by_temperature", "(f) Cause of delay minutes by temperature", _cause_by_temp),
    ]

    # one file per panel (for \figpair in latex), pdf for sharp text
    for _name, _title, _draw in _panels:
        _f, _a = plt.subplots(figsize=(6, 4))
        _draw(_a)
        _a.grid(True, axis="y", alpha=0.3)
        _f.savefig(f"report/figures/panels/{_name}.pdf", bbox_inches="tight")
        _f.savefig(f"report/figures/panels/{_name}.png", bbox_inches="tight", dpi=150)
        plt.close(_f)

    _fig, _axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for _ax, (_name, _title, _draw) in zip(_axes.flat, _panels):
        _draw(_ax)
        _ax.set_title(_title)
        _ax.grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()
    _fig.savefig("report/figures/fig_summary_stats.png", bbox_inches="tight", dpi=150)
    _fig.savefig("report/figures/fig_summary_stats.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _(pd, summary_table):
    # latex version of the summary table, same layout as the report template
    def _num(v, digits):
        if pd.isna(v):
            return "--"
        _s = f"{v:,.{digits}f}"
        return f"${_s}$" if v < 0 else _s

    _lines = [
        "\\begin{table}[H]",
        "  \\centering",
        (
            "  \\caption{Summary statistics of the modelling population (509{,}110 regular trains that reached their destination, 2018--2024). "
            "Weather values are after filling from the backup station; Missing is what remains missing. "
            "Snow depth $-1$ is the FMI code for snow-free ground.}"
        ),
        "  \\label{tab:summary}",
        "  \\small",
        "  \\begin{tabular}{lrrrrrr}",
        "    \\toprule",
        "    Variable & Mean & SD & Min & Median & Max & Missing (\\%) \\\\",
        "    \\midrule",
    ]
    for _group, _g in summary_table.groupby("Group", sort=False):
        _lines.append(f"    \\multicolumn{{7}}{{l}}{{\\textit{{{_group}}}}} \\\\")
        for _, _r in _g.iterrows():
            _label = _r["Variable"].replace("°C", "$^\\circ$C").replace(">= 5", "$\\ge 5$").replace("< 0", "$< 0$")
            _is_share = "share" in _r["Variable"]
            _cells = [
                _num(_r["Mean"], 2 if _is_share else 1),
                "--" if _is_share else _num(_r["SD"], 1),
                "--" if _is_share else _num(_r["Min"], 1),
                "--" if _is_share else _num(_r["Median"], 1),
                "--" if _is_share else _num(_r["Max"], 1),
                _num(_r["Missing %"], 1),
            ]
            _lines.append(f"    \\quad {_label} & " + " & ".join(_cells) + " \\\\")
    _lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    with open("report/figures/table_summary_stats.tex", "w") as _f:
        _f.write("\n".join(_lines) + "\n")
    print("\n".join(_lines))
    return



@app.cell
def _(os, pd, plt):
    import numpy as np

    os.makedirs("report/figures/panels", exist_ok=True)
    _style = {"font.size": 13, "axes.titlesize": 13, "legend.fontsize": 10}

    # en-route: AUC and MAE by checkpoint (outputs/enroute_v2)
    _clf = pd.read_csv("outputs/enroute_v2/classification_by_checkpoint.csv")
    _reg = pd.read_csv("outputs/enroute_v2/regression_by_checkpoint.csv")
    with plt.rc_context(_style):
        _f, _a = plt.subplots(figsize=(5.2, 4.2))
        for _name, _set, _color, _ls, _label in [
            ("Gradient Boosting (HistGB, tuned)", "E0", "gray", "-", "current delay only"),
            ("Gradient Boosting (HistGB, tuned)", "E1", "royalblue", "-", "+ context"),
            ("Gradient Boosting (HistGB, tuned)", "E2", "firebrick", "-", "+ weather"),
        ]:
            _r = _clf[(_clf["Model"] == _name) & (_clf["Set"] == _set)]
            _a.plot(_r["checkpoint_%"], _r["ROC-AUC"], _ls, marker="o", color=_color, label=_label)
        _a.axhline(0.701, color="black", linestyle=":", label="pre-departure best")
        _a.set_xticks([0, 25, 50, 75])
        _a.set_xlabel("progress through trip (%)")
        _a.set_ylabel("ROC-AUC")
        _a.legend(loc="lower right")
        _a.grid(True, alpha=0.3)
        _f.savefig("report/figures/panels/enroute_auc.pdf", bbox_inches="tight")
        plt.close(_f)

        _f, _a = plt.subplots(figsize=(5.2, 4.2))
        for _name, _set, _color, _ls, _label in [
            ("Persistence (final = current)", "-", "black", ":", "persistence"),
            ("Linear Regression (OLS)", "E2", "royalblue", "--", "OLS"),
            ("Gradient Boosting (HistGB, squared loss)", "E2", "firebrick", "-", "GB, squared loss"),
            ("Gradient Boosting (HistGB, absolute loss)", "E2", "darkgreen", "-", "GB, absolute loss"),
        ]:
            _r = _reg[(_reg["Model"] == _name) & (_reg["Set"] == _set)]
            _a.plot(_r["checkpoint_%"], _r["MAE"], _ls, marker="o", color=_color, label=_label)
        _a.set_xticks([0, 25, 50, 75])
        _a.set_xlabel("progress through trip (%)")
        _a.set_ylabel("MAE of final delay (min)")
        _a.legend(loc="lower left")
        _a.grid(True, alpha=0.3)
        _f.savefig("report/figures/panels/enroute_mae.pdf", bbox_inches="tight")
        plt.close(_f)

    # decision analysis: profit curve at r = 5 and sensitivity to r (outputs/rq, outputs/it5)
    _prior = 0.1808  # 2018-2023 late rate, see outputs/it6/cause_models.csv
    _pre = pd.read_csv("outputs/it5/test_probabilities.csv")
    _en = pd.read_csv("outputs/enroute/test_probabilities.csv")
    _en50 = _en[_en["checkpoint_%"] == 50]
    _scores = {
        "Set A (no weather)": (_pre["is_late"].to_numpy(), _pre["prob_set_A"].to_numpy(), "gray"),
        "Set C (origin weather)": (_pre["is_late"].to_numpy(), _pre["prob_set_C"].to_numpy(), "royalblue"),
        "En-route, 50%": (_en50["is_late"].to_numpy(), _en50["prob_E2"].to_numpy(), "darkorange"),
    }
    _ev = pd.read_csv("outputs/rq/ev_cost_by_ratio.csv")
    _ev_names = {"Set A (no weather)": "pre-departure Set A", "Set C (origin weather)": "pre-departure Set C", "En-route, 50%": "en-route 50% E2"}
    with plt.rc_context(_style):
        _f, _a = plt.subplots(figsize=(5.2, 4.2))
        _r = 5
        for _label, (_y, _p, _color) in _scores.items():
            _order = np.argsort(-_p)
            _tp = np.cumsum(_y[_order])
            _k = np.arange(1, len(_y) + 1)
            _a.plot(_k / len(_y) * 100, (_r * _tp - _k) / len(_y) * 1000, color=_color, label=_label)
        _a.axhline(0, color="black", linewidth=1)
        _a.set_xlabel("trains flagged, highest risk first (%)")
        _a.set_ylabel("saving vs never act (per 1,000 trains)")
        _a.legend(loc="lower center")
        _a.grid(True, alpha=0.3)
        _f.savefig("report/figures/panels/ev_profit_curve.pdf", bbox_inches="tight")
        plt.close(_f)

        _f, _a = plt.subplots(figsize=(5.2, 4.2))
        for _label, (_y, _p, _color) in _scores.items():
            _e = _ev[_ev["score"] == _ev_names[_label]]
            _a.plot(_e["cost ratio r (miss / action)"], _e["saving vs best simple rule per 1000 trains"], marker="o", color=_color, label=_label)
        _a.axhline(0, color="black", linewidth=1)
        _a.set_xlabel("cost ratio $r$")
        _a.set_ylabel("saving vs best naive policy (per 1,000)")
        _a.legend(loc="upper right")
        _a.grid(True, alpha=0.3)
        _f.savefig("report/figures/panels/ev_sensitivity.pdf", bbox_inches="tight")
        _f
    return


if __name__ == "__main__":
    app.run()
