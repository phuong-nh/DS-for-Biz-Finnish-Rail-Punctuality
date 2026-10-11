import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import matplotlib.pyplot as plt
    import pandas as pd

    from dsforbiz_finnish_rail_punctuality.extract_v2 import extract_months

    return extract_months, os, pd, plt


@app.cell
def _(extract_months, os, pd):
    os.makedirs("outputs/data_v2", exist_ok=True)
    combined_path = "data/extracted_v2_combined.csv"
    if os.path.exists(combined_path):
        df = pd.read_csv(combined_path)
    else:
        all_months = [f"{y}_{m:02d}" for y in range(2018, 2025) for m in range(1, 13)]
        df = extract_months(all_months, output_path=combined_path, max_workers=6)

    df["year_month"] = df["departure_date"].str[:7]
    df["month"] = df["departure_date"].str[5:7].astype(int)
    df
    return (df,)


@app.cell
def _(df):
    _ran = df[~df["cancelled"]]
    arrived = _ran[_ran["reached_destination"].isin([True]) & _ran["delay_minutes"].notna()].copy()
    for _c in [c for c in arrived.columns if c.startswith("cause_") and not c.endswith(("_minutes", "_dominant"))]:
        arrived[_c] = arrived[_c].astype(bool)
    for _c in [c for c in arrived.columns if c.endswith("_fallback")]:
        arrived[_c] = arrived[_c].astype(bool)
    regular = arrived[arrived["timetable_type"] == "REGULAR"].copy()
    print(len(arrived), "arrived,", len(regular), "regular")
    return arrived, regular


@app.cell
def _(plt, regular):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _delays = regular["delay_minutes"].clip(lower=-10, upper=60)
    _ax.hist(_delays, bins=35, edgecolor="black", alpha=0.7)
    _ax.axvline(5, color="red", linestyle="--", label="Late threshold (5 min)")
    _ax.set_title("Arrival Delay Distribution at Destination (regular trains)")
    _ax.set_xlabel("Delay in Minutes (clipped between -10 and 60)")
    _ax.set_ylabel("Number of Trains")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _fig.savefig("outputs/data_v2/eda_delay_distribution.png", bbox_inches="tight")
    _fig
    return


@app.cell
def _(arrived, pd):
    _desc = arrived.groupby("timetable_type")["delay_minutes"].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
    _desc["late_rate_%"] = arrived.groupby("timetable_type")["is_late"].mean() * 100
    _desc["median_train_number"] = arrived.groupby("timetable_type")["train_number"].median()
    _desc = _desc.round(2)
    _desc.to_csv("outputs/data_v2/eda_adhoc_vs_regular.csv")

    _overall = pd.DataFrame(
        {"regular trains": arrived[arrived["timetable_type"] == "REGULAR"]["delay_minutes"].describe(percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999])}
    ).round(2)
    _overall.to_csv("outputs/data_v2/eda_delay_describe.csv")
    _desc
    return


@app.cell
def _(plt, regular):
    _month_stats = regular.groupby("year_month")["is_late"].mean() * 100

    _fig, _ax = plt.subplots(figsize=(14, 4))
    _ax.bar(_month_stats.index, _month_stats.values, color="steelblue", edgecolor="black", width=0.8)
    _ax.set_title("Lateness Rate (>= 5 min) by Month (2018–2024, regular trains)")
    _ax.set_ylabel("% Delayed")
    _ax.tick_params(axis="x", rotation=90)
    _ax.grid(True, axis="y", alpha=0.3)
    _fig.savefig("outputs/data_v2/eda_lateness_by_month.png", bbox_inches="tight")
    _month_stats.round(2).to_csv("outputs/data_v2/eda_lateness_by_month.csv")
    _fig
    return


@app.cell
def _(plt, regular):
    _dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    _dow_stats = regular.groupby("day_of_week")["is_late"].mean() * 100
    _labels = [_dow_names[int(i)] for i in _dow_stats.index]

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _bars = _ax.bar(_labels, _dow_stats.values, color="cornflowerblue", edgecolor="black")
    _ax.set_title("Lateness Rate by Day of Week")
    _ax.set_ylabel("% Delayed")
    _ax.grid(True, axis="y", alpha=0.3)
    for _bar in _bars:
        _h = _bar.get_height()
        _ax.annotate(f"{_h:.1f}%", xy=(_bar.get_x() + _bar.get_width() / 2, _h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
    _fig.savefig("outputs/data_v2/eda_lateness_by_dow.png", bbox_inches="tight")
    _dow_stats.round(2).to_csv("outputs/data_v2/eda_lateness_by_dow.csv")
    _fig
    return


@app.cell
def _(plt, regular):
    _hour = regular.groupby("departure_hour_local").agg(late_rate=("is_late", "mean"), n=("is_late", "size"))
    _hour["late_rate"] = (_hour["late_rate"] * 100).round(2)
    _hour.to_csv("outputs/data_v2/eda_lateness_by_local_hour.csv")

    _fig, _ax = plt.subplots(figsize=(10, 4))
    _ax.bar(_hour.index.astype(int), _hour["late_rate"], color="darkorange", edgecolor="black")
    _ax.set_title("Lateness Rate by Local Departure Hour (Europe/Helsinki)")
    _ax.set_xlabel("Departure hour (local)")
    _ax.set_ylabel("% Delayed")
    _ax.set_xticks(range(24))
    _ax.grid(True, axis="y", alpha=0.3)
    _fig.savefig("outputs/data_v2/eda_lateness_by_local_hour.png", bbox_inches="tight")
    _hour
    return


@app.cell
def _(pd, plt, regular):
    _bins = [-50, -15, -5, 5, 15, 40]
    _bin_labels = ["< -15°C", "-15 to -5°C", "-5 to 5°C", "5 to 15°C", "> 15°C"]
    _rows = {}
    for _col, _name in [("origin_temp", "origin (departure)"), ("dest_temp", "destination (arrival)"), ("route_min_temp", "route minimum")]:
        _tmp = regular.dropna(subset=[_col]).copy()
        _tmp["temp_bin"] = pd.cut(_tmp[_col], bins=_bins, labels=_bin_labels)
        _rows[_name] = _tmp.groupby("temp_bin", observed=False)["is_late"].mean() * 100
        _rows[_name + " n"] = _tmp.groupby("temp_bin", observed=False)["is_late"].size()
    _temp_table = pd.DataFrame(_rows).round(2)
    _temp_table.to_csv("outputs/data_v2/eda_lateness_by_temp_source.csv")

    _fig, _ax = plt.subplots(figsize=(9, 4))
    _x = range(len(_bin_labels))
    for _i, (_name, _color) in enumerate([("origin (departure)", "skyblue"), ("destination (arrival)", "teal"), ("route minimum", "navy")]):
        _ax.bar([x + (_i - 1) * 0.27 for x in _x], _temp_table[_name], width=0.27, label=_name, color=_color, edgecolor="black")
    _ax.set_xticks(list(_x), _bin_labels)
    _ax.set_title("Lateness Rate by Temperature: Origin vs Destination vs Route Minimum")
    _ax.set_ylabel("% Delayed")
    _ax.legend()
    _ax.grid(True, axis="y", alpha=0.3)
    _fig.savefig("outputs/data_v2/eda_lateness_by_temp_source.png", bbox_inches="tight")
    _temp_table
    return


@app.cell
def _(pd, plt, regular):
    _bins = [-1, 0, 10, 30, 60, 300]
    _bin_labels = ["0 cm", "1-10 cm", "11-30 cm", "31-60 cm", "> 60 cm"]
    _rows = {}
    for _col, _name in [("dest_snow_depth", "destination"), ("route_max_snow_depth", "route maximum")]:
        _tmp = regular.dropna(subset=[_col]).copy()
        _tmp["snow_bin"] = pd.cut(_tmp[_col], bins=_bins, labels=_bin_labels)
        _rows[_name] = _tmp.groupby("snow_bin", observed=False)["is_late"].mean() * 100
        _rows[_name + " n"] = _tmp.groupby("snow_bin", observed=False)["is_late"].size()
    _snow_table = pd.DataFrame(_rows).round(2)
    _snow_table.to_csv("outputs/data_v2/eda_lateness_by_snow.csv")

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _x = range(len(_bin_labels))
    _ax.bar([x - 0.2 for x in _x], _snow_table["destination"], width=0.4, label="destination", color="lightsteelblue", edgecolor="black")
    _ax.bar([x + 0.2 for x in _x], _snow_table["route maximum"], width=0.4, label="route maximum", color="slategray", edgecolor="black")
    _ax.set_xticks(list(_x), _bin_labels)
    _ax.set_title("Lateness Rate by Snow Depth")
    _ax.set_ylabel("% Delayed")
    _ax.legend()
    _ax.grid(True, axis="y", alpha=0.3)
    _fig.savefig("outputs/data_v2/eda_lateness_by_snow.png", bbox_inches="tight")
    _snow_table
    return


@app.cell
def _(arrived, pd):
    _rows = []
    for _where in ["origin", "dest"]:
        for _v in ["temp", "wind", "gust", "precip_intensity", "snow_depth", "visibility", "humidity", "cloud", "present_weather"]:
            _col = f"{_where}_{_v}"
            _missing_after = arrived[_col].isna().mean() * 100
            _fallback = arrived[_col + "_fallback"].mean() * 100 if (_col + "_fallback") in arrived.columns else 0.0
            _rows.append(
                {
                    "variable": _col,
                    "main_station_missing_%": round(_missing_after + _fallback, 2),
                    "filled_from_other_station_%": round(_fallback, 2),
                    "still_missing_%": round(_missing_after, 2),
                }
            )
    for _col in [c for c in arrived.columns if c.startswith("route_")]:
        _rows.append({"variable": _col, "main_station_missing_%": None, "filled_from_other_station_%": None, "still_missing_%": round(arrived[_col].isna().mean() * 100, 2)})
    _miss = pd.DataFrame(_rows)
    _miss.to_csv("outputs/data_v2/eda_weather_missingness_full.csv", index=False)
    _miss
    return


@app.cell
def _(pd, plt, regular):
    _bins = [0, 60, 120, 180, 300, 480, 1500]
    _bin_labels = ["< 1h", "1-2h", "2-3h", "3-5h", "5-8h", "> 8h"]
    _tmp = regular.dropna(subset=["scheduled_duration_min"]).copy()
    _tmp["dur_bin"] = pd.cut(_tmp["scheduled_duration_min"], bins=_bins, labels=_bin_labels)
    _dur = _tmp.groupby("dur_bin", observed=False).agg(
        n=("is_late", "size"),
        late_rate=("is_late", "mean"),
        mean_delay=("delay_minutes", "mean"),
        median_stops=("n_commercial_stops", "median"),
    )
    _dur["late_rate"] = _dur["late_rate"] * 100
    _dur = _dur.round(2)
    _dur.to_csv("outputs/data_v2/eda_lateness_by_duration.csv")

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.bar(_dur.index.astype(str), _dur["late_rate"], color="olivedrab", edgecolor="black")
    _ax.set_title("Lateness Rate by Scheduled Trip Duration")
    _ax.set_ylabel("% Delayed")
    _ax.grid(True, axis="y", alpha=0.3)
    _fig.savefig("outputs/data_v2/eda_lateness_by_duration.png", bbox_inches="tight")
    print("corr(duration, delay):", round(regular["scheduled_duration_min"].corr(regular["delay_minutes"]), 3))
    print("corr(stops, delay):", round(regular["n_commercial_stops"].corr(regular["delay_minutes"]), 3))
    print("corr(dep delay, delay):", round(regular["dep_delay_minutes"].corr(regular["delay_minutes"]), 3))
    _dur
    return


@app.cell
def _(plt, regular):
    _top_corridors = regular["corridor"].value_counts().head(10).index
    _corr_df = regular[regular["corridor"].isin(_top_corridors)]
    _corr_stats = _corr_df.groupby("corridor")["is_late"].mean().loc[_top_corridors] * 100

    _fig, _ax = plt.subplots(figsize=(9, 4.5))
    _bars = _ax.barh(_corr_stats.index, _corr_stats.values, color="mediumpurple", edgecolor="black")
    _ax.set_title("Lateness Rate Across Top 10 Rail Corridors")
    _ax.set_xlabel("% Delayed")
    _ax.grid(True, axis="x", alpha=0.3)
    _ax.invert_yaxis()
    for _bar in _bars:
        _w = _bar.get_width()
        _ax.annotate(f"{_w:.1f}%", xy=(_w, _bar.get_y() + _bar.get_height() / 2), xytext=(5, 0), textcoords="offset points", ha="left", va="center")
    _fig.savefig("outputs/data_v2/eda_top_corridors.png", bbox_inches="tight")
    _corr_stats.round(2).to_csv("outputs/data_v2/eda_top_corridors.csv")
    _fig
    return


@app.cell
def _(pd, plt, regular):
    _groups = ["infra", "operator", "traffic", "other"]
    _late = regular[regular["is_late"] == 1]

    _has = _late.groupby("month")[[f"cause_{g}" for g in _groups]].mean() * 100
    _has.columns = _groups
    _dominant = pd.crosstab(_late["month"], _late["cause_dominant"], normalize="index") * 100
    _minutes = _late.groupby("month")[[f"cause_{g}_minutes" for g in _groups]].sum()
    _minutes = _minutes.div(_minutes.sum(axis=1), axis=0) * 100
    _minutes.columns = _groups

    _has.round(2).to_csv("outputs/data_v2/eda_cause_has_group_by_month.csv")
    _dominant.round(2).to_csv("outputs/data_v2/eda_cause_dominant_by_month.csv")
    _minutes.round(2).to_csv("outputs/data_v2/eda_cause_minutes_share_by_month.csv")

    _fig, _ax = plt.subplots(figsize=(10, 4.5))
    _bottom = pd.Series(0.0, index=_minutes.index)
    for _g, _color in zip(_groups, ["firebrick", "darkorange", "steelblue", "gray"]):
        _ax.bar(_minutes.index, _minutes[_g], bottom=_bottom, label=_g, color=_color, edgecolor="black")
        _bottom = _bottom + _minutes[_g]
    _ax.set_xticks(range(1, 13))
    _ax.set_xlabel("Month")
    _ax.set_ylabel("% of attributed delay minutes")
    _ax.set_title("Delay Minutes by Cause Group and Month (late regular trains)")
    _ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    _fig.savefig("outputs/data_v2/eda_cause_minutes_by_month.png", bbox_inches="tight")
    _fig
    return


@app.cell
def _(pd, plt, regular):
    _groups = ["infra", "operator", "traffic", "other"]
    _late = regular[regular["is_late"] == 1].dropna(subset=["route_min_temp"]).copy()
    _late["temp_bin"] = pd.cut(
        _late["route_min_temp"],
        bins=[-50, -15, -5, 5, 15, 40],
        labels=["< -15°C", "-15 to -5°C", "-5 to 5°C", "5 to 15°C", "> 15°C"],
    )
    _minutes = _late.groupby("temp_bin", observed=False)[[f"cause_{g}_minutes" for g in _groups]].sum()
    _minutes = _minutes.div(_minutes.sum(axis=1), axis=0) * 100
    _minutes.columns = _groups
    _minutes["n_late_trains"] = _late.groupby("temp_bin", observed=False).size()
    _minutes["weather_coded_%"] = _late.groupby("temp_bin", observed=False)["cause_weather_coded"].mean() * 100
    _minutes = _minutes.round(2)
    _minutes.to_csv("outputs/data_v2/eda_cause_minutes_by_temp.csv")

    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    _bottom = pd.Series(0.0, index=_minutes.index)
    for _g, _color in zip(_groups, ["firebrick", "darkorange", "steelblue", "gray"]):
        _ax.bar(_minutes.index.astype(str), _minutes[_g], bottom=_bottom, label=_g, color=_color, edgecolor="black")
        _bottom = _bottom + _minutes[_g]
    _ax.set_xlabel("Route minimum temperature")
    _ax.set_ylabel("% of attributed delay minutes")
    _ax.set_title("Delay Minutes by Cause Group and Route Temperature (late trains)")
    _ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    _fig.savefig("outputs/data_v2/eda_cause_minutes_by_temp.png", bbox_inches="tight")
    _minutes
    return


if __name__ == "__main__":
    app.run()
