import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import matplotlib.pyplot as plt
    import pandas as pd

    from dsforbiz_finnish_rail_punctuality.extract import extract_months

    return extract_months, os, pd, plt


@app.cell
def _(extract_months, os, pd):
    os.makedirs("outputs", exist_ok=True)
    combined_path = "data/extracted_combined.csv"
    if os.path.exists(combined_path):
        df = pd.read_csv(combined_path)
    else:
        all_months = [f"{y}_{m:02d}" for y in range(2018, 2025) for m in range(1, 13)]
        df = extract_months(all_months, output_path=combined_path, max_workers=6)

    df["year_month"] = df["departure_date"].str[:7]
    df
    return (df,)


@app.cell
def _(df, os, pd):
    _total_trips = len(df)
    _canc_trips = int(df["cancelled"].sum())
    non_canc = df[~df["cancelled"]].copy()
    _late_trips = int(non_canc["is_late"].sum())
    _late_pct = round((_late_trips / len(non_canc)) * 100, 2)

    _overview_df = pd.DataFrame(
        [
            {
                "Total Scheduled": _total_trips,
                "Cancelled": _canc_trips,
                "Cancellation Rate (%)": round(_canc_trips / _total_trips * 100, 2),
                "Arrived Trips": len(non_canc),
                "Late Arrivals (>= 5m)": _late_trips,
                "Late Rate (%)": _late_pct,
            }
        ]
    )
    os.makedirs("outputs", exist_ok=True)
    _overview_df.to_csv("outputs/it1/eda_overview.csv", index=False)
    _overview_df
    return (non_canc,)


@app.cell
def _(non_canc, plt):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _delays = non_canc["delay_minutes"].clip(lower=-10, upper=60)
    _ax.hist(_delays, bins=35, edgecolor="black", alpha=0.7)
    _ax.axvline(5, color="red", linestyle="--", label="Late threshold (5 min)")
    _ax.set_title("Arrival Delay Distribution at Destination")
    _ax.set_xlabel("Delay in Minutes (clipped between -10 and 60)")
    _ax.set_ylabel("Number of Trains")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _fig.savefig("outputs/it1/eda_delay_distribution.png", bbox_inches="tight")
    _fig
    return


@app.cell
def _(non_canc, plt):
    _month_stats = non_canc.groupby("year_month")["is_late"].mean() * 100

    _fig, _ax = plt.subplots(figsize=(14, 4))
    _bars = _ax.bar(
        _month_stats.index,
        _month_stats.values,
        color="steelblue",
        edgecolor="black",
        width=0.8,
    )
    _ax.set_title("Lateness Rate (>= 5 min) by Month (2018–2024)")
    _ax.set_ylabel("% Delayed")
    _ax.tick_params(axis="x", rotation=90)
    _ax.grid(True, axis="y", alpha=0.3)
    _fig.savefig("outputs/it1/eda_lateness_by_month.png", bbox_inches="tight")
    _fig
    return


@app.cell
def _(non_canc, plt):
    _dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    _dow_stats = non_canc.groupby("day_of_week")["is_late"].mean() * 100
    _labels = [_dow_names[int(i)] for i in _dow_stats.index]

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _bars = _ax.bar(_labels, _dow_stats.values, color="cornflowerblue", edgecolor="black")
    _ax.set_title("Lateness Rate by Day of Week")
    _ax.set_ylabel("% Delayed")
    _ax.set_ylim(0, 32)
    _ax.grid(True, axis="y", alpha=0.3)

    for _bar in _bars:
        _h = _bar.get_height()
        _ax.annotate(
            f"{_h:.1f}%",
            xy=(_bar.get_x() + _bar.get_width() / 2, _h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    _fig.savefig("outputs/it1/eda_lateness_by_dow.png", bbox_inches="tight")
    _fig
    return


@app.cell
def _(non_canc, pd, plt):
    _weather_df = non_canc.dropna(subset=["dest_temp"]).copy()
    _bins = [-50, -15, -5, 5, 15, 40]
    _bin_labels = ["< -15°C", "-15 to -5°C", "-5 to 5°C", "5 to 15°C", "> 15°C"]
    _weather_df["temp_bin"] = pd.cut(_weather_df["dest_temp"], bins=_bins, labels=_bin_labels)
    _temp_late = _weather_df.groupby("temp_bin", observed=False)["is_late"].mean() * 100

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _bars = _ax.bar(_temp_late.index.astype(str), _temp_late.values, color="teal", edgecolor="black")
    _ax.set_title("Lateness Rate by Destination Temperature")
    _ax.set_ylabel("% Delayed")
    _ax.set_ylim(0, 48)
    _ax.grid(True, axis="y", alpha=0.3)

    for _bar in _bars:
        _h = _bar.get_height()
        _ax.annotate(
            f"{_h:.1f}%",
            xy=(_bar.get_x() + _bar.get_width() / 2, _h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    _fig.savefig("outputs/it1/eda_lateness_by_temp.png", bbox_inches="tight")
    _fig
    return


@app.cell
def _(non_canc, plt):
    _top_corridors = non_canc["corridor"].value_counts().head(8).index
    _corr_df = non_canc[non_canc["corridor"].isin(_top_corridors)]
    _corr_stats = _corr_df.groupby("corridor")["is_late"].mean().loc[_top_corridors] * 100

    _fig, _ax = plt.subplots(figsize=(9, 4))
    _bars = _ax.barh(_corr_stats.index, _corr_stats.values, color="mediumpurple", edgecolor="black")
    _ax.set_title("Lateness Rate Across Top 8 Rail Corridors")
    _ax.set_xlabel("% Delayed")
    _ax.set_xlim(0, 40)
    _ax.grid(True, axis="x", alpha=0.3)
    _ax.invert_yaxis()

    for _bar in _bars:
        _w = _bar.get_width()
        _ax.annotate(
            f"{_w:.1f}%",
            xy=(_w, _bar.get_y() + _bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
        )

    _fig.savefig("outputs/it1/eda_top_corridors.png", bbox_inches="tight")
    _fig
    return


if __name__ == "__main__":
    app.run()
