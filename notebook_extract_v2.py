import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import pandas as pd

    from dsforbiz_finnish_rail_punctuality.extract_v2 import extract_months

    return extract_months, os, pd


@app.cell
def _(extract_months, os, pd):
    os.makedirs("outputs/data_v2", exist_ok=True)
    _combined_path = "data/extracted_v2_combined.csv"
    if os.path.exists(_combined_path):
        df = pd.read_csv(_combined_path)
    else:
        _all_months = [f"{y}_{m:02d}" for y in range(2018, 2025) for m in range(1, 13)]
        df = extract_months(_all_months, output_path=_combined_path, max_workers=6)
    df["year"] = df["departure_date"].str[:4].astype(int)
    print(df.shape)
    return (df,)


@app.cell
def _(df, pd):
    _cancelled = df["cancelled"]
    _ran = df[~_cancelled]
    _not_reached = ~_ran["reached_destination"].isin([True])
    _arrived = _ran[~_not_reached & _ran["delay_minutes"].notna()]
    _regular = _arrived[_arrived["timetable_type"] == "REGULAR"]

    _overview = pd.DataFrame(
        {
            "Total Scheduled": [len(df)],
            "Cancelled": [int(_cancelled.sum())],
            "Not Reached Destination": [int(_not_reached.sum())],
            "Arrived": [len(_arrived)],
            "Arrived ADHOC": [int((_arrived["timetable_type"] == "ADHOC").sum())],
            "Arrived REGULAR": [len(_regular)],
            "Late REGULAR (>= 5m)": [int(_regular["is_late"].sum())],
            "Late Rate REGULAR (%)": [round(100 * _regular["is_late"].mean(), 2)],
        }
    )
    _overview.to_csv("outputs/data_v2/overview.csv", index=False)
    _overview.T
    return


@app.cell
def _(df):
    _d = df.copy()
    _d["not_reached"] = ~_d["cancelled"] & ~_d["reached_destination"].isin([True])
    _d["adhoc"] = _d["timetable_type"] == "ADHOC"
    _by_year = _d.groupby("year").agg(
        trains=("train_number", "size"),
        cancelled=("cancelled", "sum"),
        not_reached=("not_reached", "sum"),
        adhoc=("adhoc", "sum"),
    )
    _reg = _d[~_d["adhoc"] & _d["is_late"].notna()]
    _by_year["late_rate_regular_%"] = (100 * _reg.groupby("year")["is_late"].mean()).round(2)
    _by_year.to_csv("outputs/data_v2/overview_by_year.csv")
    _by_year
    return


if __name__ == "__main__":
    app.run()
