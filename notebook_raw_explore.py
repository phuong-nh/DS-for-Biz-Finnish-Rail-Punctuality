import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import ast
    import gzip
    import json
    import os
    import re
    from urllib.request import Request, urlopen

    import matplotlib.pyplot as plt
    import pandas as pd

    return Request, ast, gzip, json, os, pd, plt, re, urlopen


@app.cell
def _(ast, os, pd, re):
    os.makedirs("outputs/data_v2", exist_ok=True)

    # timeTableRows is a python literal with bare nan, literal_eval can't read nan
    _nan = re.compile(r"\bnan\b")

    sample_months = ["2018_01", "2021_07", "2024_01"]
    _dfs = []
    for _m in sample_months:
        _df = pd.read_csv(f"data/matched_data_{_m}.csv", nrows=2000)
        _df["month"] = _m
        _df["stops"] = [ast.literal_eval(_nan.sub("None", s)) for s in _df["timeTableRows"]]
        _dfs.append(_df.drop(columns=["timeTableRows"]))
    raw = pd.concat(_dfs, ignore_index=True)
    raw.drop(columns=["stops"]).head()
    return raw, sample_months


@app.cell
def _(pd, raw):
    _rows = []
    for _col in ["trainCategory", "operatorShortCode", "timetableType", "cancelled", "runningCurrently", "trainType"]:
        for _val, _n in raw[_col].value_counts(dropna=False).items():
            _rows.append({"column": _col, "value": _val, "n": _n, "share_%": round(100 * _n / len(raw), 2)})
    _rows.append({"column": "commuterLineID", "value": "non-null", "n": int(raw["commuterLineID"].notna().sum()), "share_%": None})
    _top = pd.DataFrame(_rows)
    _top.to_csv("outputs/data_v2/raw_top_level_counts.csv", index=False)
    _top
    return


@app.cell
def _(pd, raw):
    _all_rows = [r for _stops in raw["stops"] for r in _stops]
    _keys = pd.Series([k for r in _all_rows for k in r]).value_counts()
    _fields = pd.DataFrame({"n_rows_with_field": _keys, "share_%": (100 * _keys / len(_all_rows)).round(2)})
    _fields.to_csv("outputs/data_v2/raw_stop_fields.csv")

    _n_stops = raw["stops"].apply(len)
    _n_commercial = raw["stops"].apply(lambda st: sum(1 for r in st if r["type"] == "ARRIVAL" and r.get("commercialStop")))
    _summary = pd.DataFrame(
        {
            "timetable rows per train": _n_stops.describe(),
            "commercial arrival stops per train": _n_commercial.describe(),
        }
    ).round(1)
    _summary.loc["share of rows trainStopping", "timetable rows per train"] = round(
        sum(1 for r in _all_rows if r.get("trainStopping")) / len(_all_rows), 3
    )
    _summary.to_csv("outputs/data_v2/raw_stops_per_train.csv")
    print(len(_all_rows), "timetable rows in sample")
    _fields
    return


@app.cell
def _(pd, raw):
    _weather_vars = [
        "Air temperature",
        "Wind speed",
        "Gust speed",
        "Relative humidity",
        "Dew-point temperature",
        "Precipitation amount",
        "Precipitation intensity",
        "Snow depth",
        "Pressure (msl)",
        "Horizontal visibility",
        "Cloud amount",
        "Present weather (auto)",
    ]
    _ok = raw[~raw["cancelled"]]
    _rows = []
    for _where, _pos in [("origin (first row)", 0), ("destination (last row)", -1)]:
        _weathers = [st[_pos].get("weather_observations") or {} for st in _ok["stops"]]
        for _v in _weather_vars:
            _main_missing = sum(1 for w in _weathers if w.get(_v) is None)
            _still_missing = sum(1 for w in _weathers if w.get(_v) is None and w.get(_v + " Other") is None)
            _rows.append(
                {
                    "where": _where,
                    "variable": _v,
                    "main_missing_%": round(100 * _main_missing / len(_weathers), 1),
                    "missing_after_other_%": round(100 * _still_missing / len(_weathers), 1),
                    "has_other_column": any((_v + " Other") in w for w in _weathers),
                }
            )
    weather_missing = pd.DataFrame(_rows)
    weather_missing.to_csv("outputs/data_v2/raw_weather_missingness.csv", index=False)
    weather_missing
    return (weather_missing,)


@app.cell
def _(plt, weather_missing):
    _dest = weather_missing[weather_missing["where"] == "destination (last row)"].set_index("variable")
    _fig, _ax = plt.subplots(figsize=(9, 4.5))
    _x = range(len(_dest))
    _ax.bar([i - 0.2 for i in _x], _dest["main_missing_%"], width=0.4, label="main station missing", color="salmon")
    _ax.bar([i + 0.2 for i in _x], _dest["missing_after_other_%"], width=0.4, label="after 'Other' station", color="teal")
    _ax.set_xticks(list(_x), _dest.index, rotation=45, ha="right")
    _ax.set_ylabel("% of trains missing")
    _ax.set_title("Destination weather missingness (sample of 3 months)")
    _ax.legend()
    _ax.grid(True, axis="y", alpha=0.3)
    _fig.savefig("outputs/data_v2/raw_weather_missingness.png", bbox_inches="tight")
    _fig
    return


@app.cell
def _(pd, raw):
    _dists = [
        v
        for st in raw["stops"]
        for r in st
        for k, v in (r.get("weather_observations") or {}).items()
        if k.endswith("Other Distance") and v is not None
    ]
    _other_dist = pd.Series(_dists, name="other_station_distance_km").describe().round(1)
    _other_dist.to_csv("outputs/data_v2/raw_other_station_distance.csv")
    _other_dist
    return


@app.cell
def _(pd, raw):
    _ok = raw[~raw["cancelled"]]
    _sched = [r["scheduledTime"] for st in _ok["stops"] for r in st]
    print("share of scheduledTime ending with Z:", sum(t.endswith("Z") for t in _sched) / len(_sched))

    _train1 = _ok[_ok["trainNumber"] == 1]
    _t1 = pd.DataFrame(
        {
            "month": _train1["month"],
            "departureDate": _train1["departureDate"],
            "first_scheduledTime": [st[0]["scheduledTime"] for st in _train1["stops"]],
        }
    ).drop_duplicates("month")
    _t1.to_csv("outputs/data_v2/raw_train1_times.csv", index=False)
    _t1
    return


@app.cell
def _(pd, raw):
    _ok = raw[~raw["cancelled"]]
    _rows = []
    for _st in _ok["stops"]:
        _last = _st[-1]
        _w = _last.get("weather_observations") or {}
        if _last.get("actualTime") and _w.get("weather_timestamp") and (_last.get("differenceInMinutes") or 0) >= 20:
            _rows.append(
                {
                    "scheduledTime": _last["scheduledTime"],
                    "actualTime": _last["actualTime"],
                    "weather_timestamp": _w["weather_timestamp"],
                    "delay": _last["differenceInMinutes"],
                }
            )
    _timing = pd.DataFrame(_rows)
    for _c in ["scheduledTime", "actualTime", "weather_timestamp"]:
        _timing[_c] = pd.to_datetime(_timing[_c])
    _timing["min_from_scheduled"] = (_timing["weather_timestamp"] - _timing["scheduledTime"]).dt.total_seconds() / 60
    _timing["min_from_actual"] = (_timing["weather_timestamp"] - _timing["actualTime"]).dt.total_seconds() / 60
    print("late (>=20 min) trains checked:", len(_timing))
    print("weather within 10 min of scheduled:", (_timing["min_from_scheduled"].abs() <= 10).mean())
    print("weather within 10 min of actual:", (_timing["min_from_actual"].abs() <= 10).mean())
    _timing.head(10).to_csv("outputs/data_v2/raw_weather_timing_examples.csv", index=False)
    _timing.head(10)
    return


@app.cell
def _(pd, raw):
    _ok = raw[~raw["cancelled"]].copy()
    _ok["last_cancelled"] = _ok["stops"].apply(lambda st: bool(st[-1].get("cancelled")))
    _ok["n_cancelled_rows"] = _ok["stops"].apply(lambda st: sum(1 for r in st if r.get("cancelled")))
    _ok["last_diff"] = _ok["stops"].apply(lambda st: st[-1].get("differenceInMinutes"))
    _ok["reached"] = _ok["stops"].apply(
        lambda st: [r["stationShortCode"] for r in st if r["type"] == "ARRIVAL" and r.get("actualTime") and not r.get("cancelled")][-1:]
    )
    _partial = pd.DataFrame(
        {
            "not cancelled trains": [len(_ok)],
            "some rows cancelled": [int((_ok["n_cancelled_rows"] > 0).sum())],
            "final stop cancelled": [int(_ok["last_cancelled"].sum())],
            "final stop cancelled but has delay value": [int((_ok["last_cancelled"] & _ok["last_diff"].notna()).sum())],
            "fully cancelled trains": [int(raw["cancelled"].sum())],
        }
    )
    _partial.to_csv("outputs/data_v2/raw_partial_cancellations.csv", index=False)
    print(_ok[_ok["last_cancelled"]][["month", "trainNumber", "trainType", "last_diff", "reached"]].head(8))
    _partial
    return


@app.cell
def _(raw):
    _ok = raw[~raw["cancelled"]].copy()
    _ok = _ok[_ok["stops"].apply(lambda st: not st[-1].get("cancelled"))]
    _ok["delay"] = _ok["stops"].apply(lambda st: st[-1].get("differenceInMinutes"))
    _ok["dep_delay"] = _ok["stops"].apply(lambda st: st[0].get("differenceInMinutes"))
    _ok["is_late"] = _ok["delay"] >= 5
    _tt = (
        _ok.groupby("timetableType")
        .agg(
            n=("trainNumber", "size"),
            median_train_number=("trainNumber", "median"),
            mean_delay=("delay", "mean"),
            min_delay=("delay", "min"),
            max_delay=("delay", "max"),
            late_rate=("is_late", "mean"),
        )
        .round(3)
    )
    _tt.to_csv("outputs/data_v2/raw_timetable_type.csv")
    print("corr(origin departure delay, final arrival delay):", round(_ok[["dep_delay", "delay"]].corr().iloc[0, 1], 3))
    _tt
    return


@app.cell
def _(Request, gzip, json, pd, urlopen):
    def _get(url):
        _req = Request(url, headers={"Accept-Encoding": "gzip"})
        with urlopen(_req, timeout=30) as _resp:
            return json.loads(gzip.decompress(_resp.read()))

    _base = "https://rata.digitraffic.fi/api/v1/metadata/"
    _cat = pd.DataFrame(_get(_base + "cause-category-codes"))[["categoryCode", "categoryName", "validFrom"]]
    _det = pd.DataFrame(_get(_base + "detailed-cause-category-codes"))[["detailedCategoryCode", "detailedCategoryName"]]
    _det["categoryCode"] = _det["detailedCategoryCode"].str[0]

    # my own translations of the finnish category names
    _english = {
        "A": "Timetable and operation",
        "E": "Running ahead of schedule",
        "H": "Staff",
        "I": "Other causes (incl. weather)",
        "J": "Delay in train formation",
        "K": "Rolling stock (units and coaches)",
        "L": "Traffic management",
        "M": "Passenger service",
        "O": "Accident",
        "P": "Traffic control systems",
        "R": "Track work",
        "S": "Electric track / catenary",
        "T": "Track structure",
        "V": "Traction stock (locomotives)",
        "X": "Causes reported to traffic centre",
    }
    _cat["categoryName_en"] = _cat["categoryCode"].map(_english)
    cause_codes = _cat.merge(_det, on="categoryCode", how="left")
    cause_codes.to_csv("outputs/data_v2/cause_codes.csv", index=False)
    cause_codes
    return (cause_codes,)


@app.cell
def _(cause_codes, pd, raw):
    _names = cause_codes.drop_duplicates("categoryCode").set_index("categoryCode")["categoryName_en"]
    _ok = raw[~raw["cancelled"]]
    _ok = _ok[_ok["stops"].apply(lambda st: not st[-1].get("cancelled") and st[-1].get("differenceInMinutes") is not None)]

    _rows = []
    for _m, _st in zip(_ok["month"], _ok["stops"]):
        _cats = [c.get("categoryCode") for r in _st for c in (r.get("causes") or [])]
        _rows.append({"month": _m, "late": _st[-1]["differenceInMinutes"] >= 5, "n_causes": len(_cats), "cats": _cats})
    _c = pd.DataFrame(_rows)

    _coverage = _c.groupby(["month", "late"]).agg(n_trains=("n_causes", "size"), share_with_cause=("n_causes", lambda x: (x > 0).mean()), median_causes=("n_causes", "median")).round(3)
    _coverage.to_csv("outputs/data_v2/raw_cause_coverage.csv")
    print(_coverage)

    _late = _c[_c["late"]].explode("cats").dropna(subset=["cats"]).reset_index(drop=True)
    _mix = (100 * pd.crosstab(_late["cats"], _late["month"], normalize="columns")).round(1)
    _mix.insert(0, "name", _mix.index.map(_names))
    _mix = _mix.sort_values(_mix.columns[-1], ascending=False)
    _mix.to_csv("outputs/data_v2/raw_cause_mix_late_trains.csv")
    _mix
    return


if __name__ == "__main__":
    app.run()
