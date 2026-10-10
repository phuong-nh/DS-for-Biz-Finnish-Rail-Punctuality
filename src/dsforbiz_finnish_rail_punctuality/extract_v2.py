import ast
import os
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any, cast

import pandas as pd

# timeTableRows uses bare nan which literal_eval can't parse
_NAN = re.compile(r"\bnan\b")

WEATHER_VARS = {
    "temp": "Air temperature",
    "wind": "Wind speed",
    "gust": "Gust speed",
    "precip_intensity": "Precipitation intensity",
    "snow_depth": "Snow depth",
    "visibility": "Horizontal visibility",
    "humidity": "Relative humidity",
    "cloud": "Cloud amount",
    "present_weather": "Present weather (auto)",
}
# these have a value from a second nearby station ("<name> Other")
FALLBACK_VARS = ["wind", "gust", "precip_intensity", "snow_depth", "visibility"]

# grouped by who is responsible: Väylävirasto (infra), VR (operator), Fintraffic (traffic)
CAUSE_GROUPS = {
    "P": "infra",
    "R": "infra",
    "S": "infra",
    "T": "infra",
    "K": "operator",
    "V": "operator",
    "H": "operator",
    "J": "operator",
    "M": "operator",
    "L": "traffic",
}
GROUP_NAMES = ["infra", "operator", "traffic", "other"]


def _to_float(val: object) -> float | None:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    return None


def _to_int(val: object) -> int | None:
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        try:
            return int(val)
        except ValueError:
            return None
    return None


def _local_hour(utc_time: object) -> int | None:
    if not isinstance(utc_time, str) or not utc_time:
        return None
    return pd.Timestamp(utc_time).tz_convert("Europe/Helsinki").hour


def _minutes_between(start: object, end: object) -> float | None:
    if not isinstance(start, str) or not isinstance(end, str) or not start or not end:
        return None
    return (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 60


def _weather(stop: dict[str, Any]) -> dict[str, Any]:
    raw = stop.get("weather_observations")
    if isinstance(raw, dict):
        return cast(dict[str, Any], raw)
    return {}


def _weather_value(w: dict[str, Any], short: str) -> tuple[float | None, bool]:
    name = WEATHER_VARS[short]
    val = _to_float(w.get(name))
    if val is None and short in FALLBACK_VARS:
        other = _to_float(w.get(name + " Other"))
        if other is not None:
            return other, True
    return val, False


def _stop_weather(stop: dict[str, Any], prefix: str) -> dict[str, object]:
    w = _weather(stop)
    out: dict[str, object] = {}
    for short in WEATHER_VARS:
        val, used_fallback = _weather_value(w, short)
        out[f"{prefix}_{short}"] = val
        if short in FALLBACK_VARS:
            out[f"{prefix}_{short}_fallback"] = used_fallback
    return out


def _route_weather(stops: list[dict[str, Any]]) -> dict[str, object]:
    values: dict[str, list[float]] = {
        s: []
        for s in [
            "temp",
            "snow_depth",
            "wind",
            "gust",
            "precip_intensity",
            "visibility",
        ]
    }
    for stop in stops:
        w = _weather(stop)
        for short, vals in values.items():
            val, _ = _weather_value(w, short)
            if val is not None:
                vals.append(val)
    return {
        "route_min_temp": min(values["temp"]) if values["temp"] else None,
        "route_max_snow_depth": max(values["snow_depth"])
        if values["snow_depth"]
        else None,
        "route_max_wind": max(values["wind"]) if values["wind"] else None,
        "route_max_gust": max(values["gust"]) if values["gust"] else None,
        "route_max_precip_intensity": max(values["precip_intensity"])
        if values["precip_intensity"]
        else None,
        "route_min_visibility": min(values["visibility"])
        if values["visibility"]
        else None,
    }


def _causes(stops: list[dict[str, Any]]) -> dict[str, object]:
    minutes = dict.fromkeys(GROUP_NAMES, 0.0)
    has_group = dict.fromkeys(GROUP_NAMES, False)
    n_causes = 0
    weather_coded = False
    last_coded_diff = 0.0
    for stop in stops:
        diff = _to_float(stop.get("differenceInMinutes"))
        raw_causes = stop.get("causes")
        causes: list[dict[str, Any]] = []
        if isinstance(raw_causes, list):
            causes = [
                cast(dict[str, Any], c)
                for c in cast(list[object], raw_causes)
                if isinstance(c, dict)
            ]
        if causes:
            increment = 0.0
            if diff is not None and diff > last_coded_diff:
                increment = diff - last_coded_diff
            for cause in causes:
                group = CAUSE_GROUPS.get(str(cause.get("categoryCode")), "other")
                has_group[group] = True
                minutes[group] += increment / len(causes)
                if str(cause.get("detailedCategoryCode")) in ("I1", "I2"):
                    weather_coded = True
            n_causes += len(causes)
            if diff is not None:
                last_coded_diff = diff

    out: dict[str, object] = {
        "n_causes": n_causes,
        "cause_weather_coded": weather_coded,
    }
    for group in GROUP_NAMES:
        out[f"cause_{group}"] = has_group[group]
        out[f"cause_{group}_minutes"] = round(minutes[group], 2)
    top = max(GROUP_NAMES, key=lambda g: minutes[g])
    out["cause_dominant"] = top if minutes[top] > 0 else None
    return out


def parse_train(row: pd.Series[Any]) -> dict[str, object] | None:
    is_cancelled = bool(row["cancelled"])
    base: dict[str, object] = {
        "train_number": _to_int(row["trainNumber"]) or 0,
        "departure_date": str(row["departureDate"]),
        "train_type": str(row["trainType"]),
        "operator": str(row["operatorShortCode"]),
        "timetable_type": str(row["timetableType"]),
        "cancelled": is_cancelled,
    }
    if is_cancelled:
        return base

    raw_eval: object = ast.literal_eval(_NAN.sub("None", str(row["timeTableRows"])))
    stops_list: list[object] = (
        cast(list[object], raw_eval) if isinstance(raw_eval, list) else []
    )
    stops: list[dict[str, Any]] = [
        cast(dict[str, Any], s) for s in stops_list if isinstance(s, dict)
    ]
    if len(stops) == 0:
        return None

    first_stop = stops[0]
    last_stop = stops[-1]
    origin = str(first_stop.get("stationShortCode") or "")
    destination = str(last_stop.get("stationShortCode") or "")

    reached = not bool(last_stop.get("cancelled"))
    delay = _to_float(last_stop.get("differenceInMinutes")) if reached else None

    # Finnish official threshold: >= 5 min at final station is late
    is_late: int | None = None
    if delay is not None:
        is_late = 1 if delay >= 5 else 0

    dep_date = str(row["departureDate"])
    out = base | {
        "reached_destination": reached,
        "n_cancelled_rows": sum(1 for s in stops if s.get("cancelled")),
        "origin": origin,
        "destination": destination,
        "corridor": f"{origin}-{destination}" if origin and destination else None,
        "departure_hour_local": _local_hour(first_stop.get("scheduledTime")),
        "arrival_hour_local": _local_hour(last_stop.get("scheduledTime")),
        "day_of_week": pd.Timestamp(dep_date).dayofweek if dep_date else None,
        "scheduled_duration_min": _minutes_between(
            first_stop.get("scheduledTime"), last_stop.get("scheduledTime")
        ),
        "n_commercial_stops": sum(
            1 for s in stops if s.get("type") == "ARRIVAL" and s.get("commercialStop")
        ),
        "delay_minutes": delay,
        "is_late": is_late,
        "dep_delay_minutes": _to_float(first_stop.get("differenceInMinutes")),
    }
    out |= _stop_weather(first_stop, "origin")
    out |= _stop_weather(last_stop, "dest")
    out |= _route_weather(stops)
    out |= _causes(stops)
    return out


def extract_data(csv_path: str, nrows: int | None = None) -> pd.DataFrame:
    cols = [
        "trainNumber",
        "departureDate",
        "trainType",
        "operatorShortCode",
        "timetableType",
        "cancelled",
        "timeTableRows",
    ]
    parsed: list[dict[str, object]] = []
    chunk_size = 2000
    for chunk in pd.read_csv(csv_path, nrows=nrows, usecols=cols, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            res = parse_train(row)
            if res is not None:
                parsed.append(res)
    return pd.DataFrame(parsed)


def extract_single_month(month: str, data_dir: str = "data") -> pd.DataFrame:
    cached_path = f"{data_dir}/extracted_v2_{month}.csv"
    if os.path.exists(cached_path):
        return pd.read_csv(cached_path)
    raw_path = f"{data_dir}/matched_data_{month}.csv"
    if not os.path.exists(raw_path):
        return pd.DataFrame()
    df = extract_data(raw_path)
    df.to_csv(cached_path, index=False)
    return df


def extract_months(
    months: list[str],
    data_dir: str = "data",
    output_path: str | None = None,
    max_workers: int = 3,
) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    total = len(months)
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            month_futures = [
                (month, executor.submit(extract_single_month, month, data_dir))
                for month in months
            ]
            for idx, (month, future) in enumerate(month_futures, 1):
                df_month = future.result()
                if not df_month.empty:
                    dfs.append(df_month)
                print(f"[{idx}/{total}] Extracted {month} ({len(df_month)} rows)")
    else:
        for idx, month in enumerate(months, 1):
            df_month = extract_single_month(month, data_dir)
            if not df_month.empty:
                dfs.append(df_month)
            print(f"[{idx}/{total}] Extracted {month} ({len(df_month)} rows)")

    combined: pd.DataFrame = (
        pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    )
    if output_path is not None:
        combined.to_csv(output_path, index=False)
    return combined


if __name__ == "__main__":
    sample_df = extract_data("data/matched_data_2024_01.csv", nrows=20)
    pd.set_option("display.max_columns", None)
    print(sample_df.T)
