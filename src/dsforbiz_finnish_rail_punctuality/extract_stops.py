import ast
import os
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any, cast

import pandas as pd

_NAN = re.compile(r"\bnan\b")

WEATHER_VARS = {
    "temp": "Air temperature",
    "wind": "Wind speed",
    "gust": "Gust speed",
    "precip_intensity": "Precipitation intensity",
    "snow_depth": "Snow depth",
    "visibility": "Horizontal visibility",
}
FALLBACK_VARS = ["wind", "gust", "precip_intensity", "snow_depth", "visibility"]
CHECKPOINTS = [0, 25, 50, 75]


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


def _weather_value(stop: dict[str, Any], short: str) -> float | None:
    raw = stop.get("weather_observations")
    w = cast(dict[str, Any], raw) if isinstance(raw, dict) else {}
    name = WEATHER_VARS[short]
    val = _to_float(w.get(name))
    if val is None and short in FALLBACK_VARS:
        val = _to_float(w.get(name + " Other"))
    return val


def _ahead_weather(stops: list[dict[str, Any]]) -> dict[str, object]:
    values: dict[str, list[float]] = {s: [] for s in WEATHER_VARS}
    for stop in stops:
        for short, vals in values.items():
            val = _weather_value(stop, short)
            if val is not None:
                vals.append(val)
    return {
        "ahead_min_temp": min(values["temp"]) if values["temp"] else None,
        "ahead_max_snow_depth": max(values["snow_depth"])
        if values["snow_depth"]
        else None,
        "ahead_max_wind": max(values["wind"]) if values["wind"] else None,
        "ahead_max_gust": max(values["gust"]) if values["gust"] else None,
        "ahead_max_precip_intensity": max(values["precip_intensity"])
        if values["precip_intensity"]
        else None,
        "ahead_min_visibility": min(values["visibility"])
        if values["visibility"]
        else None,
    }


def parse_train(row: pd.Series[Any]) -> list[dict[str, object]]:
    if bool(row["cancelled"]):
        return []
    raw_eval: object = ast.literal_eval(_NAN.sub("None", str(row["timeTableRows"])))
    stops_list: list[object] = (
        cast(list[object], raw_eval) if isinstance(raw_eval, list) else []
    )
    stops: list[dict[str, Any]] = [
        cast(dict[str, Any], s) for s in stops_list if isinstance(s, dict)
    ]
    if len(stops) < 2:
        return []

    first_stop = stops[0]
    last_stop = stops[-1]
    final_delay = _to_float(last_stop.get("differenceInMinutes"))
    if bool(last_stop.get("cancelled")) or final_delay is None:
        return []
    start = pd.Timestamp(str(first_stop.get("scheduledTime")))
    end = pd.Timestamp(str(last_stop.get("scheduledTime")))
    duration = (end - start).total_seconds() / 60
    if duration <= 0:
        return []

    # commercial departures where the train actually left, so the delay there is known
    known: list[tuple[int, float, float]] = []
    for i, s in enumerate(stops):
        delay = _to_float(s.get("differenceInMinutes"))
        if (
            s.get("type") == "DEPARTURE"
            and (s.get("commercialStop") or i == 0)
            and s.get("actualTime")
            and not s.get("cancelled")
            and delay is not None
        ):
            progress = (
                (pd.Timestamp(str(s.get("scheduledTime"))) - start).total_seconds()
                / 60
                / duration
            )
            known.append((i, progress, delay))
    if not known:
        return []

    origin = str(first_stop.get("stationShortCode") or "")
    destination = str(last_stop.get("stationShortCode") or "")
    dep_date = str(row["departureDate"])
    base: dict[str, object] = {
        "train_number": _to_int(row["trainNumber"]) or 0,
        "departure_date": dep_date,
        "train_type": str(row["trainType"]),
        "timetable_type": str(row["timetableType"]),
        "corridor": f"{origin}-{destination}",
        "departure_hour_local": start.tz_convert("Europe/Helsinki").hour,
        "day_of_week": pd.Timestamp(dep_date).dayofweek,
        "scheduled_duration_min": duration,
        "n_commercial_stops": sum(
            1 for s in stops if s.get("type") == "ARRIVAL" and s.get("commercialStop")
        ),
        "origin_delay": known[0][2] if known[0][0] == 0 else None,
        "final_delay": final_delay,
        # Finnish official threshold: >= 5 min at final station is late
        "is_late": 1 if final_delay >= 5 else 0,
    }

    out: list[dict[str, object]] = []
    for cp in CHECKPOINTS:
        before = [k for k in known if k[1] <= cp / 100 + 1e-9]
        if not before:
            continue
        idx, progress, delay = before[-1]
        prev_delay = known[known.index(before[-1]) - 1][2] if len(before) > 1 else None
        stop = stops[idx]
        sched = pd.Timestamp(str(stop.get("scheduledTime")))
        rec = base | {
            "checkpoint": cp,
            "station": str(stop.get("stationShortCode") or ""),
            "progress": round(progress, 4),
            "current_delay": delay,
            "prev_stop_delay": prev_delay,
            "remaining_min": (end - sched).total_seconds() / 60,
            "remaining_stops": sum(
                1
                for s in stops[idx + 1 :]
                if s.get("type") == "ARRIVAL" and s.get("commercialStop")
            ),
            "delay_to_go": final_delay - delay,
        }
        for short in WEATHER_VARS:
            rec[f"now_{short}"] = _weather_value(stop, short)
        rec |= _ahead_weather(stops[idx + 1 :])
        out.append(rec)
    return out


def extract_data(csv_path: str, nrows: int | None = None) -> pd.DataFrame:
    cols = [
        "trainNumber",
        "departureDate",
        "trainType",
        "timetableType",
        "cancelled",
        "timeTableRows",
    ]
    parsed: list[dict[str, object]] = []
    for chunk in pd.read_csv(csv_path, nrows=nrows, usecols=cols, chunksize=2000):
        for _, row in chunk.iterrows():
            parsed.extend(parse_train(row))
    return pd.DataFrame(parsed)


def extract_single_month(month: str, data_dir: str = "data") -> pd.DataFrame:
    cached_path = f"{data_dir}/extracted_stops_{month}.csv"
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

    combined: pd.DataFrame = (
        pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    )
    if output_path is not None:
        combined.to_csv(output_path, index=False)
    return combined


if __name__ == "__main__":
    sample_df = extract_data("data/matched_data_2024_01.csv", nrows=5)
    pd.set_option("display.max_columns", None)
    print(sample_df.T)
