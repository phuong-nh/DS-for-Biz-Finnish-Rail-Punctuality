import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, cast

import pandas as pd


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


def parse_train(row: pd.Series[Any]) -> dict[str, object] | None:
    is_cancelled = bool(row["cancelled"])
    train_num = _to_int(row["trainNumber"]) or 0
    dep_date = str(row["departureDate"])
    train_type = str(row["trainType"])

    if is_cancelled:
        return {
            "train_number": train_num,
            "departure_date": dep_date,
            "train_type": train_type,
            "cancelled": True,
            "origin": None,
            "destination": None,
            "corridor": None,
            "departure_hour": None,
            "day_of_week": None,
            "delay_minutes": None,
            "is_late": None,
            "dest_temp": None,
            "dest_snow_depth": None,
            "dest_wind_speed": None,
            "dest_precip_intensity": None,
        }

    # timeTableRows is stored as Python literals (nan/None/True/False)
    safe_env: dict[str, object] = {
        "nan": None,
        "None": None,
        "True": True,
        "False": False,
    }
    raw_eval: object = eval(str(row["timeTableRows"]), safe_env)
    stops_list: list[object] = (
        cast(list[object], raw_eval) if isinstance(raw_eval, list) else []
    )

    stops: list[dict[str, object]] = []
    for item in stops_list:
        if isinstance(item, dict):
            item_dict = cast(dict[object, object], item)
            stops.append({str(k): v for k, v in item_dict.items()})

    if len(stops) == 0:
        return None

    first_stop = stops[0]
    last_stop = stops[-1]

    origin = str(first_stop.get("stationShortCode") or "")
    destination = str(last_stop.get("stationShortCode") or "")
    corridor = f"{origin}-{destination}" if origin and destination else None

    sched_dep = str(first_stop.get("scheduledTime") or "")
    dep_hour = _to_int(sched_dep[11:13]) if len(sched_dep) >= 13 else None

    day_of_week = pd.Timestamp(dep_date).dayofweek if dep_date else None

    delay = _to_float(last_stop.get("differenceInMinutes"))

    # Finnish official threshold: >= 5 min at final station is late
    is_late: int | None = None
    if delay is not None:
        is_late = 1 if delay >= 5 else 0

    raw_weather: object = last_stop.get("weather_observations")
    weather: dict[str, object] = {}
    if isinstance(raw_weather, dict):
        weather_dict = cast(dict[object, object], raw_weather)
        weather = {str(k): v for k, v in weather_dict.items()}

    dest_temp = _to_float(weather.get("Air temperature"))
    dest_snow_depth = _to_float(weather.get("Snow depth"))
    dest_wind_speed = _to_float(weather.get("Wind speed"))
    dest_precip_intensity = _to_float(weather.get("Precipitation intensity"))

    return {
        "train_number": train_num,
        "departure_date": dep_date,
        "train_type": train_type,
        "cancelled": False,
        "origin": origin,
        "destination": destination,
        "corridor": corridor,
        "departure_hour": dep_hour,
        "day_of_week": day_of_week,
        "delay_minutes": delay,
        "is_late": is_late,
        "dest_temp": dest_temp,
        "dest_snow_depth": dest_snow_depth,
        "dest_wind_speed": dest_wind_speed,
        "dest_precip_intensity": dest_precip_intensity,
    }


def extract_data(csv_path: str, nrows: int | None = None) -> pd.DataFrame:
    cols = ["trainNumber", "departureDate", "trainType", "cancelled", "timeTableRows"]
    parsed: list[dict[str, object]] = []
    chunk_size = 2000
    for chunk in pd.read_csv(csv_path, nrows=nrows, usecols=cols, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            res = parse_train(row)
            if res is not None:
                parsed.append(res)
    return pd.DataFrame(parsed)


def extract_single_month(month: str, data_dir: str = "data") -> pd.DataFrame:
    cached_path = f"{data_dir}/extracted_{month}.csv"
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
    sample_df = extract_data("data/matched_data_2018_01.csv", nrows=10)
    print(sample_df)
