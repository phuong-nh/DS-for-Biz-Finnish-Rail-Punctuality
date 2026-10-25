# Finnish Rail Punctuality

My project for the course 30E03000 Data Science for Business I at Aalto (autumn 2026).

Finnish long-distance trains are late a lot, and everyone blames the winter. But a lot of it can already be guessed from the timetable, like the month, the route and how long the trip is. So I wanted to see if weather actually adds anything on top of that.

**Question:** does weather help predict if a train arrives 5+ minutes late, compared to just calendar, route and trip info?

I also looked at:

- whether weather at the departure station is enough
- which weather variables matter
- which kinds of delays are caused by weather
- how much the predictions would be worth in a simple cost model
- predicting the final delay while the train is already running

## Results

I used gradient boosting, trained on 2018-2023 and tested on 2024.

| Features | AUC |
|---|---|
| Calendar + route + trip (no weather) | 0.675 |
| + weather at destination | 0.693 |
| + weather at origin | 0.697 |
| + all weather | 0.701 |

So weather helps, but only a bit (around +0.02 AUC). The gain was positive in every year I tested (2021-2024) and the bootstrap intervals don't include zero, so it's not just noise.

Some other things I found:

- Weather at the departure station does about as well as using all the weather data, which is good because it's the only one you actually know before the train leaves.
- Cold matters the most. Below -15 C about 40% of trains were late, compared to about 15% in mild weather.
- Weather mostly predicts knock-on delays, where one problem spreads to other trains.
- Knowing how late the train already is matters way more than weather. AUC goes from about 0.70 before departure to 0.83 halfway through the trip.

## Data

I used the FI-TW dataset (Borin et al., 2026, on Kaggle). It has all Finnish train timetables from 2018 to 2024 with weather from the closest weather station. It's about 51 GB, so it's not in this repo.

Put the monthly files in `data/` like this:

```
data/matched_data_2018_01.csv
...
data/matched_data_2024_12.csv
```

After removing cancelled trains, trains that didn't reach the end and special ADHOC trains, there are 509,110 trips left, and 18.8% of them were late.

## How I did it

- One row per trip. Late means 5+ minutes late at the last station.
- I compared feature sets with and without weather, so weather only gets credit for what it adds.
- Split by time: train on 2018-2022, tune on 2023, test on 2024.
- For the confidence intervals I resampled whole days instead of single trains, because trains on the same day affect each other.
- Models: linear models, decision trees, random forest and gradient boosting.

## Files

- `src/dsforbiz_finnish_rail_punctuality/extract_v2.py` - turns the raw data into one row per trip
- `src/dsforbiz_finnish_rail_punctuality/extract_stops.py` - same but for the "while the train is running" part
- `project.py` - short version of the final pipeline
- `notebook_*.py` - marimo notebooks for each step
- `outputs/` - results and plots

There are a few versions of some notebooks (v1, v2, v3) because the project changed a lot along the way. The final ones are:

- `notebook_extract_v2.py` and `notebook_eda_v2.py` - data and exploration
- `notebook_modeling_v3.py` and `notebook_regression_v3.py` - main models
- `notebook_causes.py` - delay causes
- `notebook_rq_models.py` and `notebook_research_questions.py` - research questions, bootstrap, cost model
- `notebook_extract_stops.py`, `notebook_enroute.py` and `notebook_enroute_v2.py` - predictions while the train is running
- `notebook_report_figures.py` - figures for the report

The rest are older versions I kept so you can see how it developed.

## Running it

You need Python 3.15 and uv.

```bash
uv sync
uv run python project.py
uv run marimo edit notebook_modeling_v3.py
```

Run from the repo root. The first run takes a while because it parses all the raw files, but after that it uses the cached CSVs.

## Limitations

- I looked at 2024 results while developing before I added a separate validation year, so the 2024 numbers are a bit optimistic. The results for other years are a cleaner check.
- The data has weather observations, not forecasts, so in real life you'd only really have the departure weather.
- Weather stations can be up to 50 km from the track.
- Cancelled trains are left out, and those are probably the worst weather days.
- The models only explain about 10% of the variation in delay minutes, so delays are still mostly hard to predict.
