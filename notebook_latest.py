import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    from dsforbiz_finnish_rail_punctuality.extract import extract_data

    return extract_data, mo, os, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # Finnish Rail Punctuality Analysis
    *Data Science for Business I*

    Testing the extracted dataset for December 2024 (latest available in the dataset).
    """)
    return


@app.cell
def _(extract_data, os, pd):
    csv_clean = "data/extracted_2024_12.csv"
    if os.path.exists(csv_clean):
        df = pd.read_csv(csv_clean)
    else:
        df = extract_data("data/matched_data_2024_12.csv")
        df.to_csv(csv_clean, index=False)
    df
    return (df,)


@app.cell
def _(df, mo):
    total_trains = len(df)
    cancelled_count = int(df["cancelled"].sum())
    non_cancelled = df[~df["cancelled"]]
    late_count = int(non_cancelled["is_late"].sum())
    late_pct = round((late_count / len(non_cancelled)) * 100, 2)

    mo.md(
        f"""
        ### December 2024 Summary (Latest Data)
        - **Total scheduled trips:** {total_trains}
        - **Cancelled trips:** {cancelled_count} ({cancelled_count / total_trains * 100:.2f}%)
        - **Arrived trips:** {len(non_cancelled)}
        - **Late arrivals (>= 5 min):** {late_count} ({late_pct}%)
        """
    )
    return (non_cancelled,)


@app.cell
def _(mo, non_cancelled, plt):
    fig, ax = plt.subplots(figsize=(8, 4))
    delays = non_cancelled["delay_minutes"].clip(lower=-10, upper=60)
    ax.hist(delays, bins=35, edgecolor="black", alpha=0.7)
    ax.axvline(5, color="red", linestyle="--", label="Late threshold (5 min)")
    ax.set_title("Arrival Delay Distribution at Destination (Dec 2024)")
    ax.set_xlabel("Delay in Minutes (clipped between -10 and 60)")
    ax.set_ylabel("Number of Trains")
    ax.legend()
    ax.grid(True, alpha=0.3)
    mo.as_html(fig)
    return


if __name__ == "__main__":
    app.run()
