import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    from dsforbiz_finnish_rail_punctuality.extract import extract_data

    return extract_data, mo, pd


@app.cell
def _(mo):
    mo.md(
        """
        # Finnish Rail Punctuality Analysis
        *Data Science for Business I*
        """
    )
    return


@app.cell
def _(extract_data):
    df_sample = extract_data("data/matched_data_2018_01.csv", nrows=50)
    df_sample
    return (df_sample,)


@app.cell
def _(df_sample, mo):
    mo.md(
        f"""
        ### Sample Summary (50 trains):
        - **Total non-cancelled:** {(~df_sample["cancelled"]).sum()}
        - **Cancelled:** {df_sample["cancelled"].sum()}
        - **Late (>= 5 min):** {df_sample["is_late"].sum()}
        """
    )
    return


if __name__ == "__main__":
    app.run()
