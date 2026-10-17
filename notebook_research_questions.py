import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    return np, os, pd, plt, roc_auc_score


@app.cell
def _(os, pd):
    os.makedirs("outputs/rq", exist_ok=True)
    # saved 2024 test probabilities, no refitting here
    pre = pd.read_csv("outputs/it5/test_probabilities.csv")
    enroute = pd.read_csv("outputs/enroute/test_probabilities.csv")
    # late rate of the training years (2018-2023), used to undo the balanced class weights
    _cm = pd.read_csv("outputs/it6/cause_models.csv")
    train_prior = float(_cm.loc[_cm["target"] == "is_late", "positive_rate_train_%"].iloc[0]) / 100
    print("pre-departure test rows:", len(pre), "| en-route test rows:", len(enroute), "| train prior:", train_prior)
    return enroute, pre, train_prior


@app.cell
def _(np, roc_auc_score):
    def day_bootstrap(frame, col_a, col_b, n_boot=1000, by_day=True, seed=42):
        # resample whole days, trains on the same day are not independent (cascading delays)
        _rng = np.random.default_rng(seed)
        _y = frame["is_late"].to_numpy()
        _a = frame[col_a].to_numpy()
        _b = frame[col_b].to_numpy()
        _point = roc_auc_score(_y, _b) - roc_auc_score(_y, _a)
        if by_day:
            _groups = list(frame.groupby("departure_date").indices.values())
        else:
            _groups = [np.array([i]) for i in range(len(frame))]
        _diffs = []
        for _ in range(n_boot):
            _pick = _rng.integers(0, len(_groups), size=len(_groups))
            _ix = np.concatenate([_groups[i] for i in _pick])
            _diffs.append(roc_auc_score(_y[_ix], _b[_ix]) - roc_auc_score(_y[_ix], _a[_ix]))
        _diffs = np.array(_diffs)
        return {
            "point dAUC": round(float(_point), 4),
            "ci_low": round(float(np.percentile(_diffs, 2.5)), 4),
            "ci_high": round(float(np.percentile(_diffs, 97.5)), 4),
            "share_boot_le_0": round(float((_diffs <= 0).mean()), 4),
        }

    return (day_bootstrap,)


@app.cell
def _(day_bootstrap, enroute, pd, pre):
    _rows = []
    for _set, _name in [("B", "destination weather"), ("C", "origin weather"), ("D", "all weather")]:
        _rows.append({"comparison": f"pre-departure Set {_set} vs A ({_name})", "resampling": "by day"} | day_bootstrap(pre, "prob_set_A", f"prob_set_{_set}"))
    # same comparison but resampling single trains, to see how much the day dependence matters
    _rows.append({"comparison": "pre-departure Set C vs A (origin weather)", "resampling": "single trains (iid)"} | day_bootstrap(pre, "prob_set_A", "prob_set_C", by_day=False, n_boot=300))
    for _cp, _g in enroute.groupby("checkpoint_%"):
        _rows.append({"comparison": f"en-route {_cp}%: E2 vs E1 (weather now + ahead)", "resampling": "by day"} | day_bootstrap(_g, "prob_E1", "prob_E2"))
    boot = pd.DataFrame(_rows)
    boot["ci_width"] = (boot["ci_high"] - boot["ci_low"]).round(4)
    boot.to_csv("outputs/rq/bootstrap_weather_gain.csv", index=False)
    boot
    return (boot,)


@app.cell
def _(boot, plt):
    _b = boot[boot["resampling"] == "by day"].iloc[::-1]
    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    _ax.errorbar(
        _b["point dAUC"],
        range(len(_b)),
        xerr=[_b["point dAUC"] - _b["ci_low"], _b["ci_high"] - _b["point dAUC"]],
        fmt="o",
        color="firebrick",
        capsize=4,
    )
    _ax.axvline(0, color="black", linewidth=1)
    _ax.set_yticks(range(len(_b)), _b["comparison"], fontsize=8)
    _ax.set_xlabel("AUC gain from weather (2024 test), 95% day-block bootstrap CI")
    _ax.set_title("Is the weather gain different from zero?")
    _ax.grid(True, axis="x", alpha=0.3)
    _fig.savefig("outputs/rq/bootstrap_weather_gain.png", bbox_inches="tight")
    _fig
    return


@app.cell
def _(enroute, np, pd, pre, train_prior):
    def to_real_prob(p, prior):
        # balanced weights act like a 50/50 prior, move the odds back to the real late rate
        _odds = p / (1 - p) * prior / (1 - prior)
        return _odds / (1 + _odds)

    scores = {}
    for _set in ["A", "C", "D"]:
        scores[f"pre-departure Set {_set}"] = (pre["is_late"].to_numpy(), to_real_prob(pre[f"prob_set_{_set}"].to_numpy(), train_prior))
    for _cp in [0, 50]:
        _g = enroute[enroute["checkpoint_%"] == _cp]
        for _set in ["E1", "E2"]:
            scores[f"en-route {_cp}% {_set}"] = (_g["is_late"].to_numpy(), to_real_prob(_g[f"prob_{_set}"].to_numpy(), train_prior))

    _rows = []
    for _name, (_y, _p) in scores.items():
        _balanced = _p / (1 - _p) * (1 - train_prior) / train_prior
        _rows.append(
            {
                "score": _name,
                "actual late rate 2024": round(float(_y.mean()), 4),
                "mean balanced prob": round(float(np.mean(_balanced / (1 + _balanced))), 4),
                "mean corrected prob": round(float(np.mean(_p)), 4),
            }
        )
    _calib = pd.DataFrame(_rows)
    _calib.to_csv("outputs/rq/ev_calibration.csv", index=False)
    _calib
    return (scores,)


@app.cell
def _(np, pd, scores):
    # cost units: taking precautions for a train = 1 (late or not), a missed late train = r
    # with real probabilities the cheapest rule is: act if p > 1 / r (no tuning on the test year)
    _rows = []
    for _r in [2, 3, 5, 7, 10]:
        for _name, (_y, _p) in scores.items():
            _act = _p > 1 / _r
            _tp = int((_act & (_y == 1)).sum())
            _fp = int((_act & (_y == 0)).sum())
            _fn = int((~_act & (_y == 1)).sum())
            _n = len(_y)
            _cost = (_tp + _fp + _r * _fn) / _n
            _never = _r * _y.mean()
            _always = 1.0
            _rows.append(
                {
                    "cost ratio r (miss / action)": _r,
                    "score": _name,
                    "threshold 1/r": round(1 / _r, 3),
                    "share flagged": round(float(_act.mean()), 4),
                    "recall": round(_tp / max(_tp + _fn, 1), 4),
                    "precision": round(_tp / max(_tp + _fp, 1), 4),
                    "cost per train": round(float(_cost), 4),
                    "never act": round(float(_never), 4),
                    "always act": _always,
                    "saving vs best simple rule per 1000 trains": round(float(1000 * (min(_never, _always) - _cost)), 1),
                }
            )
    ev = pd.DataFrame(_rows)
    ev.to_csv("outputs/rq/ev_cost_by_ratio.csv", index=False)

    # value of weather = cost without weather - cost with weather
    _w = ev.pivot_table(index="cost ratio r (miss / action)", columns="score", values="cost per train")
    _value = pd.DataFrame(
        {
            "pre-departure: Set A - Set C (per 1000 trains)": (1000 * (_w["pre-departure Set A"] - _w["pre-departure Set C"])).round(1),
            "pre-departure: Set A - Set D (per 1000 trains)": (1000 * (_w["pre-departure Set A"] - _w["pre-departure Set D"])).round(1),
            "en-route 0%: E1 - E2 (per 1000 trains)": (1000 * (_w["en-route 0% E1"] - _w["en-route 0% E2"])).round(1),
            "en-route 50%: E1 - E2 (per 1000 trains)": (1000 * (_w["en-route 50% E1"] - _w["en-route 50% E2"])).round(1),
        }
    )
    _value.to_csv("outputs/rq/ev_value_of_weather.csv")
    print(_value)
    np.round(ev.head(), 3)
    return (ev,)


@app.cell
def _(ev, np, plt, scores):
    _r = 5
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for _name, _color in [
        ("pre-departure Set A", "gray"),
        ("pre-departure Set C", "royalblue"),
        ("pre-departure Set D", "firebrick"),
        ("en-route 50% E2", "darkorange"),
    ]:
        _y, _p = scores[_name]
        _order = np.argsort(-_p)
        _tp = np.cumsum(_y[_order])
        _k = np.arange(1, len(_y) + 1)
        # profit curve: saving vs "never act" when flagging the top-k trains
        _saving = (_r * _tp - _k) / len(_y) * 1000
        _axes[0].plot(_k / len(_y), _saving, color=_color, label=_name)
    _axes[0].axhline(0, color="black", linewidth=1)
    _axes[0].set_xlabel("share of trains flagged (highest risk first)")
    _axes[0].set_ylabel("saving vs never acting (cost units / 1000 trains)")
    _axes[0].set_title(f"Profit curve, cost ratio r = {_r}")
    _axes[0].legend(fontsize=8)
    _axes[0].grid(True, alpha=0.3)

    for _name, _color in [
        ("pre-departure Set A", "gray"),
        ("pre-departure Set C", "royalblue"),
        ("pre-departure Set D", "firebrick"),
        ("en-route 0% E2", "goldenrod"),
        ("en-route 50% E2", "darkorange"),
    ]:
        _e = ev[ev["score"] == _name]
        _axes[1].plot(_e["cost ratio r (miss / action)"], _e["saving vs best simple rule per 1000 trains"], marker="o", color=_color, label=_name)
    _axes[1].axhline(0, color="black", linewidth=1)
    _axes[1].set_xlabel("cost ratio r = cost of a missed late train / cost of a precaution")
    _axes[1].set_ylabel("saving vs best of never / always act (per 1000 trains)")
    _axes[1].set_title("Sensitivity to the cost assumption (threshold = 1 / r)")
    _axes[1].legend(fontsize=8)
    _axes[1].grid(True, alpha=0.3)
    _fig.savefig("outputs/rq/ev_profit_and_sensitivity.png", bbox_inches="tight")
    _fig
    return


if __name__ == "__main__":
    app.run()
