"""Run the full pipeline: validate on two back-test folds, calibrate stock-out risk, forecast 2018 Q1.

Usage:  python src/train.py        (about 3-4 minutes on a laptop)
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from forecasting import *  # noqa: E402,F401,F403

REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
FOLD_A, FOLD_B, FINAL = "2017-06-30", "2017-09-30", "2017-12-31"   # training cut-offs


def main():
    df = load_sales()
    Y, series, dates = to_matrix(df)
    feat = build_features(Y, series, dates)

    # ---------------------------------------------------------------- back-test two folds
    folds, models = {}, {}
    for name, cut in (("A", FOLD_A), ("B", FOLD_B)):
        m = fit_model(feat, cut)
        v = fold_rows(feat, cut)
        v["hgb"] = predict(m, v)
        v["snaive"] = seasonal_naive(v)
        v["dow4w"] = v["dow_mean_4w"].fillna(v["roll_91"]).values
        folds[name], models[name] = v, m
        print(f"fold {name}: {v['date'].min().date()} to {v['date'].max().date()}")

    rows = []
    for label, col in (("Weekday average (last 4 weeks)", "dow4w"),
                       ("Seasonal naive x YoY growth", "snaive"),
                       ("Gradient boosting (HGB, 18 features)", "hgb")):
        for name, v in folds.items():
            rows.append({"Model": label, "Fold": name, **metrics(v["sales"], v[col])})
    comp = pd.DataFrame(rows)
    avg = comp.groupby("Model", sort=False)[["SMAPE", "MAE", "RMSE", "WAPE %"]].mean().round(3)
    comp.to_csv(REPORTS / "model_comparison_by_fold.csv", index=False)
    avg.to_csv(REPORTS / "model_comparison.csv")
    print(avg)

    # ---------------------------------------------------------------- aggregate (for ARIMA / LSTM notebook)
    vb = folds["B"]
    daily = vb.groupby("date")[["sales", "hgb", "snaive"]].sum().rename(columns={"sales": "actual_total"})
    daily.to_csv(REPORTS / "fold_b_daily_total.csv")
    agg_smape = {"hgb": round(smape(daily["actual_total"], daily["hgb"]), 3),
                 "snaive": round(smape(daily["actual_total"], daily["snaive"]), 3)}

    # ---------------------------------------------------------------- stock-out calibration + back-test
    va = folds["A"].rename(columns={"hgb": "forecast"})
    vb2 = folds["B"].rename(columns={"hgb": "forecast"})
    sigma_a, factors_a = calibrate(va)                     # calibrate on fold A only ...
    bt = []
    for L in (7, 14):
        w = window_errors(vb2, L).join(sigma_a, on=["store", "item"])
        for sl in (0.90, 0.95, 0.98):
            z = norm.ppf(sl)
            cal = w["actual"] <= w["forecast"] + z * c_factor(factors_a, L) * w["sigma_d"] * np.sqrt(L)
            raw = w["actual"] <= w["forecast"] + z * w["sigma_d"] * np.sqrt(L)   # no calibration factor
            bt.append({"lead_time_days": L, "target_service_level": sl,
                       "achieved_calibrated": round(float(cal.mean()), 3),
                       "achieved_uncalibrated": round(float(raw.mean()), 3)})
    bt = pd.DataFrame(bt)                                  # ... and test on fold B (unseen)
    bt.to_csv(REPORTS / "stockout_backtest.csv", index=False)
    print(bt)

    # production calibration uses both folds
    sigma, _ = calibrate(pd.concat([va, vb2]))
    factors = {}
    for L in LEAD_TIMES:
        parts = [window_errors(v, L).join(sigma, on=["store", "item"]) for v in (va, vb2)]
        w = pd.concat(parts)
        z = (w["actual"] - w["forecast"]) / (w["sigma_d"] * np.sqrt(L))
        factors[L] = float(np.sqrt(np.mean(z ** 2)))
    (REPORTS / "calibration.json").write_text(json.dumps({str(k): round(v, 4) for k, v in factors.items()}, indent=2))

    # ---------------------------------------------------------------- feature importance (fold B)
    sample = vb.sample(8000, random_state=42)
    pi = permutation_importance(models["B"], sample[FEATURES], np.log1p(sample["sales"]),
                                n_repeats=3, random_state=42, scoring="neg_mean_absolute_error")
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False).round(4)
    imp.to_csv(REPORTS / "feature_importance.csv", header=["importance"])

    # ---------------------------------------------------------------- final model + 2018 Q1 forecast
    final = fit_model(feat, FINAL)
    fut = fold_rows(feat, FINAL)
    fut["forecast"] = predict(final, fut)
    assert fut["date"].min() == pd.Timestamp("2018-01-01") and fut["date"].max() == pd.Timestamp("2018-03-31")
    fut[["store", "item", "date", "forecast"]].round(2).to_csv(ROOT / "data" / "forecasts_2018Q1.csv", index=False)
    joblib.dump(final, ROOT / "models" / "hgb_forecaster.joblib", compress=3)

    # ---------------------------------------------------------------- ABC + series stats
    last_year = df[df["date"] > "2016-12-31"]
    abc = abc_classes(last_year)
    stats = (last_year.groupby(["store", "item"])["sales"].mean().rename("avg_daily_2017").to_frame()
             .join(abc).join(sigma).reset_index())
    stats.round(3).to_csv(REPORTS / "series_stats.csv", index=False)

    # safety-stock units: uniform 95% vs ABC policy (A 98 / B 95 / C 90), lead time 7 days
    fc_mean = fut.groupby(["store", "item"])["forecast"].mean().rename("fc").reset_index()
    st = stats.merge(fc_mean, on=["store", "item"])
    c7 = c_factor(factors, 7)
    st["ss95"] = norm.ppf(0.95) * c7 * st["sigma_d"] * np.sqrt(7)
    st["ss_abc"] = st["abc"].map({"A": 0.98, "B": 0.95, "C": 0.90}).map(norm.ppf) * c7 * st["sigma_d"] * np.sqrt(7)
    policy = {"series_per_class": stats["abc"].value_counts().to_dict(),
              "volume_share_per_class_pct": (last_year.merge(abc.reset_index(), on=["store", "item"])
                                             .groupby("abc")["sales"].sum() / last_year["sales"].sum() * 100).round(1).to_dict(),
              "safety_stock_units_uniform95": round(float(st["ss95"].sum()), 0),
              "safety_stock_units_abc_policy": round(float(st["ss_abc"].sum()), 0)}

    out = {"validation_folds": {"A": "2017-07-01 to 2017-09-28", "B": "2017-10-01 to 2017-12-29"},
           "average_over_folds": avg.to_dict("index"), "aggregate_total_smape_fold_b": agg_smape,
           "calibration_factors": {str(k): round(v, 3) for k, v in factors.items()}, "inventory_policy": policy}
    (REPORTS / "metrics.json").write_text(json.dumps(out, indent=2, default=str))

    # ---------------------------------------------------------------- figures
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily.index, daily["actual_total"], label="Actual", lw=1.6)
    ax.plot(daily.index, daily["hgb"], label="Gradient boosting", lw=1.6)
    ax.plot(daily.index, daily["snaive"], label="Seasonal naive", lw=1, ls="--", alpha=.8)
    ax.set_title("All stores and items: daily total demand, Oct to Dec 2017 (unseen fold)")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "forecast_vs_actual_total.png", dpi=130); plt.close(fig)

    g = vb[(vb["store"] == 1) & (vb["item"] == 1)]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(g["date"], g["sales"], label="Actual"); ax.plot(g["date"], g["hgb"], label="Forecast")
    ax.set_title("Store 1, item 1 (unseen fold)"); ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "forecast_vs_actual_series.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    b7 = bt[bt["lead_time_days"] == 7].set_index("target_service_level")
    b7[["achieved_uncalibrated", "achieved_calibrated"]].plot.bar(ax=ax, rot=0)
    for i, sl in enumerate(b7.index):
        ax.hlines(sl, i - .4, i + .4, colors="k", linestyles="--")
    ax.set_ylim(0.6, 1.02); ax.set_title("Service level achieved on unseen fold (7-day lead time)\n(dashed = target)")
    fig.tight_layout(); fig.savefig(FIG / "service_level_backtest.png", dpi=130); plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
