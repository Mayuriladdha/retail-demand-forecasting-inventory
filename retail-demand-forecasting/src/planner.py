"""Inventory planning helpers used by the Streamlit app (pure functions, easy to test)."""
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from forecasting import ROOT, c_factor, inventory_numbers, stockout_probability

SERVICE_BY_CLASS = {"A": 0.98, "B": 0.95, "C": 0.90}     # recommended cycle service levels


def load_assets():
    hist = pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["date"])
    fc = pd.read_csv(ROOT / "data" / "forecasts_2018Q1.csv", parse_dates=["date"])
    stats = pd.read_csv(ROOT / "reports" / "series_stats.csv")
    factors = {int(k): v for k, v in json.loads((ROOT / "reports" / "calibration.json").read_text()).items()}
    return hist, fc, stats, factors


def series_plan(fc, stats, factors, store, item, start, lead_time, service_level, stock, uplift=0.0):
    """Reorder maths for one store-item. `uplift` is a demand what-if (0.2 = +20%)."""
    g = fc[(fc["store"] == store) & (fc["item"] == item)].sort_values("date")
    window = g[(g["date"] >= pd.Timestamp(start))].head(lead_time)
    row = stats[(stats["store"] == store) & (stats["item"] == item)].iloc[0]
    mu = window["forecast"].sum() * (1 + uplift)
    inv = inventory_numbers(mu, row["sigma_d"], lead_time, factors, service_level)
    inv.update({"abc": row["abc"], "stockout_probability": stockout_probability(stock, mu, inv["sigma_L"]),
                "order_gap": max(0.0, inv["reorder_point"] - stock), "days_in_window": len(window)})
    return inv


def portfolio_risk(fc, stats, factors, start, lead_time, cover_days, uplift=0.0):
    """Stock-out probability for every store-item if each holds `cover_days` of average forecast demand."""
    w = fc[fc["date"] >= pd.Timestamp(start)].sort_values("date")
    w = w.groupby(["store", "item"]).head(lead_time)
    mu = w.groupby(["store", "item"])["forecast"].sum().rename("mu_L") * (1 + uplift)
    avg = w.groupby(["store", "item"])["forecast"].mean().rename("avg_daily")
    out = stats.set_index(["store", "item"]).join(mu).join(avg)
    out["stock"] = out["avg_daily"] * cover_days
    out["sigma_L"] = c_factor(factors, lead_time) * out["sigma_d"] * np.sqrt(lead_time)
    out["stockout_prob"] = 1 - norm.cdf((out["stock"] - out["mu_L"]) / out["sigma_L"])
    return out.reset_index().sort_values("stockout_prob", ascending=False)
