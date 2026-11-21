"""Core forecasting utilities: feature building, metrics, model, stock-out maths.

Design: every feature for target day t uses only sales from day t-H or earlier (H = 90 = forecast
horizon), so one model can predict a full 90-day horizon in one shot with no recursive error build-up
and no look-ahead leakage.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor

H = 90                      # forecast horizon in days
HORIZON_END = "2018-03-31"  # last date to forecast (Kaggle test period is 2018-01-01..2018-03-31)
ROOT = Path(__file__).resolve().parents[1]

FEATURES = ["store", "item", "dow", "month", "dom", "woy", "doy",
            "lag_h", "dow_mean_4w", "ly_same_day", "ly_smooth", "lag_728",
            "roll_7", "roll_28", "roll_91", "roll_365", "yoy_growth", "ly_ratio"]


# ----------------------------------------------------------------------------- data
def load_sales(path=None):
    path = path or ROOT / "data" / "train.csv"
    return pd.read_csv(path, parse_dates=["date"])


def to_matrix(df, end=HORIZON_END):
    """Long table -> (series x days) matrix. Future days are NaN. Returns Y, series index, dates."""
    dates = pd.date_range(df["date"].min(), end)
    wide = df.pivot_table(index=["store", "item"], columns="date", values="sales").reindex(columns=dates)
    return wide.values.astype(float), wide.index.to_frame(index=False), dates


def _shift(a, k):
    out = np.full_like(a, np.nan)
    out[:, k:] = a[:, :-k]
    return out


def _roll_mean_shifted(Y, window, shift):
    r = pd.DataFrame(Y.T).rolling(window, min_periods=window).mean().values.T
    return _shift(r, shift)


def build_features(Y, series, dates):
    """Return a long feature table (one row per series-day) with target `sales`."""
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    S, T = Y.shape
    f = {}
    lag_h = _shift(Y, H)
    f["lag_h"] = lag_h
    k_dow = int(np.ceil(H / 7) * 7)                       # same weekday, >= H days back (91)
    f["dow_mean_4w"] = np.nanmean(np.stack([_shift(Y, k_dow + 7 * i) for i in range(4)]), axis=0)
    f["ly_same_day"] = _shift(Y, 364)
    f["ly_smooth"] = np.nanmean(np.stack([_shift(Y, 357), _shift(Y, 364), _shift(Y, 371)]), axis=0)
    f["lag_728"] = _shift(Y, 728)
    for w in (7, 28, 91, 365):
        f[f"roll_{w}"] = _roll_mean_shifted(Y, w, H)
    f["yoy_growth"] = f["roll_365"] / _shift(f["roll_365"], 365)       # level now vs one year before
    f["ly_ratio"] = f["ly_smooth"] / f["roll_91"]                      # seasonal index a year ago
    cal = pd.DataFrame({"date": dates})
    cal["dow"] = cal["date"].dt.dayofweek
    cal["month"] = cal["date"].dt.month
    cal["dom"] = cal["date"].dt.day
    cal["woy"] = cal["date"].dt.isocalendar().week.astype(int).values
    cal["doy"] = cal["date"].dt.dayofyear
    out = {"store": np.repeat(series["store"].values, T), "item": np.repeat(series["item"].values, T),
           "date": np.tile(dates.values, S), "sales": Y.ravel()}
    for c in ("dow", "month", "dom", "woy", "doy"):
        out[c] = np.tile(cal[c].values, S)
    for k, v in f.items():
        out[k] = v.ravel().astype(np.float32)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- metrics
def smape(actual, pred):
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    denom = np.abs(a) + np.abs(p)
    return float(100 * np.mean(np.where(denom == 0, 0, 2 * np.abs(a - p) / np.where(denom == 0, 1, denom))))


def metrics(actual, pred):
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return {"SMAPE": round(smape(a, p), 3), "MAE": round(float(np.mean(np.abs(a - p))), 3),
            "RMSE": round(float(np.sqrt(np.mean((a - p) ** 2))), 3),
            "WAPE %": round(float(100 * np.abs(a - p).sum() / a.sum()), 3)}


# ----------------------------------------------------------------------------- baselines + model
def seasonal_naive(feat):
    """Same weekday a year ago (smoothed over 3 weeks) scaled by year-on-year growth."""
    return (feat["ly_smooth"] * feat["yoy_growth"]).fillna(feat["roll_91"]).values


def make_model(seed=42):
    return HistGradientBoostingRegressor(
        learning_rate=0.08, max_iter=350, max_leaf_nodes=63, min_samples_leaf=100,
        l2_regularization=1.0, categorical_features=[0, 1], random_state=seed)


def fit_model(feat, cut):
    """Train on all rows with target date <= cut (log1p target)."""
    train = feat[(feat["date"] <= cut) & feat["ly_smooth"].notna() & feat["roll_91"].notna()]
    model = make_model()
    model.fit(train[FEATURES], np.log1p(train["sales"]))
    return model


def predict(model, rows):
    return np.clip(np.expm1(model.predict(rows[FEATURES])), 0, None)


def fold_rows(feat, cut):
    """The H days after `cut` (the validation / forecast window)."""
    cut = pd.Timestamp(cut)
    return feat[(feat["date"] > cut) & (feat["date"] <= cut + pd.Timedelta(days=H))].copy()


# ----------------------------------------------------------------------------- stock-out maths
LEAD_TIMES = [1, 2, 3, 5, 7, 10, 14, 21, 28]


def daily_sigma(df_valid, pred_col="forecast"):
    """Per-series std of daily forecast error."""
    err = df_valid["sales"] - df_valid[pred_col]
    return err.groupby([df_valid["store"], df_valid["item"]]).std().rename("sigma_d")


def window_errors(df_valid, L, pred_col="forecast"):
    """Errors of non-overlapping L-day demand sums per series, scaled-ready."""
    rows = []
    for (s, i), g in df_valid.sort_values("date").groupby(["store", "item"]):
        a, f = g["sales"].values, g[pred_col].values
        for start in range(0, len(g) - L + 1, L):
            rows.append((s, i, a[start:start + L].sum(), f[start:start + L].sum()))
    return pd.DataFrame(rows, columns=["store", "item", "actual", "forecast"])


def calibrate(df_valid, pred_col="forecast"):
    """Return (sigma_d per series, c_L factors). sigma_L = c_L * sigma_d * sqrt(L)."""
    sig = daily_sigma(df_valid, pred_col)
    factors = {}
    for L in LEAD_TIMES:
        w = window_errors(df_valid, L, pred_col).join(sig, on=["store", "item"])
        z = (w["actual"] - w["forecast"]) / (w["sigma_d"] * np.sqrt(L))
        factors[L] = float(np.sqrt(np.mean(z ** 2)))
    return sig, factors


def c_factor(factors, L):
    xs = sorted(int(k) for k in factors)
    return float(np.interp(L, xs, [factors[k] if k in factors else factors[str(k)] for k in xs]))


def inventory_numbers(mu_L, sigma_d, L, factors, service_level):
    """Safety stock, reorder point for lead time L and a target cycle service level."""
    sigma_L = c_factor(factors, L) * sigma_d * np.sqrt(L)
    z = norm.ppf(service_level)
    ss = z * sigma_L
    return {"mu_L": mu_L, "sigma_L": sigma_L, "safety_stock": ss, "reorder_point": mu_L + ss}


def stockout_probability(stock, mu_L, sigma_L):
    return float(1 - norm.cdf((stock - mu_L) / sigma_L))


def abc_classes(df):
    """A = top 80% of volume, B = next 15%, C = rest (by total sales per store-item)."""
    tot = df.groupby(["store", "item"])["sales"].sum().sort_values(ascending=False)
    share = tot.cumsum() / tot.sum()
    cls = pd.Series(np.where(share <= 0.80, "A", np.where(share <= 0.95, "B", "C")), index=tot.index, name="abc")
    return cls
