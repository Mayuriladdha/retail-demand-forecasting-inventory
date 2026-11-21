"""Streamlit app: retail demand forecast, stock-out risk and inventory planner."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from planner import SERVICE_BY_CLASS, load_assets, portfolio_risk, series_plan  # noqa: E402

st.set_page_config(page_title="Demand & Inventory Planner", page_icon="📦", layout="wide")


@st.cache_data
def assets():
    return load_assets()


hist, fc, stats, factors = assets()

st.title("📦 Retail Demand Forecast & Inventory Planner")
st.caption("90-day demand forecast (Jan to Mar 2018) for 10 stores x 50 items, with stock-out risk and reorder points. "
           "Model: gradient boosting on calendar + lagged-sales features (SMAPE about 11.7% on unseen back-tests).")

tab1, tab2, tab3 = st.tabs(["Store-item planner", "Portfolio risk (what-if)", "About"])

# ------------------------------------------------------------------ tab 1
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    store = c1.selectbox("Store", sorted(stats["store"].unique()))
    item = c2.selectbox("Item", sorted(stats["item"].unique()))
    lead = c3.slider("Supplier lead time (days)", 1, 28, 7)
    start = c4.date_input("Planning date", pd.Timestamp("2018-01-01"),
                          min_value=pd.Timestamp("2018-01-01"), max_value=pd.Timestamp("2018-03-31") - pd.Timedelta(days=lead - 1))

    abc = stats[(stats["store"] == store) & (stats["item"] == item)]["abc"].iloc[0]
    d1, d2, d3 = st.columns(3)
    service = d1.slider("Target service level", 0.80, 0.995, SERVICE_BY_CLASS[abc], 0.005,
                        help=f"Recommended for class {abc}: {SERVICE_BY_CLASS[abc]:.0%}")
    uplift = d2.slider("Demand what-if (promo / shock)", -30, 50, 0, 5, format="%d%%") / 100
    base = series_plan(fc, stats, factors, store, item, start, lead, service, 0, uplift)
    stock = d3.number_input("Stock on hand (units)", min_value=0, value=int(round(base["reorder_point"])))

    plan = series_plan(fc, stats, factors, store, item, start, lead, service, stock, uplift)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ABC class", plan["abc"])
    m2.metric("Expected demand in lead time", f"{plan['mu_L']:.0f}")
    m3.metric("Safety stock", f"{plan['safety_stock']:.0f}")
    m4.metric("Reorder point", f"{plan['reorder_point']:.0f}")
    m5.metric("Stock-out probability", f"{plan['stockout_probability']:.1%}")

    if plan["order_gap"] > 0:
        st.error(f"Stock is below the reorder point: order at least **{plan['order_gap']:.0f} units** now.")
    else:
        st.success("Stock covers expected lead-time demand plus safety stock. No order needed yet.")

    h = hist[(hist["store"] == store) & (hist["item"] == item)].tail(90)
    f = fc[(fc["store"] == store) & (fc["item"] == item)]
    sigma_d = stats[(stats["store"] == store) & (stats["item"] == item)]["sigma_d"].iloc[0]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(h["date"], h["sales"], label="Actual (last 90 days)", color="#555")
    ax.plot(f["date"], f["forecast"] * (1 + uplift), label="Forecast", color="#1f77b4")
    ax.fill_between(f["date"], (f["forecast"] * (1 + uplift) - 1.645 * sigma_d).clip(lower=0),
                    f["forecast"] * (1 + uplift) + 1.645 * sigma_d, alpha=0.2, label="Approx. 90% range")
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=lead), color="orange", alpha=0.15,
               label="Lead-time window")
    ax.set_ylabel("units / day"); ax.legend(loc="upper left", ncol=2)
    st.pyplot(fig)

# ------------------------------------------------------------------ tab 2
with tab2:
    st.write("If every store-item holds the same days of cover, which ones are most likely to stock out during the lead time? (Risk is 50% when days of cover equal the lead time, because stock then equals expected demand.)")
    p1, p2, p3 = st.columns(3)
    cover = p1.slider("Days of cover on hand", 3, 30, 10)
    lead2 = p2.slider("Lead time (days)", 1, 28, 7, key="lead2")
    uplift2 = p3.slider("Demand uplift", -30, 50, 0, 5, format="%d%%", key="up2") / 100
    risk = portfolio_risk(fc, stats, factors, "2018-01-01", lead2, cover, uplift2)
    k1, k2, k3 = st.columns(3)
    k1.metric("Series with stock-out risk > 20%", int((risk["stockout_prob"] > 0.2).sum()))
    k2.metric("Average stock-out probability", f"{risk['stockout_prob'].mean():.1%}")
    k3.metric("Class A series at risk > 20%", int(((risk["stockout_prob"] > 0.2) & (risk["abc"] == "A")).sum()))
    by_store = risk.groupby("store")["stockout_prob"].mean().rename("avg stock-out probability")
    st.bar_chart(by_store)
    shown = risk.head(20)[["store", "item", "abc", "avg_daily", "stock", "mu_L", "stockout_prob"]].round(3)
    st.subheader("Top 20 highest-risk store-items")
    st.dataframe(shown, use_container_width=True, hide_index=True)
    st.download_button("Download full risk table", risk.round(4).to_csv(index=False), "stockout_risk.csv", "text/csv")

# ------------------------------------------------------------------ tab 3
with tab3:
    st.markdown("""
**How it works**
1. A single gradient-boosting model forecasts daily demand for all 500 store-item series, using calendar features, last-year sales for the same weekday, and rolling averages. Every feature is at least 90 days old, so the full 90-day horizon is forecast without compounding errors.
2. Forecast error per series (measured on two back-test periods) gives the uncertainty of demand over a lead time.
3. **Safety stock** = z x uncertainty over the lead time, with z set by the target service level.
   **Reorder point** = expected lead-time demand + safety stock. **Stock-out probability** is the chance that lead-time demand exceeds stock on hand.
4. **ABC policy**: A items (top 80% of volume) 98%, B items 95%, C items 90% service level.

Assumptions: demand over the lead time is approximately normal, the lead time is fixed, and one order is placed per cycle. Costs (holding, stock-out) are not in the dataset, so service levels are policy inputs, not cost-optimised.
""")
