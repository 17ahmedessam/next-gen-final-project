from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent

st.set_page_config(page_title="Customer Segments & 90-Day Spend", layout="wide")


@st.cache_resource
def load_bundle():
    bundle = joblib.load(BASE / "clv_model.joblib")
    # The forest was trained with n_jobs=-1. Inside a web app this opens a process pool for every
    # prediction and crashes on Windows, so we predict single-threaded (fast enough for one customer).
    forest = bundle["model"].regressor_.steps[-1][1]
    if hasattr(forest, "n_jobs"):
        forest.n_jobs = 1
    return bundle


@st.cache_data
def load_customers():
    return pd.read_csv(BASE / "customers_scored.csv").set_index("Customer ID")


bundle = load_bundle()
customers = load_customers()
model = bundle["model"]
features = bundle["features"]
actions = bundle["actions"]

st.title("Customer Lifetime Value & Segmentation")
st.caption(
    "Online Retail II (UK gift retailer, 2009-2011). Customers are grouped by behavior "
    "(Recency, Frequency, Monetary) and the model predicts spend in the next 90 days."
)

tab_dash, tab_lookup = st.tabs(["Segments dashboard", "Customer lookup"])

# ------------------------------------------------------------------ dashboard
with tab_dash:
    summary = (
        customers.groupby("Segment")
        .agg(
            Customers=("Recency", "size"),
            Median_Recency_days=("Recency", "median"),
            Median_Orders=("Frequency", "median"),
            Median_Total_Spend=("Monetary", "median"),
            Avg_Predicted_Spend=("predicted_next_90d", "mean"),
            Actual_Spend_Total=("actual_next_90d", "sum"),
        )
        .sort_values("Actual_Spend_Total", ascending=False)
    )
    summary["Share_of_Customers_%"] = (summary["Customers"] / summary["Customers"].sum() * 100).round(1)
    summary["Share_of_Revenue_%"] = (
        summary["Actual_Spend_Total"] / summary["Actual_Spend_Total"].sum() * 100
    ).round(1)
    summary["Recommended_Action"] = summary.index.map(actions)

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(customers):,}")
    top_seg = summary.index[0]
    c2.metric("Top segment by revenue", top_seg)
    c3.metric(
        f"{top_seg}: share of customers / revenue",
        f"{summary.loc[top_seg, 'Share_of_Customers_%']}% / {summary.loc[top_seg, 'Share_of_Revenue_%']}%",
    )

    st.subheader("Segment profiles")
    st.dataframe(
        summary[
            [
                "Customers",
                "Share_of_Customers_%",
                "Median_Recency_days",
                "Median_Orders",
                "Median_Total_Spend",
                "Avg_Predicted_Spend",
                "Share_of_Revenue_%",
                "Recommended_Action",
            ]
        ].round(0),
        
    )

    st.subheader("Revenue at stake by segment (actual spend, Sep-Dec 2011)")
    st.bar_chart(summary["Actual_Spend_Total"])
    st.caption("Revenue at stake = what each segment actually spent in the 90 days after the cutoff date (2011-09-11).")

# --------------------------------------------------------------------- lookup
with tab_lookup:
    st.subheader("Find a customer")
    example_ids = customers.sort_values("predicted_next_90d", ascending=False).head(5).index.tolist()
    st.caption("Example IDs: " + ", ".join(str(i) for i in example_ids))

    cid = st.number_input("Customer ID", min_value=0, value=int(example_ids[0]), step=1)

    if int(cid) not in customers.index:
        st.warning("Customer ID not found. Try one of the example IDs above.")
    else:
        row = customers.loc[[int(cid)]]
        predicted = float(model.predict(row[features])[0])
        segment = row["Segment"].iloc[0]

        a, b, c = st.columns(3)
        a.metric("Segment", segment)
        b.metric("Predicted spend, next 90 days", f"£{predicted:,.0f}")
        c.metric("Orders so far", int(row["Frequency"].iloc[0]))

        st.info("**Recommended action:** " + actions.get(segment, "-"))

        st.write("Customer profile")
        profile = row[["Recency", "Frequency", "Monetary", "Tenure", "AvgOrderValue", "Spend_last90"]].rename(
            columns={
                "Recency": "Days since last order",
                "Frequency": "Orders",
                "Monetary": "Net spend (£)",
                "Tenure": "Days since first order",
                "AvgOrderValue": "Avg order value (£)",
                "Spend_last90": "Spend in last 90 days (£)",
            }
        )
        st.dataframe(profile.round(0))

st.divider()
st.caption(
    "Limitations: the model predicts a typical spend value, not an exact amount, so use it to rank "
    "customers and choose actions, not to forecast total revenue. Customers who appear for the first "
    "time after the cutoff date cannot be predicted."
)