from pathlib import Path
import streamlit as st
import pandas as pd

from src.valuation import load_model, predict_price, load_sample_data
from src.analytics import market_summary, brand_summary, city_summary, fuel_summary
from src.schema import FEATURE_COLUMNS

st.set_page_config(
    page_title="AutoValue AI | Used Car Valuation",
    page_icon="🚘",
    layout="wide",
)

st.markdown("""
<style>
.stApp {background:#f5f7fb;}
[data-testid="stSidebar"] {background:#101b35;}
[data-testid="stSidebar"] * {color:#f5f7fb !important;}
.hero {
    padding:1.4rem 1.7rem;
    border-radius:18px;
    color:white;
    background:linear-gradient(115deg,#111d3a,#334d83);
    margin-bottom:1rem;
}
.hero h1 {color:white;margin:0;}
.hero p {color:#dce5ff;margin:.4rem 0 0;}
div[data-testid="stMetric"] {
    background:white;padding:14px;border:1px solid #e6eaf2;border-radius:12px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    return load_sample_data()


@st.cache_resource
def get_model():
    return load_model()


st.sidebar.title("🚘 AutoValue AI")
page = st.sidebar.radio(
    "Navigation",
    ["Price Prediction", "Market Analytics", "About & Data"],
)
st.sidebar.caption("Used-car valuation • ML regression")

st.markdown(
    '<div class="hero"><h1>AutoValue AI</h1>'
    '<p>Used-car price estimation and market insights</p></div>',
    unsafe_allow_html=True,
)

try:
    data = get_data()
except Exception as exc:
    st.error(f"Dataset loading failed: {exc}")
    st.stop()

if page == "Price Prediction":
    st.subheader("Estimate a used car's listed price")
    st.info(
        "This is a machine-learning estimate of a historical listing price, "
        "not a guaranteed sale price or dealer quote."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        brands = sorted(data["brand"].dropna().unique().tolist())
        brand = st.selectbox("Brand", brands)

        models = sorted(
            data.loc[data["brand"] == brand, "model_name"]
            .dropna().unique().tolist()
        )
        model_name = st.selectbox("Model", models)

        rows = data[
            (data["brand"] == brand) &
            (data["model_name"] == model_name)
        ]
        years = sorted(
            pd.to_numeric(rows["year"], errors="coerce")
            .dropna().astype(int).unique().tolist(),
            reverse=True,
        )
        year = st.selectbox("Manufacturing year", years)

    with col2:
        km = st.number_input(
            "Kilometres driven",
            min_value=0,
            max_value=800000,
            value=45000,
            step=1000,
        )
        fuel_options = sorted(rows["fuel_type"].dropna().unique().tolist())
        fuel = st.selectbox("Fuel type", fuel_options)

    with col3:
        cities = sorted(data["city"].dropna().unique().tolist())
        city = st.selectbox("City / market", cities)

        historical = rows[rows["year"] == year]["price"]
        if not historical.empty:
            st.metric("Historical median listing", f"₹{historical.median():,.0f}")

    if st.button("Estimate listed price", type="primary", use_container_width=True):
        try:
            model = get_model()
            full_names = rows["full_model_name"].dropna()
            if full_names.empty:
                st.error("No full model name found for this selection.")
            else:
                full_name = full_names.value_counts().index[0]
                record = {
                    "brand": brand,
                    "model_name": model_name,
                    "full_model_name": full_name,
                    "year": year,
                    "distance_travelled(kms)": km,
                    "fuel_type": fuel,
                    "city": city,
                }
                prediction = predict_price(model, record)
                st.success("Estimated historical listed price")
                st.metric("Estimated value", f"₹{prediction:,.0f}")
                st.caption(
                    "Actual value depends on condition, ownership, location, "
                    "documentation, and current market demand."
                )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

elif page == "Market Analytics":
    st.subheader("Used-car listing market overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Listings in dataset", f"{len(data):,}")
    c2.metric("Brands represented", f"{data['brand'].nunique():,}")
    c3.metric("Cities represented", f"{data['city'].nunique():,}")

    st.plotly_chart(market_summary(data), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Brand overview")
        st.dataframe(brand_summary(data), use_container_width=True)
    with right:
        st.markdown("#### City overview")
        st.dataframe(city_summary(data), use_container_width=True)

    st.markdown("#### Fuel-type overview")
    st.dataframe(fuel_summary(data), use_container_width=True)

else:
    st.subheader("About AutoValue AI")
    st.write(
        "AutoValue AI estimates used-car listing prices using vehicle "
        "characteristics and analyzes patterns in the supplied dataset."
    )
    st.markdown("#### Dataset and limitations")
    st.write(f"Records available: **{len(data):,}**")
    st.write(
        "- The target is historical listed asking price in INR, not verified final transaction price."
    )
    st.write(
        "- The dataset is limited and may not represent current market prices."
    )
    st.write(
        "- Predictions are estimates and should not replace vehicle inspection or professional valuation."
    )

    metrics_path = Path("reports/model_metrics.json")
    if metrics_path.exists():
        import json
        with open(metrics_path, encoding="utf-8") as file:
            metrics = json.load(file)
        st.markdown("#### Held-out model evaluation")
        st.json(metrics.get("model_comparison", []))
