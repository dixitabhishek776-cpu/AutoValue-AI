from pathlib import Path
import json
import streamlit as st
import pandas as pd

from src.valuation import load_model, predict_price, load_sample_data
from src.analytics import market_summary, brand_summary, city_summary, fuel_summary

st.set_page_config(
    page_title="AutoValue AI | Intelligent Car Valuation",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
    --navy:#0b1220; --navy2:#14233d; --blue:#2563eb;
    --cyan:#38bdf8; --ink:#152238; --muted:#64748b;
    --line:#e5eaf2; --surface:#ffffff; --bg:#f3f6fb;
}
.stApp {background:var(--bg); color:var(--ink);}
.block-container {max-width:1250px; padding-top:1.7rem; padding-bottom:2.5rem;}
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0b1220 0%,#14233d 100%);
    border-right:1px solid #263957;
}
[data-testid="stSidebar"] * {color:#eaf1ff !important;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding:10px 12px; border-radius:10px;
}
.hero {
    position:relative; overflow:hidden; padding:28px 32px;
    border-radius:22px; color:white;
    background:linear-gradient(115deg,#0b1220 0%,#172b4d 58%,#2455a4 100%);
    box-shadow:0 12px 30px rgba(15,35,70,.14);
    margin-bottom:24px;
}
.hero:after {
    content:""; position:absolute; width:230px; height:230px;
    border:35px solid rgba(255,255,255,.055); border-radius:50%;
    right:-55px; top:-95px;
}
.eyebrow {
    text-transform:uppercase; letter-spacing:2px; font-size:11px;
    font-weight:800; color:#7dd3fc; margin-bottom:8px;
}
.hero h1 {color:#fff !important; font-size:36px !important;
    letter-spacing:-1px; margin:0; line-height:1.15;}
.hero p {color:#d5e2f7 !important; font-size:15px; margin:9px 0 0;}
.section-title {font-size:22px; font-weight:800; color:#14233d;
    letter-spacing:-.5px; margin:0 0 4px;}
.section-sub {color:#64748b; font-size:13px; margin-bottom:18px;}
.panel {
    background:white; border:1px solid var(--line); border-radius:17px;
    padding:20px 22px; box-shadow:0 4px 16px rgba(15,35,70,.035);
}
div[data-testid="stMetric"] {
    background:white; padding:17px 18px; border:1px solid var(--line);
    border-radius:15px; box-shadow:0 3px 12px rgba(15,35,70,.025);
}
div[data-testid="stMetricLabel"] {color:#64748b !important; font-size:12px;}
div[data-testid="stMetricValue"] {color:#14233d !important; font-weight:800;}
.stButton > button {
    border-radius:11px; min-height:46px; font-weight:700;
    border:0; background:linear-gradient(100deg,#2563eb,#0ea5e9);
    box-shadow:0 5px 14px rgba(37,99,235,.18);
}
.stButton > button:hover {border:0; filter:brightness(1.05);}
.stSelectbox, .stNumberInput {margin-bottom:5px;}
div[data-baseweb="select"] > div, div[data-testid="stNumberInput"] input {
    border-radius:10px !important; background:white;
}
hr {border-color:#e5eaf2;}
[data-testid="stDataFrame"] {border:1px solid #e5eaf2; border-radius:12px;}
div[data-testid="stAlert"] {border-radius:12px;}
.small-note {color:#64748b; font-size:12px;}
.result-card {
    background:linear-gradient(120deg,#eff6ff,#f0f9ff);
    border:1px solid #bfdbfe; border-radius:17px; padding:22px;
    margin-top:15px;
}
.result-label {font-size:12px; text-transform:uppercase; letter-spacing:1.5px;
    color:#1d4ed8; font-weight:800;}
.result-price {font-size:38px; font-weight:850; color:#0f2444;
    letter-spacing:-1px; margin:5px 0;}
@media(max-width:700px) {
    .block-container {padding:1rem .8rem;}
    .hero {padding:22px 20px;}
    .hero h1 {font-size:29px !important;}
    .result-price {font-size:30px;}
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    return load_sample_data()


@st.cache_resource
def get_model():
    return load_model()


def money(value):
    return f"₹{value:,.0f}"


st.sidebar.markdown("""
<div style="padding:8px 4px 20px">
<div style="font-size:27px">🚘</div>
<div style="font-size:20px;font-weight:850;letter-spacing:-.5px">AutoValue <span style="color:#7dd3fc">AI</span></div>
<div style="font-size:11px;color:#9fb2d0;letter-spacing:1.4px;text-transform:uppercase">Intelligent valuation</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "WORKSPACE",
    ["Price Prediction", "Market Analytics", "About & Data"],
)
st.sidebar.markdown("---")
st.sidebar.caption("ML-powered used-car insights")
st.sidebar.caption("Historical listing data • INR")

st.markdown("""
<div class="hero">
  <div class="eyebrow">Automotive intelligence</div>
  <h1>Know the value.<br>Drive the decision.</h1>
  <p>Machine-learning price estimates and used-car market insights.</p>
</div>
""", unsafe_allow_html=True)

try:
    data = get_data()
except Exception as exc:
    st.error(f"Dataset loading failed: {exc}")
    st.stop()


if page == "Price Prediction":
    st.markdown('<div class="section-title">Vehicle valuation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Enter the vehicle details to estimate its historical listed price.</div>',
        unsafe_allow_html=True,
    )

    brands = sorted(data["brand"].dropna().unique().tolist())
    if not brands:
        st.error("No brands are available in the dataset.")
        st.stop()

    with st.container(border=True):
        st.markdown("#### 01 / Vehicle details")
        col1, col2 = st.columns(2)

        with col1:
            brand = st.selectbox("Manufacturer", brands)
            models = sorted(
                data.loc[data["brand"] == brand, "model_name"]
                .dropna().unique().tolist()
            )
            model_name = st.selectbox("Car model", models)

        rows = data[
            (data["brand"] == brand) &
            (data["model_name"] == model_name)
        ]

        with col2:
            years = sorted(
                pd.to_numeric(rows["year"], errors="coerce")
                .dropna().astype(int).unique().tolist(),
                reverse=True,
            )
            year = st.selectbox("Manufacturing year", years)
            fuel_options = sorted(rows["fuel_type"].dropna().unique().tolist())
            fuel = st.selectbox("Fuel type", fuel_options)

        st.markdown("#### 02 / Usage & market")
        col3, col4 = st.columns(2)
        with col3:
            km = st.number_input(
                "Kilometres driven",
                min_value=0, max_value=800000,
                value=45000, step=1000,
            )
        with col4:
            cities = sorted(data["city"].dropna().unique().tolist())
            city = st.selectbox("City / market", cities)

        historical = rows[rows["year"] == year]["price"]
        if not historical.empty:
            st.caption(
                f"Dataset reference · Median listing for this model/year: **{money(historical.median())}**"
            )

        predict_clicked = st.button(
            "✦  Estimate listed price",
            type="primary",
            use_container_width=True,
        )

    if predict_clicked:
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
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Estimated historical listed price</div>
                    <div class="result-price">{money(prediction)}</div>
                    <div style="color:#475569;font-size:13px">
                    {year} {brand}{model_name} · {km:,} km · {fuel} · {city}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(
                    "An estimate—not a guaranteed sale price. Actual value depends on "
                    "condition, ownership, documentation, and current market demand."
                )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    else:
        st.markdown(
            '<p class="small-note">Tip: accurate vehicle details help make the estimate more meaningful.</p>',
            unsafe_allow_html=True,
        )


elif page == "Market Analytics":
    st.markdown('<div class="section-title">Market intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Explore patterns across the historical used-car listings in the dataset.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total listings", f"{len(data):,}")
    c2.metric("Manufacturers", f"{data['brand'].nunique():,}")
    c3.metric("Markets covered", f"{data['city'].nunique():,}")

    st.markdown("#### Listing price trend")
    st.plotly_chart(market_summary(data), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Manufacturer insights")
        st.dataframe(brand_summary(data), use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### City comparison")
        st.dataframe(city_summary(data), use_container_width=True, hide_index=True)

    st.markdown("#### Fuel-type distribution")
    st.dataframe(fuel_summary(data), use_container_width=True, hide_index=True)


else:
    st.markdown('<div class="section-title">About the project</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">A machine-learning application for used-car listing estimates and market exploration.</div>',
        unsafe_allow_html=True,
    )

    a, b = st.columns(2)
    a.metric("Dataset records", f"{len(data):,}")
    b.metric("Input features", "7")

    st.markdown("""
    <div class="panel">
      <h4 style="margin-top:0;color:#14233d">How it works</h4>
      <p style="color:#526176;font-size:14px">
      AutoValue AI uses vehicle attributes—including manufacturer, model, year,
      distance travelled, fuel type, and city—to estimate a historical listed price.
      The analytics workspace summarizes patterns in the supplied listings.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Dataset & responsible use")
    st.write("- Target: historical listed asking price in INR—not verified transaction price.")
    st.write("- The dataset is limited and may not represent today's market.")
    st.write("- Estimates do not replace inspection, documentation checks, or professional valuation.")

    metrics_path = Path("reports/model_metrics.json")
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as file:
            metrics = json.load(file)
        st.markdown("#### Held-out model evaluation")
        st.json(metrics.get("model_comparison", []))
