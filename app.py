import streamlit as st
import pandas as pd
from pathlib import Path
from src.valuation import load_model, predict_price, load_sample_data
from src.analytics import market_summary, brand_summary

st.set_page_config(page_title="AutoValue AI | Used Car Valuation", page_icon="🚘", layout="wide")
st.markdown("""
<style>
.stApp {background: #f5f7fb;}
[data-testid="stSidebar"] {background: #101b35;}
[data-testid="stSidebar"] * {color: #f5f7fb !important;}
.hero {padding: 1.5rem 1.7rem; border-radius: 18px; color: white;
background: linear-gradient(115deg,#111d3a,#334d83); margin-bottom: 1rem;}
.hero h1 {color:white; margin:0;} .hero p {color:#dce5ff; margin:.4rem 0 0;}
.small-note {color:#64748b;font-size:.85rem}
div[data-testid="stMetric"] {background:white;padding:14px;border:1px solid #e6eaf2;border-radius:12px;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🚘 AutoValue AI")
page = st.sidebar.radio("Navigation", ["Price Prediction", "Market Analytics", "About & Data"])
st.sidebar.caption("Used-car valuation • ML regression")

st.markdown('<div class="hero"><h1>AutoValue AI</h1><p>Intelligent used-car price estimation and market insights</p></div>', unsafe_allow_html=True)

sample = load_sample_data()
model_path = Path("models/used_car_model.joblib")
model = load_model(model_path)

if page == "Price Prediction":
    st.subheader("Estimate your car's resale value")
    st.info("Enter the vehicle details. The estimate is a model-based indication, not a guaranteed sale or dealer quote.")
    col1, col2, col3 = st.columns(3)
    brands = sorted(sample["brand"].dropna().unique().tolist())
    with col1:
        brand = st.selectbox("Brand", brands)
        year = st.number_input("Manufacturing year", min_value=2000, max_value=2026, value=2020, step=1)
        fuel = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    with col2:
        km = st.number_input("Kilometres driven", min_value=0, max_value=500000, value=45000, step=1000)
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        owner = st.selectbox("Previous owners", [0, 1, 2, 3, 4], index=1)
    with col3:
        mileage = st.number_input("Mileage (km/l or km/charge equivalent)", min_value=1.0, max_value=60.0, value=18.0, step=0.5)
        engine = st.number_input("Engine capacity (CC)", min_value=0, max_value=8000, value=1200, step=100)
        seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9], index=2)

    if st.button("Estimate resale price", type="primary", use_container_width=True):
        if model is None:
            st.error("No trained model found. Run `python train.py` first. The app will not invent a price.")
        else:
            record = {"brand":brand, "year":int(year), "km_driven":int(km), "fuel":fuel,
                      "transmission":transmission, "owner":int(owner), "mileage":float(mileage),
                      "engine_cc":int(engine), "seats":int(seats)}
            price = predict_price(model, record)
            st.success("Prediction generated")
            st.metric("Estimated resale value", f"₹{price:,.0f}", help="Model output in Indian Rupees")
            st.caption("This estimate depends on the training data's coverage, quality and target definition.")

    with st.expander("Model status"):
        if model is None:
            st.warning("Model artifact not available. Train using a verified dataset.")
        else:
            st.write("A model artifact was loaded from `models/used_car_model.joblib`.")
            st.write("Check `reports/model_metrics.json` for its evaluation metadata.")

elif page == "Market Analytics":
    st.subheader("Dataset overview")
    st.caption("Analytics below describe the currently loaded dataset. Replace the bundled demonstration data with a licensed real listing dataset for market conclusions.")
    a,b,c = st.columns(3)
    a.metric("Records", f"{len(sample):,}")
    b.metric("Brands", f"{sample.brand.nunique():,}")
    c.metric("Median listed price", f"₹{sample.selling_price.median():,.0f}")
    st.plotly_chart(market_summary(sample), use_container_width=True)
    st.subheader("Brand-level price overview")
    st.dataframe(brand_summary(sample), use_container_width=True, hide_index=True)

else:
    st.subheader("About the project")
    st.write("""
    AutoValue AI is a machine-learning prototype for estimating used-car resale prices.
    The pipeline separates data validation, preprocessing, model training and inference.
    """)
    st.warning("""
    **Dataset disclosure:** The included CSV is a synthetically generated demonstration dataset.
    It is provided to make the project runnable and to test the application workflow. It is NOT
    real CarDekho data, real market evidence, or suitable for claiming commercial prediction accuracy.
    Replace it with a properly licensed real-world dataset before presenting real-world performance.
    """)
    st.markdown("**Recommended evaluation:** MAE, RMSE, R², residual plots and a held-out test set.")
    st.markdown("**Target variable:** `selling_price` (INR).")
    st.markdown("**Input columns:** brand, year, km_driven, fuel, transmission, owner, mileage, engine_cc, seats.")
