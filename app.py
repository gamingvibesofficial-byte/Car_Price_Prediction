
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="AutoValue AI | Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ---------------- DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

.stApp {
    background: radial-gradient(circle at top left, #172d50, #08111f 55%);
    color: #edf4ff;
    font-family: 'DM Sans', sans-serif;
}
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}
h1,h2,h3 {color:#f4f7ff !important;}
p,label {color:#aabbd4;}
.hero {
    padding: 38px;
    border-radius: 24px;
    background: linear-gradient(120deg,#163d70,#126b85);
    border: 1px solid #315c86;
    margin: 15px 0 25px 0;
}
.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}
.hero p {color:#d6e9ff;font-size:17px;}
.panel {
    background:#111e32;
    padding:25px;
    border-radius:18px;
    border:1px solid #293d59;
    margin-bottom:18px;
}
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background:#0b1729 !important;
    color:white !important;
    border-radius:10px !important;
    border:1px solid #344d70 !important;
}
div[data-testid="stFormSubmitButton"] button {
    width:100%;
    background:linear-gradient(100deg,#2878ee,#16a6c2);
    color:white;
    border:0;
    border-radius:12px;
    padding:13px;
    font-weight:800;
    font-size:17px;
}
div[data-testid="stFormSubmitButton"] button:hover {
    color:white;
    border:1px solid #80c4ff;
}
.result {
    background:linear-gradient(120deg,#104e48,#142e3c);
    border:1px solid #37a78b;
    padding:30px;
    border-radius:20px;
    text-align:center;
    margin-top:20px;
}
.price {
    color:#a7f3d0;
    font-size:45px;
    font-weight:800;
}
.metric {
    background:#14243a;
    padding:18px;
    border-radius:15px;
    border:1px solid #293d59;
    text-align:center;
}
.footer {
    text-align:center;
    color:#7187a5;
    padding:25px;
    border-top:1px solid #293d59;
    margin-top:30px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load(
        Path(__file__).parent / "car_price_model.pkl"
    )

try:
    model = load_model()
except Exception as e:
    st.error("Could not load the prediction model.")
    st.exception(e)
    st.stop()

# ---------------- HEADER ----------------
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;">
    <h3>🚗 AUTOVALUE <span style="color:#60a5fa">AI</span></h3>
    <span style="color:#86efac">● AI MODEL READY</span>
</div>
<div class="hero">
    <h1>Know What Your Car Is Worth.</h1>
    <p>Smart used-car price estimation powered by Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- FEATURE CARDS ----------------
a,b,c = st.columns(3)

with a:
    st.markdown("""
    <div class="metric">
    <h2>⚡</h2><b>Instant Prediction</b>
    <p>Quick price estimation</p>
    </div>""", unsafe_allow_html=True)

with b:
    st.markdown("""
    <div class="metric">
    <h2>🧠</h2><b>Random Forest AI</b>
    <p>Trained ML pipeline</p>
    </div>""", unsafe_allow_html=True)

with c:
    st.markdown("""
    <div class="metric">
    <h2>🇮🇳</h2><b>Indian Currency</b>
    <p>Prices in INR</p>
    </div>""", unsafe_allow_html=True)

st.write("")

left,right = st.columns([1.25,.75],gap="large")

# ---------------- CAR INPUT ----------------
with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("🚘 Enter Your Car Details")
    st.caption("Fill in the information to calculate an estimated resale value.")

    with st.form("car_details"):
        car_name = st.text_input(
            "Car Model / Name",
            placeholder="e.g. Maruti Swift"
        )

        col1,col2 = st.columns(2)

        with col1:
            year = st.number_input(
                "Manufacturing Year",
                min_value=1900,
                max_value=2026,
                value=2020
            )

            fuel = st.selectbox(
                "Fuel Type",
                ["Petrol","Diesel","CNG"]
            )

            transmission = st.selectbox(
                "Transmission",
                ["Manual","Automatic"]
            )

        with col2:
            km_driven = st.number_input(
                "Kilometres Driven",
                min_value=0,
                max_value=1000000,
                value=45000,
                step=1000
            )

            seller_type = st.selectbox(
                "Seller Type",
                ["Individual","Dealer","Trustmark Dealer"]
            )

            owner = st.selectbox(
                "Ownership",
                [
                    "First Owner",
                    "Second Owner",
                    "Third Owner",
                    "Fourth & Above Owner"
                ]
            )

        submitted = st.form_submit_button(
            "✨  PREDICT CAR PRICE"
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- INFORMATION PANEL ----------------
with right:
    st.markdown("""
    <div class="panel">
        <h3>📊 How It Works</h3>
        <p><b style="color:#60a5fa">01 — Enter Details</b><br>
        Provide your vehicle information.</p>
        <p><b style="color:#60a5fa">02 — AI Processing</b><br>
        Random Forest analyses the details.</p>
        <p><b style="color:#60a5fa">03 — Get Estimate</b><br>
        View your predicted resale price.</p>
    </div>
    <div class="panel">
        <h3>💡 Did You Know?</h3>
        <p>Car resale value can depend on vehicle age, mileage, fuel type, ownership and transmission.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if submitted:
    if not car_name.strip():
        st.warning("Please enter your car model.")
    else:
        car_input = pd.DataFrame([{
            "name": car_name.strip(),
            "year": int(year),
            "km_driven": int(km_driven),
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner
        }])

        try:
            predicted_price = float(model.predict(car_input)[0])
            predicted_price = max(0,predicted_price)

            st.markdown("---")
            st.subheader("🎯 Your Prediction Result")

            st.markdown(f"""
            <div class="result">
                <p style="color:#a7f3d0;font-size:15px;">
                    ESTIMATED RESALE VALUE
                </p>
                <div class="price">₹ {predicted_price:,.0f}</div>
                <p style="color:#b7d8ce;">
                    Indian Rupees (INR) • Random Forest Prediction
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.success("Your car price prediction is ready!")

            st.caption(
                "Note: This is an ML-based estimate, not a guaranteed selling price. Actual market values may vary."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <b>AUTOVALUE AI</b><br>
    Academic Machine Learning Project<br>
    Built using Python • Streamlit • Random Forest
</div>
""", unsafe_allow_html=True)
