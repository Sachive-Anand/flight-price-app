import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import gdown

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
    min-height: 100vh;
}

.hero-section {
    background: linear-gradient(135deg, #0052cc 0%, #0077ff 50%, #00a8ff 100%);
    border-radius: 24px;
    padding: 48px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0, 120, 255, 0.3);
}

.hero-section::before {
    content: '✈';
    position: absolute;
    right: 40px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 120px;
    opacity: 0.15;
}

.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin: 0 0 8px 0;
    letter-spacing: -1px;
}

.hero-sub {
    font-size: 18px;
    color: rgba(255,255,255,0.8);
    margin: 0;
    font-weight: 300;
}

.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 700;
    color: #0099ff;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 20px;
}

.result-box {
    background: linear-gradient(135deg, #003d99 0%, #0066ff 100%);
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    box-shadow: 0 20px 60px rgba(0, 100, 255, 0.4);
    margin: 24px 0;
}

.result-price {
    font-family: 'Syne', sans-serif;
    font-size: 64px;
    font-weight: 800;
    color: white;
    margin: 0;
    letter-spacing: -2px;
}

.result-label {
    font-size: 14px;
    color: rgba(255,255,255,0.7);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

.badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 50px;
    font-size: 13px;
    font-weight: 600;
    margin-top: 16px;
}

.badge-budget  { background: rgba(0,200,100,0.2);  color: #00c864; border: 1px solid rgba(0,200,100,0.3); }
.badge-mid     { background: rgba(255,180,0,0.2);  color: #ffb400; border: 1px solid rgba(255,180,0,0.3); }
.badge-premium { background: rgba(255,60,60,0.2);  color: #ff4444; border: 1px solid rgba(255,60,60,0.3); }

.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: white;
}

.metric-label {
    font-size: 12px;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

.tip-box {
    background: rgba(0, 153, 255, 0.08);
    border: 1px solid rgba(0, 153, 255, 0.2);
    border-left: 4px solid #0099ff;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 16px 0;
    font-size: 14px;
    color: rgba(255,255,255,0.75);
}

.stSelectbox > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: white !important;
}

.stSlider > div > div > div { background: #0066ff !important; }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0052cc, #0099ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 32px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 8px 30px rgba(0, 100, 255, 0.4) !important;
    transition: all 0.3s ease !important;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 40px rgba(0, 100, 255, 0.6) !important;
}

label { color: rgba(255,255,255,0.7) !important; font-size: 13px !important; }
.stSelectbox label, .stSlider label { font-weight: 500 !important; }
hr { border-color: rgba(255,255,255,0.08) !important; }

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load Model from Google Drive ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = 'flight_price_model.pkl'

    if not os.path.exists(model_path):
        # ──────────────────────────────────────────────────────────────────────
        # HOW TO GET YOUR FILE ID:
        #   1. Upload flight_price_model.pkl to Google Drive
        #   2. Right click the file → Share → Anyone with the link → Copy link
        #   3. Your link looks like:
        #      https://drive.google.com/file/d/1ABC123XYZ.../view?usp=sharing
        #   4. Copy ONLY the part between /d/ and /view  →  1ABC123XYZ...
        #   5. Paste it below replacing PASTE_YOUR_FILE_ID_HERE
        # ──────────────────────────────────────────────────────────────────────
        file_id = '18gJXx4HANAsAuRm9gvLfe5Qq9qvIpfd8'

        with st.spinner('⏳ Downloading model for first time, please wait...'):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)

    with open(model_path, 'rb') as f:
        return pickle.load(f)

package = load_model()


# ── Encoding Maps ─────────────────────────────────────────────────────────────
AIRLINE_MAP = {
    "AirAsia": 0, "Air India": 1, "GO FIRST": 2,
    "IndiGo":  3, "SpiceJet": 4, "Vistara":  5
}
CITY_MAP = {
    "Bangalore": 0, "Chennai":  1, "Delhi":    2,
    "Hyderabad": 3, "Kolkata":  4, "Mumbai":   5
}
TIME_MAP = {
    "Afternoon": 0, "Early Morning": 1, "Evening": 2,
    "Late Night": 3, "Morning":      4, "Night":   5
}
STOPS_MAP = {"One": 0, "Two or More": 1, "Zero": 2}
CLASS_MAP = {"Business": 0, "Economy": 1}


# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <p class="hero-title">✈️ Flight Price Predictor</p>
    <p class="hero-sub">Predict Indian domestic flight prices instantly using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

if package is None:
    st.error("❌ Model could not be loaded. Please check your Google Drive File ID in app.py")
    st.stop()


# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><p class="card-title">✈ Flight Details</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    airline = st.selectbox("🏢 Airline",
        ["IndiGo", "Air India", "Vistara", "GO FIRST", "AirAsia", "SpiceJet"])

    source_city = st.selectbox("🛫 Source City",
        ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"])

    destination_city = st.selectbox("🛬 Destination City",
        ["Mumbai", "Delhi", "Bangalore", "Kolkata", "Hyderabad", "Chennai"])

with col2:
    flight_class = st.selectbox("💺 Class", ["Economy", "Business"])

    departure_time = st.selectbox("🌅 Departure Time",
        ["Morning", "Early Morning", "Evening", "Night", "Afternoon", "Late Night"])

    arrival_time = st.selectbox("🌆 Arrival Time",
        ["Evening", "Morning", "Early Morning", "Night", "Afternoon", "Late Night"])

with col3:
    stops = st.selectbox("🔄 Number of Stops", ["Zero", "One", "Two or More"])

    duration = st.slider("⏱️ Duration (hours)", 1.0, 50.0, 2.5, 0.5,
        help="Total flight duration in hours")

    days_left = st.slider("📅 Days Left for Departure", 1, 49, 15,
        help="How many days before departure are you booking?")

st.markdown('</div>', unsafe_allow_html=True)


# ── Tip Box ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tip-box">
💡 <strong>Tip:</strong> Prices tend to be lowest when booked 3–6 weeks in advance.
   Late night and early morning departures are usually cheaper than evening flights.
</div>
""", unsafe_allow_html=True)


# ── Predict Button ────────────────────────────────────────────────────────────
predict_btn = st.button("🔮  Predict Flight Price")

if predict_btn:
    try:
        model  = package['model']
        scaler = package['scaler']

        # Build input dataframe
        # 'flight' column is set to 0 — it has very low importance in the model
        input_data = pd.DataFrame([{
            'airline'         : AIRLINE_MAP[airline],
            'flight'          : 0,
            'source_city'     : CITY_MAP[source_city],
            'departure_time'  : TIME_MAP[departure_time],
            'stops'           : STOPS_MAP[stops],
            'arrival_time'    : TIME_MAP[arrival_time],
            'destination_city': CITY_MAP[destination_city],
            'class'           : CLASS_MAP[flight_class],
            'duration'        : duration,
            'days_left'       : days_left
        }])

        # Scale & predict
        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]

        # Price category
        if prediction < 5000:
            badge_html = '<span class="badge badge-budget">💚 Budget — Great Deal!</span>'
            tip = "🎉 That's a fantastic price! Book quickly before it changes."
        elif prediction < 15000:
            badge_html = '<span class="badge badge-mid">💛 Mid-Range Flight</span>'
            tip = "👍 Reasonable price. Consider booking soon to lock it in."
        else:
            badge_html = '<span class="badge badge-premium">🔴 Premium / Business</span>'
            tip = "💼 Premium pricing. Check Economy class for cheaper alternatives."

        # Result box
        st.markdown(f"""
        <div class="result-box">
            <p class="result-label">Estimated Flight Price</p>
            <p class="result-price">₹ {prediction:,.0f}</p>
            {badge_html}
        </div>
        """, unsafe_allow_html=True)

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{airline}</div>
                <div class="metric-label">Airline</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{source_city} → {destination_city}</div>
                <div class="metric-label">Route</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{days_left} days</div>
                <div class="metric-label">Days to Departure</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{duration}h</div>
                <div class="metric-label">Duration</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="tip-box">💡 {tip}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.info("💡 Make sure your Google Drive File ID is correct in app.py")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:rgba(255,255,255,0.3); font-size:13px; padding:10px'>
    Built with ❤️ using ExtraTreesRegressor · scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)