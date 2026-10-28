import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page layout
st.set_page_config(page_title="Mumbai Real Estate Predictor & ROI Evaluator", page_icon="🏢", layout="wide")

st.title("🏢 Mumbai Real Estate Yield & Investment Evaluator")
st.write("Predict rental yield, auto-fetch micro-market appreciation & rent growth rates, and evaluate long-term investment profitability.")

# Load Model & Dataset safely
@st.cache_resource
def load_model():
    return joblib.load("models/mumbai_rent_model.pkl")

@st.cache_data
def load_data():
    paths = [
        r"C:\Users\dhruv\OneDrive\Desktop\mumbai_real_estate.csv",
        "data/mumbai_real_estate.csv",
        "mumbai_real_estate.csv"
    ]
    for path in paths:
        try:
            return pd.read_csv(path)
        except Exception:
            continue
    return pd.DataFrame()

model = load_model()
df = load_data()

# Haversine distance calculator
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return 6367 * 2 * np.arcsin(np.sqrt(a))

# Locality Micro-Market Growth Profiler
def get_locality_growth_rates(locality_name, dist_bkc_km, dist_nm_airport_km):
    loc_lower = str(locality_name).lower()
    
    if any(k in loc_lower for k in ['worli', 'bkc', 'bandra west', 'cuffe parade', 'nariman', 'malabar hill', 'lower parel', 'juhu', 'khar']):
        return 7.5, 6.0, "Prime Premium CBD", 2.5
    elif dist_nm_airport_km < 15.0 or any(k in loc_lower for k in ['navi mumbai', 'vashi', 'panvel', 'kharghar', 'belapur', 'airoli']):
        return 8.2, 5.2, "High-Growth Infra Corridor (Navi Mumbai / NMIA)", 3.0
    elif any(k in loc_lower for k in ['andheri', 'powai', 'goregaon', 'malad', 'borivali', 'ghatkopar', 'chembur', 'dadar', 'santacruz']):
        return 6.8, 5.5, "Established Suburban Corridor", 2.8
    elif any(k in loc_lower for k in ['thane', 'kalyan', 'dombivli', 'kandivali', 'mira road', 'bhayandar', 'mulund']):
        return 7.2, 5.0, "High-Demand Suburban Growth Hub", 3.2
    else:
        return 6.2, 5.0, "Standard MMR Market", 3.0

# Comprehensive Transportation Reference Data
RAILWAY_STATIONS = {
    # Western Line
    'Churchgate': (18.9322, 72.8264),
    'Marine Lines': (18.9447, 72.8244),
    'Charni Road': (18.9519, 72.8183),
    'Grant Road': (18.9620, 72.8157),
    'Mumbai Central': (18.9690, 72.8193),
    'Mahalaxmi': (18.9827, 72.8242),
    'Lower Parel': (18.9953, 72.8300),
    'Prabhadevi': (19.0142, 72.8361),
    'Dadar': (19.0178, 72.8478),
    'Matunga Road': (19.0287, 72.8447),
    'Mahim': (19.0406, 72.8458),
    'Bandra': (19.0544, 72.8402),
    'Khar Road': (19.0697, 72.8398),
    'Santacruz': (19.0827, 72.8413),
    'Vile Parle': (19.0991, 72.8444),
    'Andheri': (19.1197, 72.8464),
    'Jogeshwari': (19.1360, 72.8489),
    'Ram Mandir': (19.1513, 72.8504),
    'Goregaon': (19.1646, 72.8493),
    'Malad': (19.1866, 72.8486),
    'Kandivali': (19.2045, 72.8519),
    'Borivali': (19.2290, 72.8573),
    'Dahisar': (19.2494, 72.8596),
    'Mira Road': (19.2801, 72.8558),
    'Bhayandar': (19.3130, 72.8515),
    'Vasai Road': (19.3813, 72.8325),
    'Virar': (19.4554, 72.8113),
    # Central Line
    'CSMT': (18.9400, 72.8353),
    'Masjid': (18.9525, 72.8381),
    'Sandhurst Road': (18.9604, 72.8384),
    'Byculla': (18.9766, 72.8328),
    'Chinchpokli': (18.9830, 72.8340),
    'Currey Road': (18.9904, 72.8350),
    'Parel': (19.0069, 72.8390),
    'Matunga': (19.0261, 72.8514),
    'Sion': (19.0392, 72.8617),
    'Kurla': (19.0653, 72.8792),
    'Vidyavihar': (19.0792, 72.8970),
    'Ghatkopar': (19.0864, 72.9081),
    'Vikhroli': (19.1111, 72.9279),
    'Kanjurmarg': (19.1245, 72.9328),
    'Bhandup': (19.1442, 72.9360),
    'Nahur': (19.1557, 72.9464),
    'Mulund': (19.1718, 72.9557),
    'Thane': (19.1860, 72.9759),
    'Kalwa': (19.1996, 72.9967),
    'Mumbra': (19.1912, 73.0232),
    'Diva': (19.1884, 73.0423),
    'Kopar': (19.2081, 73.0645),
    'Dombivli': (19.2184, 73.0867),
    'Thakurli': (19.2274, 73.1070),
    'Kalyan': (19.2355, 73.1298),
    # Harbour Line
    'Vadala Road': (19.0163, 72.8593),
    'GTB Nagar': (19.0306, 72.8654),
    'Chunabhatti': (19.0494, 72.8703),
    'Tilak Nagar': (19.0682, 72.8953),
    'Chembur': (19.0620, 72.9011),
    'Govandi': (19.0558, 72.9150),
    'Mankhurd': (19.0487, 72.9320),
    'Vashi': (19.0772, 72.9984),
    'Sanpada': (19.0631, 73.0131),
    'Juinagar': (19.0531, 73.0175),
    'Nerul': (19.0332, 73.0163),
    'Seawoods-Darave': (19.0205, 73.0174),
    'Belapur CBD': (19.0189, 73.0390),
    'Kharghar': (19.0292, 73.0645),
    'Mansarovar': (19.0152, 73.0838),
    'Khandeshwar': (19.0067, 73.0975),
    'Panvel': (18.9894, 73.1175)
}

METRO_STATIONS = {
    # Line 1 (Blue)
    'Versova Metro': (19.1312, 72.8153),
    'DN Nagar Metro': (19.1250, 72.8360),
    'Azad Nagar Metro': (19.1245, 72.8407),
    'Andheri Metro': (19.1197, 72.8464),
    'WEH Metro': (19.1154, 72.8564),
    'Chakala Metro': (19.1114, 72.8604),
    'Airport Road Metro': (19.1107, 72.8653),
    'Marol Naka Metro': (19.1102, 72.8722),
    'Saki Naka Metro': (19.0964, 72.8800),
    'Asalpha Metro': (19.0984, 72.8906),
    'Jagruti Nagar Metro': (19.0924, 72.8996),
    'Ghatkopar Metro': (19.0864, 72.9081),
    
    # Line 2A (Yellow)
    'Dahisar East Metro': (19.2570, 72.8680),
    'Anand Nagar Metro': (19.2482, 72.8640),
    'Kandarpada Metro': (19.2424, 72.8576),
    'Mandapeshwar Metro': (19.2373, 72.8524),
    'Eksar Metro': (19.2323, 72.8463),
    'Borivali West Metro': (19.2272, 72.8415),
    'Pahadi Eksar Metro': (19.2201, 72.8398),
    'Kandivali West Metro': (19.2132, 72.8385),
    'Dahanukarwadi Metro': (19.2063, 72.8365),
    'Valnai Metro': (19.1983, 72.8351),
    'Malad West Metro': (19.1912, 72.8340),
    'Lower Malad Metro': (19.1834, 72.8335),
    'Pahadi Goregaon Metro': (19.1765, 72.8330),
    'Goregaon West Metro': (19.1678, 72.8327),
    'Oshiwara Metro': (19.1554, 72.8322),
    'Lower Oshiwara Metro': (19.1441, 72.8318),
    'Andheri West Metro': (19.1293, 72.8315),
    
    # Line 7 (Red)
    'Ovaripada Metro': (19.2480, 72.8672),
    'National Park Metro': (19.2355, 72.8660),
    'Devipada Metro': (19.2263, 72.8654),
    'Magathane Metro': (19.2173, 72.8647),
    'Poisar Metro': (19.2091, 72.8638),
    'Akurli Metro': (19.2005, 72.8624),
    'Kurar Metro': (19.1895, 72.8601),
    'Dindoshi Metro': (19.1772, 72.8583),
    'Aarey Metro': (19.1450, 72.8540),
    'Goregaon East Metro': (19.1645, 72.8560),
    'Jogeshwari East Metro': (19.1360, 72.8510),
    'Mogra Metro': (19.1245, 72.8480),
    'Gundavali Metro': (19.1150, 72.8570),
    
    # Line 3 (Aqua)
    'Cuffe Parade Metro': (18.9150, 72.8250),
    'Vidhan Bhavan Metro': (18.9240, 72.8240),
    'Churchgate Metro': (18.9320, 72.8270),
    'Hutatma Chowk Metro': (18.9335, 72.8315),
    'CSMT Metro': (18.9405, 72.8355),
    'Kalbadevi Metro': (18.9470, 72.8285),
    'Girgaon Metro': (18.9540, 72.8180),
    'Grant Road Metro': (18.9610, 72.8150),
    'Mumbai Central Metro': (18.9710, 72.8190),
    'Mahalaxmi Metro': (18.9830, 72.8240),
    'Science Centre Metro': (18.9925, 72.8205),
    'Acharya Atre Chowk Metro': (18.9980, 72.8180),
    'Worli Metro': (19.0030, 72.8160),
    'Siddhivinayak Metro': (19.0170, 72.8270),
    'Dadar Metro': (19.0225, 72.8440),
    'Sitaladevi Metro': (19.0350, 72.8400),
    'Dharavi Metro': (19.0470, 72.8540),
    'BKC Metro': (19.0650, 72.8690),
    'Vidyanagari Metro': (19.0740, 72.8620),
    'Santacruz Metro': (19.0830, 72.8540),
    'CSMI Airport T1 Metro': (19.0910, 72.8520),
    'Sahar Road Metro': (19.0980, 72.8670),
    'CSMI Airport T2 Metro': (19.0970, 72.8750),
    'MIDC Metro': (19.1180, 72.8760),
    'SEEPZ Metro': (19.1240, 72.8780),
    'Aarey JVLR Metro': (19.1300, 72.8850)
}

def find_nearest_transit(lat, lon, station_dict):
    min_dist = float('inf')
    nearest_name = ""
    for name, coords in station_dict.items():
        dist = haversine_distance(lat, lon, coords[0], coords[1])
        if dist < min_dist:
            min_dist = dist
            nearest_name = name
    return nearest_name, min_dist

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🔍 Building Search (Optional)")
if not df.empty and 'title' in df.columns:
    building_list = ["-- Select or Search Building --"] + df['title'].unique().tolist()
    selected_building = st.sidebar.selectbox("Choose Building", building_list)
else:
    selected_building = "-- Select or Search Building --"

# Defaults or building selection values
if selected_building != "-- Select or Search Building --" and not df.empty:
    b_data = df[df['title'] == selected_building].iloc[0]
    default_price = int(b_data['price'])
    default_area = int(b_data['area'])
    default_bhk = int(b_data['bedroom_num'])
    default_bath = int(b_data['bathroom_num'])
    default_age = int(b_data['age'])
    default_lat = float(b_data['latitude'])
    default_lon = float(b_data['longitude'])
    default_locality = str(b_data.get('locality', 'Mumbai'))
    furnished_opts = ["Unfurnished", "Semi-Furnished", "Furnished"]
    default_furnished = b_data['furnished'] if b_data['furnished'] in furnished_opts else "Unfurnished"
else:
    default_price = 12000000
    default_area = 750
    default_bhk = 2
    default_bath = 2
    default_age = 2
    default_lat = 19.0760
    default_lon = 72.8777
    default_locality = "Mumbai Suburbs"
    default_furnished = "Unfurnished"

st.sidebar.header("📍 Property Details")
price = st.sidebar.number_input("Property Price (₹)", min_value=500000, max_value=2000000000, value=default_price, step=500000)
area = st.sidebar.number_input("Carpet Area (sq. ft.)", min_value=100, max_value=10000, value=default_area, step=50)
bedroom_num = st.sidebar.slider("Bedrooms (BHK)", 1, 6, default_bhk)
bathroom_num = st.sidebar.slider("Bathrooms", 1, 6, default_bath)
age = st.sidebar.slider("Property Age (Years)", 0, 30, default_age)

furnished_opts = ["Unfurnished", "Semi-Furnished", "Furnished"]
furnished = st.sidebar.selectbox("Furnishing Status", furnished_opts, index=furnished_opts.index(default_furnished))

st.sidebar.header("🗺️ Location Coordinates")
lat = st.sidebar.number_input("Latitude", value=default_lat, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=default_lon, format="%.4f")

st.sidebar.header("⏳ User Investment Term")
holding_years = st.sidebar.slider("Investment Horizon (Years)", 1, 20, 10)

# Calculations
furnished_map = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}
furnished_score = furnished_map[furnished]

dist_bkc_km = haversine_distance(lat, lon, 19.0657, 72.8686)
dist_nariman_km = haversine_distance(lat, lon, 18.9256, 72.8242)
dist_airport_km = haversine_distance(lat, lon, 19.0896, 72.8656)
dist_nm_airport_km = haversine_distance(lat, lon, 18.9912, 73.0686)
dist_csmt_km = haversine_distance(lat, lon, 18.9400, 72.8353)

nearest_rly, rly_dist = find_nearest_transit(lat, lon, RAILWAY_STATIONS)
nearest_metro, metro_dist = find_nearest_transit(lat, lon, METRO_STATIONS)

auto_apprec_pct, auto_rent_pct, loc_tier, target_yield = get_locality_growth_rates(default_locality, dist_bkc_km, dist_nm_airport_km)

# --- MAIN DASHBOARD SECTION ---
st.subheader("🗺️ Property Location")
st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}), zoom=13)

st.markdown("---")

if st.button("Evaluate Investment & ROI Verdict", type="primary"):
    input_features = pd.DataFrame([[
        area, bedroom_num, bathroom_num, age,
        dist_bkc_km, dist_nariman_km, dist_airport_km, furnished_score
    ]], columns=['area', 'bedroom_num', 'bathroom_num', 'age', 'dist_bkc_km', 'dist_nariman_km', 'dist_airport_km', 'furnished_score'])

    pred_rent = model.predict(input_features)[0]
    pred_rent = max(1000, pred_rent)
    year1_rent = pred_rent * 12
    year1_yield = (year1_rent / price) * 100

    feasible_value = year1_rent / (target_yield / 100.0)
    valuation_diff_pct = ((feasible_value - price) / price) * 100

    g_rent = auto_rent_pct / 100.0
    g_app = auto_apprec_pct / 100.0

    cumulative_rent = sum([year1_rent * ((1 + g_rent) ** t) for t in range(holding_years)])
    future_price = price * ((1 + g_app) ** holding_years)
    total_capital_gain = future_price - price
    cagr = (((future_price + cumulative_rent) / price) ** (1 / holding_years) - 1) * 100

    st.subheader("📊 Initial Forecast (Year 1)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Est. Monthly Rent (Year 1)", f"₹{pred_rent:,.0f} / mo")
    c2.metric("Gross Rental Yield (Year 1)", f"{year1_yield:.2f}% p.a.")
    c3.metric("Holding Horizon", f"{holding_years} Years")

    st.markdown("---")

    st.subheader("⚖️ Current Deal Assessment (Is it worth buying today?)")
    if valuation_diff_pct >= 5.0:
        deal_status = "🟢 Great Deal (Undervalued)"
        deal_msg = f"The asking price is highly attractive. You are acquiring this property for less than its fair intrinsic value based on standard {target_yield:.1f}% yields for {loc_tier} properties."
    elif valuation_diff_pct <= -5.0:
        deal_status = "🔴 Overvalued (Premium)"
        deal_msg = f"The asking price carries a premium. You are paying more than the standard fair value based on typical {target_yield:.1f}% yields in {loc_tier} areas. Negotiate the price."
    else:
        deal_status = "🟡 Fairly Priced"
        deal_msg = f"The asking price accurately reflects current market dynamics, aligning perfectly with the standard {target_yield:.1f}% yield expected in {loc_tier} locations."

    v1, v2, v3 = st.columns(3)
    v1.metric("Current Asking Price", f"₹{price:,.0f}")
    v2.metric("Estimated Feasible Value", f"₹{feasible_value:,.0f}", f"{valuation_diff_pct:+.1f}% vs Asking", delta_color="normal")
    v3.metric("Market Status", deal_status)
    st.caption(deal_msg)

    st.markdown("---")
    
    st.subheader(f"📍 Auto-Fetched Micro-Market Growth Rates ({default_locality})")
    st.info(
        f"🏢 **Locality Tier:** {loc_tier}\n\n"
        f"📈 **Auto-Derived Property Appreciation:** **{auto_apprec_pct:.1f}% p.a.**\n\n"
        f"💵 **Auto-Derived Rental Escalation:** **{auto_rent_pct:.1f}% p.a.**"
    )

    st.markdown("---")

    st.subheader(f"📈 Multi-Year Projection ({holding_years}-Year Horizon)")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric(f"Cumulative Rent ({holding_years} yrs)", f"₹{cumulative_rent:,.0f}")
    p2.metric(f"Projected Property Value", f"₹{future_price:,.0f}")
    p3.metric("Total Capital Gain", f"₹{total_capital_gain:,.0f}")
    p4.metric("Total Annualized Return (CAGR)", f"{cagr:.2f}% p.a.")

    st.markdown("---")

    st.subheader("🎯 Long-Term Investment Verdict")
    if cagr >= 10.0:
        st.success(
            f"🟢 **RECOMMENDATION: HIGHLY RECOMMENDED**\n\n"
            f"Projected Annualized Return (CAGR) of **{cagr:.2f}%** over {holding_years} years significantly "
            f"outperforms standard Indian benchmarks (~7.5% FD/G-Sec rate)."
        )
    elif cagr >= 7.5:
        st.info(
            f"🟡 **RECOMMENDATION: CONSIDER / HOLD**\n\n"
            f"Projected Annualized Return (CAGR) of **{cagr:.2f}%** aligns with standard risk-free fixed income rates. "
            f"Suitable for end-use or capital preservation."
        )
    else:
        st.error(
            f"🔴 **RECOMMENDATION: AVOID / WEAK RETURNS**\n\n"
            f"Projected Annualized Return (CAGR) of **{cagr:.2f}%** falls below standard benchmark interest rates (~7.5%)."
        )

    st.markdown("---")

    st.subheader("🚆 Nearest Transit & Key Hub Accessibility")
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Nearest Railway", f"{nearest_rly}", f"{rly_dist:.1f} km away", delta_color="off")
    t2.metric("Nearest Metro", f"{nearest_metro}", f"{metro_dist:.1f} km away", delta_color="off")
    t3.metric("Mumbai Airport (CSMIA)", f"{dist_airport_km:.1f} km")
    t4.metric("Navi Mumbai Airport", f"{dist_nm_airport_km:.1f} km")
    t5.metric("CSMT Terminus", f"{dist_csmt_km:.1f} km")

    st.markdown("---")
    
    st.subheader("💡 Key Valuation Drivers")
    st.write("Breakdown of property features and landmark accessibility:")

    feature_labels = [
        f"Carpet Area ({area} sq. ft.)",
        f"Bedrooms ({bedroom_num} BHK)",
        f"Bathrooms ({bathroom_num})",
        f"Property Age ({age} years)",
        f"Proximity to BKC Hub ({dist_bkc_km:.1f} km)",
        f"Proximity to Nariman Point ({dist_nariman_km:.1f} km)",
        f"Proximity to Mumbai Airport ({dist_airport_km:.1f} km)",
        f"Furnishing Status ({furnished})"
    ]

    try:
        contribs = model.predict(input_features, pred_contrib=True)[0][:-1]
    except Exception:
        try:
            contribs = model.booster_.predict(input_features, pred_contrib=True)[0][:-1]
        except Exception:
            importances = getattr(model, 'feature_importances_', np.ones(8))
            norm_imp = importances / np.sum(importances)
            baselines = [700, 2, 2, 10, 15.0, 25.0, 12.0, 1]
            current_vals = [area, bedroom_num, bathroom_num, age, dist_bkc_km, dist_nariman_km, dist_airport_km, furnished_score]
            contribs = []
            for i in range(8):
                diff = (baselines[i] - current_vals[i]) if i in [3, 4, 5, 6] else (current_vals[i] - baselines[i])
                contribs.append(diff * norm_imp[i] * 40)

    impact_df = pd.DataFrame({
        'Label': feature_labels,
        'Impact': contribs
    }).sort_values(by='Impact', key=abs, ascending=False)

    for idx, row in impact_df.iterrows():
        lbl = row['Label']
        imp = row['Impact']
        if imp > 0:
            st.write(f"🟢 **{lbl}**: Added **+₹{abs(imp):,.0f}/month** to valuation.")
        else:
            st.write(f"🔴 **{lbl}**: Reduced valuation by **-₹{abs(imp):,.0f}/month**.")
else:
    st.info("👈 Set your investment horizon in the sidebar, then click **Evaluate Investment & ROI Verdict**.")