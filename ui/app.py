
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import streamlit as st

st.set_page_config(page_title="Voyage — Travel ML", layout="wide")

with open("ui/styles.css","r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

API_BASE = os.environ.get("API_BASE", "http://voyage:8080")

@st.cache_data(ttl=30)
def get_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

@st.cache_data
def load_csvs():
    flights = hotels = None
    try:
        flights = pd.read_csv("./data/flights.csv")
    except Exception:
        pass
    try:
        hotels = pd.read_csv("./data/hotels.csv")
    except Exception:
        pass
    return flights, hotels

def call_api(path, payload):
    url = f"{API_BASE}{path}"
    r = requests.post(url, json=payload, timeout=15)
    return r

def kpi(value, label, delta=None, help=None):
    st.markdown('<div class="voy-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1,3])
    with col1:
        st.metric(label, value, delta=delta, help=help)
    with col2:
        st.caption(datetime.utcnow().strftime("UTC %Y-%m-%d %H:%M:%S"))
    st.markdown('</div>', unsafe_allow_html=True)

h = get_health()
ok = h.get("ok", False) and h.get("has_flight_model") is not False
badge = '<span class="voy-badge">API ONLINE</span>' if ok else '<span class="voy-badge danger">API OFFLINE</span>'

st.markdown(f'''
<div class="voy-gradient">
  <h1>Voyage Analytics</h1>
  <p class="small">Fast estimates for flights & hotels, backed by ML models. {badge}</p>
</div>
''', unsafe_allow_html=True)

flights_df, hotels_df = load_csvs()

with st.sidebar:
    st.subheader("Server")
    st.text(f"API: {API_BASE}")
    st.json(h, expanded=False)
    st.divider()
    st.caption("Data snapshots")
    if flights_df is not None:
        st.caption(f"Flights rows: {len(flights_df):,}")
    if hotels_df is not None:
        st.caption(f"Hotels rows: {len(hotels_df):,}")

tab1, tab2, tab3 = st.tabs(["Flight Price", "Hotel Price", "Batch (CSV)"])

with tab1:
    st.subheader("Flight Price Estimator")
    colA, colB, colC = st.columns(3)
    from_opts = sorted(flights_df["from"].astype(str).str.lower().unique()) if (flights_df is not None and "from" in flights_df) else []
    to_opts   = sorted(flights_df["to"].astype(str).str.lower().unique())   if (flights_df is not None and "to" in flights_df)   else []
    ftype_opts= sorted(flights_df["flightType"].astype(str).str.lower().unique()) if (flights_df is not None and "flightType" in flights_df) else []
    agency_opts=sorted(flights_df["agency"].astype(str).str.lower().unique()) if (flights_df is not None and "agency" in flights_df) else []

    with colA:
        from_city = st.selectbox("From", from_opts, index=from_opts.index("del") if "del" in from_opts else 0) if from_opts else st.text_input("From", "del")
        flight_type = st.selectbox("Flight Type", ftype_opts, index=0) if ftype_opts else st.text_input("Flight Type", "economy")
    with colB:
        to_city = st.selectbox("To", to_opts, index=to_opts.index("bom") if "bom" in to_opts else 0) if to_opts else st.text_input("To", "bom")
        agency = st.selectbox("Agency", agency_opts, index=0) if agency_opts else st.text_input("Agency", "sample_air")
    with colC:
        if flights_df is not None and "distance" in flights_df:
            default_dist = float(pd.to_numeric(flights_df["distance"], errors="coerce").dropna().median())
        else:
            default_dist = 1400.0
        distance = st.number_input("Distance (km, approx)", min_value=0.0, value=default_dist, step=10.0)

    c1, c2, _ = st.columns([1,1,2])
    with c1:
        time_str = st.text_input("Flight Time (HH:MM)", "2:10")
    with c2:
        date = st.date_input("Date", datetime.utcnow().date() + timedelta(days=14))

    if "recent_flights" not in st.session_state: st.session_state.recent_flights = []

    if st.button("Estimate Flight Price", type="primary"):
        payload = {
            "from": from_city,
            "to": to_city,
            "flightType": flight_type,
            "agency": agency,
            "distance": float(distance),
            "time": time_str,
            "date": str(date)
        }
        with st.spinner("Calling API..."):
            r = call_api("/predict/flight_price", payload)
        if r.ok:
            price = r.json().get("estimated_price")
            kpi(f"₹ {price:,.2f}", "Estimated Flight Price")
            st.code(payload, language="json")
            st.session_state.recent_flights.insert(0, {"ts": datetime.utcnow().isoformat(timespec="seconds"), **payload, "estimated_price": price})
        else:
            st.error(f"API error {r.status_code}: {r.text}")
            st.code(payload, language="json")

    if st.session_state.recent_flights:
        st.markdown("#### Recent Flight Estimates")
        st.dataframe(pd.DataFrame(st.session_state.recent_flights))

with tab2:
    st.subheader("Hotel Nightly & Total")
    colA, colB, colC = st.columns(3)
    places = sorted(hotels_df["place"].astype(str).str.lower().unique()) if (hotels_df is not None and "place" in hotels_df) else []
    with colA:
        place = st.selectbox("Place", places, index=places.index("mumbai") if "mumbai" in places else 0) if places else st.text_input("Place", "mumbai")
    with colB:
        days = st.number_input("Days", min_value=1, max_value=60, value=3, step=1)
    with colC:
        date_h = st.date_input("Check-in Date", datetime.utcnow().date() + timedelta(days=14))

    name = st.text_input("Hotel Name (optional)", "placeholder hotel")

    if "recent_hotels" not in st.session_state: st.session_state.recent_hotels = []

    if st.button("Estimate Hotel Price", type="primary", key="hotel"):
        payload = {"place": place, "days": int(days), "date": str(date_h), "name": name}
        with st.spinner("Calling API..."):
            r = call_api("/predict/hotel_price", payload)
        if r.ok:
            js = r.json()
            c1, c2 = st.columns(2)
            with c1: kpi(f"₹ {js['nightly_price']:,.2f}", "Nightly Rate")
            with c2: kpi(f"₹ {js['hotel_total']:,.2f}", "Trip Total")
            st.code(payload, language="json")
            st.session_state.recent_hotels.insert(0, {"ts": datetime.utcnow().isoformat(timespec="seconds"), **payload, **js})
        else:
            st.error(f"API error {r.status_code}: {r.text}")
            st.code(payload, language="json")

    if st.session_state.recent_hotels:
        st.markdown("#### Recent Hotel Estimates")
        st.dataframe(pd.DataFrame(st.session_state.recent_hotels))

with tab3:
    st.subheader("Batch Estimates from CSV")
    st.caption("Upload a CSV for flights with columns: from, to, flightType, agency, distance, time, date")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        if st.button("Run Batch", type="primary", key="batch"):
            results = []
            prog = st.progress(0)
            for i, row in df.iterrows():
                payload = {k: row.get(k) for k in ["from","to","flightType","agency","distance","time","date"]}
                payload["distance"] = None if pd.isna(payload.get("distance")) else float(payload["distance"])
                r = call_api("/predict/flight_price", payload)
                price = r.json().get("estimated_price") if r.ok else np.nan
                results.append(price)
                prog.progress(int((i+1)/len(df)*100))
            df["estimated_price"] = results
            st.success("Batch complete")
            st.dataframe(df)
            st.download_button("Download Results CSV", df.to_csv(index=False).encode("utf-8"),
                               file_name="flight_batch_results.csv", mime="text/csv")
