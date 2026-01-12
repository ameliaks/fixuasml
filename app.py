import streamlit as st
import pandas as pd
import numpy as np

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Prediksi Gizi Anak",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# SESSION STATE
# ================================
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Home"

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ================================
# THEME HANDLER
# ================================
if st.session_state.dark_mode:
    bg = "#0F172A"
    card = "rgba(30,41,59,0.9)"
    text = "#E5E7EB"
    hero = "linear-gradient(120deg, #0F766E, #115E59)"
else:
    bg = "#ECFDFD"
    card = "rgba(255,255,255,0.75)"
    text = "#1F2937"
    hero = "linear-gradient(120deg, #1ABC9C, #148F77)"

# ================================
# CSS PREMIUM + ANIMATION
# ================================
st.markdown(f"""
<style>
.main {{
    background: {bg};
    color: {text};
}}

.fade {{
    animation: fadeIn 1s ease-in-out;
}}

.slide {{
    animation: slideUp 0.9s ease;
}}

@keyframes fadeIn {{
    from {{opacity: 0}}
    to {{opacity: 1}}
}}

@keyframes slideUp {{
    from {{transform: translateY(40px); opacity:0}}
    to {{transform: translateY(0); opacity:1}}
}}

.hero {{
    background: {hero};
    padding: 4rem;
    border-radius: 30px;
    color: white;
    box-shadow: 0 25px 60px rgba(0,0,0,0.3);
}}

.card {{
    background: {card};
    backdrop-filter: blur(16px);
    border-radius: 22px;
    padding: 2rem;
    box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    transition: 0.3s;
}}

.card:hover {{
    transform: translateY(-8px);
}}

.stButton > button {{
    border-radius: 25px;
    font-weight: bold;
    padding: 0.6rem 1.6rem;
}}

section[data-testid="stSidebar"] {{
    background: {hero};
    color: white;
}}
</style>
""", unsafe_allow_html=True)

# ================================
# SIDEBAR
# ================================
with st.sidebar:

    if st.button("🏠 Home"):
        st.session_state.active_tab = "Home"
    if st.button("📂 Dataset"):
        st.session_state.active_tab = "Dataset"
    if st.button("📊 Dashboard"):
        st.session_state.active_tab = "Dashboard"
    if st.button("🤖 Machine Learning"):
        st.session_state.active_tab = "ML"
    if st.button("🔮 Prediction"):
        st.session_state.active_tab = "Prediction"
    if st.button("📞 Contact"):
        st.session_state.active_tab = "Contact"

    st.divider()
    st.toggle("🌙 Dark Mode", key="dark_mode")

    st.markdown("—")
    st.markdown("👩‍🎓 **Amelia Kusuma Wardani**")
    st.markdown("🎓 S1 Sains Data • UNIMUS")

# ================================
# HOME
# ================================
if st.session_state.active_tab == "Home":
    st.markdown("""
    <div class="hero fade">
        <h1>Prediksi Status Gizi Anak</h1>
        <h4>
            Sistem Prediksi Status Gizi Anak<br>
            Berbasis <b>Machine Learning</b>
        </h4>
        <p>
            Project UAS Machine Learning<br>
            Universitas Muhammadiyah Semarang
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card slide">
            <h3>Dataset</h3>
            <p>Informasi dataset.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Dataset"):
            st.session_state.active_tab = "Dataset"

    with c2:
        st.markdown("""
        <div class="card slide">
            <h3>Dashboard</h3>
            <p>Insight & Visualisasi interaktif.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Dashboard"):
            st.session_state.active_tab = "Dashboard"

    with c3:
        st.markdown("""
        <div class="card slide">
            <h3>Prediction</h3>
            <p>Prediksi status gizi anak.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Try Prediction"):
            st.session_state.active_tab = "Prediction"

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Data", "1.200+")
    m2.metric("Akurasi", "90%+")
    m3.metric("Kelas", "3")

# ================================
# DATASET
# ================================
elif st.session_state.active_tab == "Dataset":
    st.markdown("## 📂 About Dataset")
    st.markdown("""
    <div class="card fade">
    Dataset kesehatan anak untuk klasifikasi status gizi:
    <b>Normal, At Risk, Malnourished</b>.
    </div>
    """, unsafe_allow_html=True)

    import about
    about.about_dataset()

# ================================
# DASHBOARD
# ================================
elif st.session_state.active_tab == "Dashboard":
    st.markdown("## 📊 Dashboard Analisis")
    st.markdown("""
    <div class="card fade">
    Visualisasi data untuk memahami pola status gizi anak.
    </div>
    """, unsafe_allow_html=True)

    import visualisasi
    visualisasi.chart()

# ================================
# MACHINE LEARNING
# ================================
elif st.session_state.active_tab == "ML":
    st.markdown("## 🤖 Machine Learning")
    st.markdown("""
    <div class="card fade">
    Perbandingan performa model klasifikasi status gizi.
    </div>
    """, unsafe_allow_html=True)

    import machine_learning
    machine_learning.ml_model()

# ================================
# PREDICTION
# ================================
elif st.session_state.active_tab == "Prediction":
    st.markdown("## 🔮 Prediksi Status Gizi Anak")
    st.markdown("""
    <div class="card fade">
    Masukkan data kesehatan anak untuk mendapatkan hasil prediksi.
    </div>
    """, unsafe_allow_html=True)

    import prediction
    prediction.prediction_app()

# ================================
# CONTACT
# ================================
elif st.session_state.active_tab == "Contact":
    st.markdown("## 📞 Contact")
    st.markdown("""
    <div class="card fade">
    Jika ada pertanyaan atau diskusi, silakan hubungi saya.
    </div>
    """, unsafe_allow_html=True)

    import contact
    contact.contact_app()

# ================================
# FOOTER
# ================================
st.divider()
st.markdown("""
<p style="text-align:center;opacity:0.7;">
© 2025 • Project UAS Machine Learning<br>
Amelia Kusuma Wardani
</p>
""", unsafe_allow_html=True)
