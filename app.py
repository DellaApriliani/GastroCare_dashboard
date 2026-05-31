# =========================================================
# GASTROCARE ANALYTICS DASHBOARD
# =========================================================

# ==============================
# IMPORT LIBRARIES
# ==============================

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

import plotly.io as pio

pio.templates.default = "plotly_white"

# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="GastroCare Dashboard",
    page_icon="🩺",
    layout="wide"
)


# ==============================
# CUSTOM CSS
# ==============================

CUSTOM_CSS = """
<style>

/* =========================================================
   GLOBAL
========================================================= */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #FFFFFF;
}

/* =========================================================
   MAIN CONTAINER
========================================================= */

.main .block-container {
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 2rem;
}

/* =========================================================
   SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #EAF4FF 0%,
        #D9F3F0 100%
    );
    border-right: 1px solid #D6EAF8;
}

/* Semua text sidebar */
section[data-testid="stSidebar"] * {
    color: #0D47A1 !important;
}

/* Sidebar title */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #1565C0 !important;
    font-weight: 700;
}

/* =========================================================
   MULTISELECT BOX
========================================================= */

/* Box utama */
div[data-baseweb="select"] > div {
    background-color: white !important;
    border: 2px solid #4DB6AC !important;
    border-radius: 14px !important;
    min-height: 52px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

/* Tag selected item */
[data-baseweb="tag"] {
    background-color: #26A69A !important;
    border-radius: 10px !important;
    padding-left: 8px !important;
    padding-right: 8px !important;
    border: none !important;
}

[data-baseweb="tag"] span {
    color: white !important;
    font-weight: 600 !important;
}

[data-baseweb="tag"] svg {
    fill: white !important;
}

section[data-testid="stSidebar"] > div {
    background: linear-gradient(
        180deg,
        #EAF4FF 0%,
        #DDF7F2 100%
    );
}


/* Dropdown */
div[role="listbox"] {
    background: white !important;
    border-radius: 12px !important;
    border: 1px solid #B2DFDB !important;
}

/* Hover dropdown */
div[role="option"]:hover {
    background-color: #E0F2F1 !important;
}

/* =========================================================
   TITLES
========================================================= */

h1 {
    color: #0D47A1 !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #1565C0 !important;
    font-weight: 700 !important;
}


/* =========================================================
   PLOTLY CLEAN
========================================================= */

[data-testid="stPlotlyChart"] {
    border-radius: 18px !important;
    overflow: hidden;
}

div[data-testid="column"] > div {
    transition: 0.3s ease;
}

div[data-testid="column"] > div:hover {
    transform: translateY(-3px);
}

/* =========================================================
   KPI CARDS PREMIUM
========================================================= */

[data-testid="metric-container"] {
    background: white;
    padding: 24px;
    border-radius: 20px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    transition: 0.3s ease;
}

[data-testid="metric-container"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
}

/* Label */
[data-testid="metric-container"] label {
    color: #6B7280 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* Value */
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0D47A1 !important;
    font-size: 36px !important;
    font-weight: 800 !important;
}

/* =========================================================
   ALERT / INSIGHT
========================================================= */

.stAlert {
    border-radius: 18px !important;
    background-color: #F0FDFA !important;
    border: 1px solid #99F6E4 !important;
}

/* =========================================================
   HORIZONTAL LINE
========================================================= */

hr {
    border: none;
    height: 1px;
    background: #E5E7EB;
}

/* =========================================================
   FOOTER
========================================================= */

footer {
    visibility: hidden;
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================
# COLOR PALETTE
# ==============================

MEDICAL_COLORS = [
    "#0D47A1",  # deep medical blue
    "#1565C0",
    "#1E88E5",
    "#42A5F5",
    "#4FC3F7",
    "#26A69A",  # teal
    "#4DB6AC",
    "#66BB6A",
    "#81C784",
    "#A5D6A7"
]

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("split_train.csv")

# ==============================
# LOAD FEATURE LIST
# ==============================

all_features = joblib.load("features_list.pkl")

# ==============================
# CONSTANTS
# ==============================

HEATMAP_FEATURES = [
    "nyeri_ulu_hati",
    "mual",
    "muntah",
    "kembung",
    "heartburn",
    "asam_naik_tenggorokan",
    "diare",
    "demam",
    "penurunan_berat_badan",
    "cepat_kenyang",
    "muntah_tak_tercerna",
    "nyeri_perut_kosong",
    "feses_hitam",
    "riwayat_nsaid",
    "stres_tinggi",
    "batuk_kronis",
    "suara_serak",
    "anemia_lemas",
    "nafsu_makan_hilang",
    "nyeri_dada"
]

EXCLUDED_FEATURES = [
    "gejala_refluks",
    "gejala_ulkus",
    "gejala_infeksi",
    "gejala_kanker",
    "gejala_motilitas",
    "gejala_fungsional",
    "total_gejala",
    "skor_gerd",
    "skor_gastritis",
    "skor_kanker",
    "skor_infeksi"
]


# ==============================
# FEATURE SELECTION
# ==============================

symptom_columns = [
    col for col in all_features
    if col not in EXCLUDED_FEATURES
]

binary_features = [
    col for col in symptom_columns
    if df[col].dropna().isin([0, 1]).all()
]


# ==============================
# HEADER SECTION
# ==============================

st.title("🩺 GastroCare Analytics Dashboard")

st.markdown("""
<div style="
background: linear-gradient(90deg,#E3F2FD,#E0F2F1);
padding: 20px;
border-radius: 18px;
margin-bottom: 20px;
border: 1px solid #D6EAF8;
">
<h3 style="color:#0D47A1;">
AI-Based Gastrointestinal Disease Classification System
</h3>

<p style="color:#374151;font-size:16px;">
Interactive analytics dashboard for gastrointestinal disease monitoring and symptom analysis.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ==============================
# SIDEBAR
# ==============================

st.sidebar.title("🩺 GastroCare")

st.sidebar.markdown("""
Healthcare Analytics Dashboard
""")

st.sidebar.header("Filter Disease")

selected_diseases = st.sidebar.multiselect(
    "Select Disease",
    options=df["label"].unique(),
    default=df["label"].unique()
)


# ==============================
# FILTER DATA
# ==============================

filtered_df = df[df["label"].isin(selected_diseases)]

if filtered_df.empty:
    st.warning("No data available.")
    st.stop()


# ==============================
# KPI CALCULATION
# ==============================

total_patients = len(filtered_df)

most_common_disease = filtered_df["label"].mode()[0]

symptom_prevalence = filtered_df[binary_features].mean()

most_common_symptom = symptom_prevalence.idxmax()


# ==============================
# KPI SECTION
# ==============================

st.subheader("📌 Dashboard Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("👥 Total Patients", total_patients)

with col2:
    st.metric("🛡️ Most Common Disease", most_common_disease)

with col3:
    st.metric("📈 Top Symptom", most_common_symptom)

st.markdown("---")

# =========================================================
# DISEASE DISTRIBUTION
# =========================================================

st.subheader("📊 Disease Distribution")

disease_distribution = (
    filtered_df["label"]
    .value_counts()
    .reset_index()
)

disease_distribution.columns = ["Disease", "Count"]

col1, col2 = st.columns(2)

col1, col2 = st.columns(
    2,
    gap="large"
)

# ==============================
# BAR CHART
# ==============================

fig_bar = px.bar(
    disease_distribution,
    x="Disease",
    y="Count",
    text_auto=True,
    color="Disease",
    color_discrete_sequence=MEDICAL_COLORS
)

fig_bar.update_layout( 
    title="Disease Distribution", 
    xaxis_title="Disease", 
    yaxis_title="Number of Patients", 
    margin=dict(
        l=20, 
        r=20, 
        t=60, 
        b=20
    )
)

with col1:
    st.plotly_chart(fig_bar, use_container_width=True)


# ==============================
# DONUT CHART
# ==============================

fig_donut = px.pie(
    disease_distribution,
    names="Disease",
    values="Count",
    hole=0.5,
    color_discrete_sequence=MEDICAL_COLORS
)

fig_donut.update_layout(
    title="Disease Percentage",
    margin=dict(
        l=20, 
        r=20, 
        t=60, 
        b=20
    )
)

with col2:
    st.plotly_chart(fig_donut, use_container_width=True)

fig_donut.update_traces(
    textposition='inside',
    textinfo='percent+label'
)

fig_donut.update_layout(
     paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(
        l=20, 
        r=20, 
        t=60, 
        b=20
    )
)

st.markdown("---")


# =========================================================
# TOP SYMPTOM PREVALENCE
# =========================================================

st.subheader("🧬 Top Symptom Prevalence")

symptom_data = (
    filtered_df[binary_features]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

symptom_data.columns = ["Symptom", "Prevalence"]

fig_symptom = px.bar(
    symptom_data.head(10),
    x="Prevalence",
    y="Symptom",
    orientation="h",
    text_auto=".2f",
    color="Prevalence",
    color_continuous_scale="GnBu"
)

fig_symptom.update_layout(
    title="Top 10 Most Common Symptoms",
    xaxis_title="Prevalence",
    yaxis_title="Symptom",
    yaxis=dict(autorange="reversed"),
    margin=dict(
        l=20, 
        r=20, 
        t=60, 
        b=20
    )
)

st.plotly_chart(fig_symptom, use_container_width=True)

st.markdown("---")


# =========================================================
# HEATMAP
# =========================================================

st.subheader("🔥 Average Symptom Intensity per Disease")

heatmap_df = filtered_df.dropna(
    subset=HEATMAP_FEATURES
)

heatmap_data = (
    heatmap_df
    .groupby("label")[HEATMAP_FEATURES]
    .mean()
    .round(2)
)

fig_heatmap = px.imshow(
    heatmap_data,
    text_auto=True,
    aspect="auto",
    color_continuous_scale=[
        "#E3F2FD",
        "#90CAF9",
        "#42A5F5",
        "#26A69A",
        "#66BB6A"
    ]
)

fig_heatmap.update_layout(
    xaxis_title="Symptoms",
    yaxis_title="Disease",
    margin=dict(
        l=20, 
        r=20, 
        t=60, 
        b=20
    )
)

fig_heatmap.update_xaxes(showgrid=False)
fig_heatmap.update_yaxes(showgrid=False)


st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown("---")



# =========================================================
# INSIGHT & RECOMMENDATION
# =========================================================

st.subheader("💡 Insight & Recommendation")

top_3_symptoms = symptom_data.head(3)["Symptom"].tolist()

st.success(f"""
### Insight Utama

- Penyakit yang paling dominan adalah **{most_common_disease}**
- Gejala yang paling sering muncul adalah **{most_common_symptom}**

### Top 3 Gejala dengan Prevalensi Tertinggi
1. {top_3_symptoms[0]}
2. {top_3_symptoms[1]}
3. {top_3_symptoms[2]}

### Rekomendasi

- Fokus pada deteksi dini berdasarkan gejala dominan
- Meningkatkan pemantauan gejala untuk pencegahan penyakit gastrointestinal
- Memanfaatkan sistem GastroCare sebagai pendukung analisis kesehatan berbasis data
""")




# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("""
GastroCare Dashboard © 2026

AI-Based Gastrointestinal Disease Classification System
""")
