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

.main {
    background-color: #F5F7FA;
}

h1 {
    color: #2F80ED;
    font-weight: bold;
}

h2, h3 {
    color: #1F3B5B;
}

[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #E5E7EB;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)



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
### AI-Based Gastrointestinal Disease Classification System

Interactive analytics dashboard for gastrointestinal disease monitoring and symptom analysis.
""")

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
    st.metric("Total Patients", total_patients)

with col2:
    st.metric("Most Common Disease", most_common_disease)

with col3:
    st.metric("Top Symptom", most_common_symptom)

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


# ==============================
# BAR CHART
# ==============================

fig_bar = px.bar(
    disease_distribution,
    x="Disease",
    y="Count",
    text_auto=True,
    color="Disease",
    color_discrete_sequence=px.colors.sequential.Blues_r
)

fig_bar.update_layout(
    title="Disease Distribution",
    xaxis_title="Disease",
    yaxis_title="Number of Patients"
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
    color_discrete_sequence=px.colors.sequential.Blues_r
)

fig_donut.update_layout(
    title="Disease Percentage"
)

with col2:
    st.plotly_chart(fig_donut, use_container_width=True)

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
    color_continuous_scale="Teal"
)

fig_symptom.update_layout(
    title="Top 10 Most Common Symptoms",
    xaxis_title="Prevalence",
    yaxis_title="Symptom",
    yaxis=dict(autorange="reversed")
)

st.plotly_chart(fig_symptom, use_container_width=True)

st.markdown("---")


# =========================================================
# HEATMAP
# =========================================================

st.subheader("🔥 Average Symptom Intensity per Disease")

heatmap_df = df.dropna(subset=HEATMAP_FEATURES)

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
    color_continuous_scale="RdYlGn"
)

fig_heatmap.update_layout(
    xaxis_title="Symptoms",
    yaxis_title="Disease"
)

st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown("---")



# =========================================================
# INSIGHT & RECOMMENDATION
# =========================================================

st.subheader("💡 Insight & Recommendation")

top_3_symptoms = symptom_data.head(3)["Symptom"].tolist()

st.info(f"""
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