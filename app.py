# =========================================================
# GASTROCARE ANALYTICS DASHBOARD
# Sistem Klasifikasi Penyakit Gastrointestinal Berbasis AI
# =========================================================

# ==============================
# IMPORT LIBRARIES
# ==============================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import joblib

# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="GastroCare Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set default Plotly template
pio.templates.default = "plotly_white"

# ==============================
# CONSTANTS
# ==============================

# Warna utama dashboard (medical blue-green)
MEDICAL_COLORS = [
    "#1565C0", "#1976D2", "#1E88E5", "#42A5F5",
    "#26A69A", "#00897B", "#4CAF50", "#66BB6A",
    "#0288D1", "#0097A7"
]

# Fitur heatmap (gejala yang ingin ditampilkan di heatmap)
HEATMAP_FEATURES = [
    "nyeri_ulu_hati", "mual", "muntah", "kembung",
    "heartburn", "asam_naik_tenggorokan", "diare",
    "demam", "penurunan_berat_badan", "cepat_kenyang",
    "muntah_tak_tercerna", "nyeri_perut_kosong",
    "feses_hitam", "riwayat_nsaid", "stres_tinggi",
    "batuk_kronis", "suara_serak", "anemia_lemas",
    "nafsu_makan_hilang", "nyeri_dada"
]

# Kolom skor/agregat yang bukan gejala individu (dikecualikan)
EXCLUDED_FEATURES = [
    "gejala_refluks", "gejala_ulkus", "gejala_infeksi",
    "gejala_kanker", "gejala_motilitas", "gejala_fungsional",
    "total_gejala", "skor_gerd", "skor_gastritis",
    "skor_kanker", "skor_infeksi"
]

# ==============================
# CUSTOM CSS
# ==============================

def load_css():
    """Inject custom CSS untuk tampilan medical profesional."""
    css = """
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    /* ---- GLOBAL ---- */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background-color: #F0F4F8 !important;
        color: #1E293B !important;
    }

    /* ---- SIDEBAR ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(160deg, #1565C0 0%, #00796B 100%) !important;
    }

    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] .stMarkdown p {
        color: #B2EBF2 !important;
        font-size: 14px;
    }

    /* Multiselect box di sidebar */
    div[data-baseweb="select"] > div {
        background-color: rgba(255,255,255,0.15) !important;
        border: 1.5px solid rgba(255,255,255,0.4) !important;
        border-radius: 10px !important;
    }

    [data-baseweb="tag"] {
        background-color: #26A69A !important;
        border-radius: 8px !important;
    }

    [data-baseweb="tag"] span {
        color: white !important;
        font-weight: 600 !important;
    }

    /* ---- MAIN CONTENT ---- */
    .main .block-container {
        padding: 1.5rem 2rem 2rem 2rem;
        max-width: 1400px;
    }

    /* ---- HEADINGS ---- */
    h1 {
        color: #1565C0 !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
    }

    h2, h3 {
        color: #1565C0 !important;
        font-weight: 700 !important;
    }

    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: #1565C0 !important;
    }

    /* ---- KPI METRIC CARDS ---- */
    [data-testid="metric-container"] {
        background: #FFFFFF !important;
        border: 1px solid #BBDEFB !important;
        border-left: 5px solid #1565C0 !important;
        border-radius: 14px !important;
        padding: 20px 24px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(21,101,192,0.12);
    }

    [data-testid="metric-container"] label {
        color: #607D8B !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    [data-testid="stMetricValue"] {
        color: #1565C0 !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }

    /* ---- CHART CONTAINER ---- */
    [data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 8px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }

    /* ---- ALERT / SUCCESS ---- */
    .stAlert {
        border-radius: 14px !important;
        border: 1px solid #B2DFDB !important;
        background-color: #E8F5E9 !important;
    }

    /* ---- HORIZONTAL RULE ---- */
    hr {
        border: none;
        height: 1px;
        background-color: #E2E8F0;
        margin: 1.5rem 0;
    }

    /* Semua teks di main area */
    .stMarkdown, .stText, p, span, div, label {
        color: #1E293B !important;
    }

    /* Hide footer */
    footer { visibility: hidden; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ==============================
# HELPER FUNCTIONS
# ==============================

def apply_chart_theme(fig):
    """Terapkan tema konsisten pada semua chart Plotly."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="DM Sans, sans-serif", color="#1E293B", size=13),
        title_font=dict(color="#1565C0", size=17, family="DM Sans, sans-serif"),
        legend=dict(font=dict(color="#1E293B")),
        margin=dict(t=50, b=30, l=20, r=20),
    )
    fig.update_xaxes(
        tickfont=dict(color="#1E293B"),
        title_font=dict(color="#1E293B"),
        gridcolor="#F1F5F9"
    )
    fig.update_yaxes(
        tickfont=dict(color="#1E293B"),
        title_font=dict(color="#1E293B"),
        gridcolor="#F1F5F9"
    )
    return fig


def get_symptom_columns(all_features, df):
    """Ambil kolom gejala binary (0/1) saja, tanpa kolom agregat."""
    symptom_cols = [col for col in all_features if col not in EXCLUDED_FEATURES]
    binary_cols = [
        col for col in symptom_cols
        if df[col].dropna().isin([0, 1]).all()
    ]
    return binary_cols


# ==============================
# LOAD DATA
# ==============================

@st.cache_data
def load_data():
    df = pd.read_csv("split_train.csv")
    all_features = joblib.load("features_list.pkl")
    return df, all_features


# ==============================
# MAIN APP
# ==============================

def main():
    # Inject CSS
    load_css()

    # Load data
    df, all_features = load_data()
    binary_features = get_symptom_columns(all_features, df)

    # ---- SIDEBAR ----
    st.sidebar.title("🩺 GastroCare")
    st.sidebar.markdown("Healthcare Analytics Dashboard")
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filter Penyakit")

    selected_diseases = st.sidebar.multiselect(
        "Pilih Penyakit",
        options=sorted(df["label"].unique()),
        default=list(df["label"].unique()),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<small style='color:#B2EBF2;'>GastroCare Dashboard © 2026<br>"
        "Sistem Klasifikasi Berbasis AI</small>",
        unsafe_allow_html=True
    )

    # ---- FILTER DATA ----
    filtered_df = df[df["label"].isin(selected_diseases)]

    if filtered_df.empty:
        st.warning("⚠️ Tidak ada data untuk filter yang dipilih.")
        st.stop()

    # ---- HEADER ----
    st.title("🩺 GastroCare Analytics Dashboard")
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #E3F2FD 0%, #E0F2F1 100%);
        padding: 18px 24px;
        border-radius: 14px;
        border: 1px solid #BBDEFB;
        margin-bottom: 8px;
    ">
        <p style="color:#1565C0; font-size:16px; font-weight:600; margin:0;">
            🏥 Sistem Klasifikasi Penyakit Gastrointestinal Berbasis AI —
            <span style="font-weight:400; color:#374151;">
                Dashboard interaktif untuk monitoring dan analisis gejala.
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ---- KPI CARDS ----
    st.subheader("📌 Ringkasan Data")

    total_patients = len(filtered_df)
    most_common_disease = filtered_df["label"].mode()[0]

    symptom_prevalence = filtered_df[binary_features].mean().sort_values(ascending=False)
    most_common_symptom = symptom_prevalence.index[0]
    top_symptom_pct = f"{symptom_prevalence.iloc[0]*100:.1f}%"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("👥 Total Pasien", f"{total_patients:,}")

    with col2:
        st.metric("🏥 Penyakit Terbanyak", most_common_disease)

    with col3:
        st.metric("📈 Gejala Teratas", most_common_symptom, top_symptom_pct)

    st.markdown("---")

    # ---- DISTRIBUSI PENYAKIT ----
    st.subheader("📊 Distribusi Penyakit")

    disease_dist = (
        filtered_df["label"]
        .value_counts()
        .reset_index()
    )
    disease_dist.columns = ["Disease", "Count"]

    col1, col2 = st.columns(2, gap="large")

    # Bar chart
    fig_bar = px.bar(
        disease_dist,
        x="Disease",
        y="Count",
        color="Disease",
        text_auto=True,
        color_discrete_sequence=MEDICAL_COLORS,
        title="Jumlah Pasien per Penyakit"
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(showlegend=False, xaxis_title="Penyakit", yaxis_title="Jumlah Pasien")
    fig_bar = apply_chart_theme(fig_bar)

    with col1:
        st.plotly_chart(fig_bar, use_container_width=True)

    # Donut chart
    fig_donut = px.pie(
        disease_dist,
        names="Disease",
        values="Count",
        hole=0.52,
        color_discrete_sequence=MEDICAL_COLORS,
        title="Proporsi Penyakit (%)"
    )
    fig_donut.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont_size=12
    )
    fig_donut = apply_chart_theme(fig_donut)

    with col2:
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")

    # ---- TOP SYMPTOM PREVALENCE ----
    st.subheader("🧬 Prevalensi Gejala Teratas")
    st.caption("Menampilkan 10 gejala dengan prevalensi tertinggi, diurutkan dari yang paling umum.")

    # Sort descending: gejala paling sering muncul di atas
    symptom_data = (
        filtered_df[binary_features]
        .mean()
        .sort_values(ascending=False)  # descending: tertinggi di atas
        .head(10)
        .reset_index()
    )
    symptom_data.columns = ["Symptom", "Prevalence"]

    # Untuk horizontal bar chart yang terbaca dengan baik:
    # sort ascending agar bar terpanjang ada di atas (Plotly horizontal bar default)
    symptom_data_plot = symptom_data.sort_values("Prevalence", ascending=True)

    fig_symptom = px.bar(
        symptom_data_plot,
        x="Prevalence",
        y="Symptom",
        orientation="h",
        text_auto=".1%",
        color="Prevalence",
        color_continuous_scale=["#B2EBF2", "#0097A7", "#1565C0"],
        title="10 Gejala dengan Prevalensi Tertinggi"
    )
    fig_symptom.update_layout(
        xaxis_title="Prevalensi",
        yaxis_title="Gejala",
        coloraxis_showscale=False,
        xaxis_tickformat=".0%"
    )
    fig_symptom = apply_chart_theme(fig_symptom)
    st.plotly_chart(fig_symptom, use_container_width=True)

    st.markdown("---")

    # ---- HEATMAP ----
    st.subheader("🔥 Rata-rata Intensitas Gejala per Penyakit")
    st.caption("Warna semakin gelap menunjukkan gejala sangat sering muncul pada penyakit tersebut.")

    # Filter hanya heatmap features yang tersedia di dataframe
    available_heatmap = [f for f in HEATMAP_FEATURES if f in filtered_df.columns]

    heatmap_data = (
        filtered_df.dropna(subset=available_heatmap)
        .groupby("label")[available_heatmap]
        .mean()
        .round(2)
    )

    fig_heatmap = px.imshow(
        heatmap_data,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=["#E3F2FD", "#90CAF9", "#1E88E5", "#26A69A", "#00796B"],
        title="Heatmap Gejala per Penyakit"
    )
    fig_heatmap.update_layout(
        xaxis_title="Gejala",
        yaxis_title="Penyakit",
        xaxis=dict(tickangle=-35)
    )
    fig_heatmap.update_xaxes(showgrid=False)
    fig_heatmap.update_yaxes(showgrid=False)
    fig_heatmap = apply_chart_theme(fig_heatmap)

    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")

    # ==============================
    # ---- METRIK EVALUASI MODEL ----
    # ==============================
    render_model_evaluation()
 
 

    # ---- INSIGHT & REKOMENDASI ----
    st.subheader("💡 Insight & Rekomendasi")

    top3 = symptom_data["Symptom"].head(3).tolist()

    st.success(f"""
**📊 Insight Utama**

- Penyakit yang paling dominan dalam dataset ini adalah **{most_common_disease}**.
- Gejala yang paling sering muncul adalah **{most_common_symptom}** dengan prevalensi **{top_symptom_pct}**.

**🧬 Top 3 Gejala dengan Prevalensi Tertinggi**
1. {top3[0]}
2. {top3[1]}
3. {top3[2]}

**📋 Rekomendasi**
- Fokus pada deteksi dini berdasarkan gejala dominan yang teridentifikasi.
- Tingkatkan pemantauan terhadap gejala-gejala dengan prevalensi tinggi untuk pencegahan lebih awal.
- Manfaatkan sistem GastroCare sebagai alat bantu analisis kesehatan berbasis data.
    """)

    # ---- FOOTER ----
    st.markdown("---")
    st.caption("GastroCare Dashboard © 2026 — Sistem Klasifikasi Penyakit Gastrointestinal Berbasis AI")


# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    main()
