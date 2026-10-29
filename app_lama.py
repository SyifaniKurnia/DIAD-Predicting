import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import shap
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  KONFIGURASI HALAMAN
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DIAD Predictor | Prediksi Autoimunitas Akibat Obat",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  KONSTANTA
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR     = "model"
MODEL_PATH    = os.path.join(MODEL_DIR, "xgb_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "selected_features.pkl")
METRICS_PATH  = os.path.join(MODEL_DIR, "model_metrics.pkl")
SHAP_PATH     = os.path.join(MODEL_DIR, "shap_values.pkl")
SCALER_PATH   = os.path.join(MODEL_DIR, "scaler.pkl")

# Versi cache — bump untuk paksa reload
MODEL_VERSION = "v10_xgb_mi_8020_holdout"
SHAP_VERSION  = "v10_xgb_mi_8020_holdout_shap"

TRAIN_DATA    = "dataset/DIA_trainingset_RDKit_descriptors.csv"
TEST_DATA     = "dataset/DIA_testset_RDKit_descriptors.csv"
XLSX_DATA     = "dataset/RDKit_ChemDes.xlsx"

# Jumlah fitur
N_FEATURES_TOTAL = 196   # total deskriptor RDKit (setelah drop SMILES/Label)
N_FEATURES_MI    = 94    # fitur tersisa setelah Mutual Information (MI > 0.01)

TARGET    = "Label"
DROP_COLS = ["Label", "SMILES"]
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  DATA PERBANDINGAN MODEL
#  Sumber: 01_Validation_Strategy.ipynb — Skenario: Full Data 80:20 Holdout
#  15 kombinasi: 3 Model × 5 Teknik Seleksi Fitur
# ─────────────────────────────────────────────────────────────────────────────
COMPARISON_DATA = pd.DataFrame([
    # ── SVM ───────────────────────────────────────────────────────────────────
    {"Model": "SVM", "Seleksi Fitur": "Baseline",            "Jumlah Fitur": 196, "Akurasi": 0.7833, "MCC": 0.3612, "F1-Score": 0.6724, "ROC-AUC": 0.7761},
    {"Model": "SVM", "Seleksi Fitur": "Variance Threshold",  "Jumlah Fitur": 169, "Akurasi": 0.7917, "MCC": 0.3894, "F1-Score": 0.6841, "ROC-AUC": 0.7803},
    {"Model": "SVM", "Seleksi Fitur": "Zero Dominant",       "Jumlah Fitur": 111, "Akurasi": 0.7417, "MCC": 0.3704, "F1-Score": 0.6790, "ROC-AUC": 0.7582},
    {"Model": "SVM", "Seleksi Fitur": "Spearman",            "Jumlah Fitur": 147, "Akurasi": 0.7750, "MCC": 0.3278, "F1-Score": 0.6455, "ROC-AUC": 0.7834},
    {"Model": "SVM", "Seleksi Fitur": "Mutual Information",  "Jumlah Fitur":  94, "Akurasi": 0.7667, "MCC": 0.3312, "F1-Score": 0.6601, "ROC-AUC": 0.7608},
    # ── Random Forest ─────────────────────────────────────────────────────────
    {"Model": "Random Forest", "Seleksi Fitur": "Baseline",           "Jumlah Fitur": 196, "Akurasi": 0.8250, "MCC": 0.4851, "F1-Score": 0.6994, "ROC-AUC": 0.8294},
    {"Model": "Random Forest", "Seleksi Fitur": "Variance Threshold", "Jumlah Fitur": 169, "Akurasi": 0.8167, "MCC": 0.4681, "F1-Score": 0.6823, "ROC-AUC": 0.8156},
    {"Model": "Random Forest", "Seleksi Fitur": "Zero Dominant",      "Jumlah Fitur": 111, "Akurasi": 0.8083, "MCC": 0.4573, "F1-Score": 0.6789, "ROC-AUC": 0.8123},
    {"Model": "Random Forest", "Seleksi Fitur": "Spearman",           "Jumlah Fitur": 147, "Akurasi": 0.8167, "MCC": 0.4621, "F1-Score": 0.6731, "ROC-AUC": 0.8178},
    {"Model": "Random Forest", "Seleksi Fitur": "Mutual Information", "Jumlah Fitur":  94, "Akurasi": 0.8083, "MCC": 0.4534, "F1-Score": 0.6648, "ROC-AUC": 0.8098},
    # ── XGBoost ───────────────────────────────────────────────────────────────
    {"Model": "XGBoost", "Seleksi Fitur": "Baseline",            "Jumlah Fitur": 196, "Akurasi": 0.8333, "MCC": 0.5247, "F1-Score": 0.7530, "ROC-AUC": 0.8278},
    {"Model": "XGBoost", "Seleksi Fitur": "Variance Threshold",  "Jumlah Fitur": 169, "Akurasi": 0.8250, "MCC": 0.5072, "F1-Score": 0.7435, "ROC-AUC": 0.8261},
    {"Model": "XGBoost", "Seleksi Fitur": "Zero Dominant",       "Jumlah Fitur": 111, "Akurasi": 0.8333, "MCC": 0.6265, "F1-Score": 0.7643, "ROC-AUC": 0.9026},
    {"Model": "XGBoost", "Seleksi Fitur": "Spearman",            "Jumlah Fitur": 147, "Akurasi": 0.8083, "MCC": 0.4529, "F1-Score": 0.7143, "ROC-AUC": 0.7981},
    {"Model": "XGBoost", "Seleksi Fitur": "Mutual Information",  "Jumlah Fitur":  94, "Akurasi": 0.8583, "MCC": 0.6181, "F1-Score": 0.8090, "ROC-AUC": 0.8967},  # ← TERBAIK
])

# Tabel ringkasan seleksi fitur
FS_TABLE = pd.DataFrame([
    {"Metode": "Baseline",           "Fitur Terpilih": 196, "Total": 196, "Rasio": "100.0%", "Keterangan": "Semua deskriptor RDKit digunakan tanpa seleksi"},
    {"Metode": "Variance Threshold", "Fitur Terpilih": 169, "Total": 196, "Rasio": "86.2%",  "Keterangan": "Eliminasi fitur dengan variance ≤ 0.01 (pada data training)"},
    {"Metode": "Zero Dominant",      "Fitur Terpilih": 111, "Total": 196, "Rasio": "56.6%",  "Keterangan": "Eliminasi fitur dengan nilai dominan > 80%"},
    {"Metode": "Spearman",           "Fitur Terpilih": 147, "Total": 196, "Rasio": "75.0%",  "Keterangan": "Eliminasi fitur berkorelasi Spearman > 0.85 (murni tanpa VT)"},
    {"Metode": "Mutual Information", "Fitur Terpilih":  94, "Total": 196, "Rasio": "48.0%",  "Keterangan": "Pertahankan fitur dengan MI Score > 0.01 terhadap label ★ TERBAIK"},
])

# ─────────────────────────────────────────────────────────────────────────────
#  WARNA TEMA
# ─────────────────────────────────────────────────────────────────────────────
C_BG      = "#06081A"
C_CARD    = "#0C0F26"
C_CARD2   = "#111435"
C_BORDER  = "#1C2050"
C_BORDER2 = "#2A1F60"
C_BLUE    = "#5B72F5"
C_VIOLET  = "#8B63F0"
C_PINK    = "#C56AD4"
C_TEXT    = "#DDE3FF"
C_TEXT2   = "#9CA8E8"
C_TEXT3   = "#6672B8"
C_GRAD    = "linear-gradient(135deg, #5B72F5 0%, #8B63F0 50%, #C56AD4 100%)"

# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; background-color: {C_BG}; color: {C_TEXT}; }}
.main .block-container {{ padding: 1.5rem 2.5rem 2rem 2.5rem; max-width: 1300px; }}
[data-testid="collapsedControl"] {{ display: none; }}
[data-testid="stSidebar"] {{ display: none; }}
.hero-wrap {{
    background: linear-gradient(135deg, #080B28 0%, #130A30 40%, #1C0D3A 70%, #0A0820 100%);
    border-radius: 22px; padding: 2.8rem 3.5rem; margin-bottom: 1.6rem;
    position: relative; overflow: hidden; border: 1px solid rgba(91,114,245,0.22);
    box-shadow: 0 8px 40px rgba(91,114,245,0.18), 0 0 80px rgba(139,99,240,0.10);
}}
.hero-wrap::before {{
    content:''; position:absolute; inset:0;
    background: radial-gradient(ellipse at 78% 40%, rgba(139,99,240,0.18) 0%, transparent 55%),
                radial-gradient(ellipse at 20% 80%, rgba(91,114,245,0.12) 0%, transparent 50%);
}}
.hero-wrap::after {{
    content:''; position:absolute; top:-80px; right:-80px; width:380px; height:380px;
    border-radius:50%; background: radial-gradient(circle, rgba(197,106,212,0.14) 0%, transparent 65%);
}}
.hero-badge {{
    display:inline-flex; align-items:center; gap:0.4rem;
    background: rgba(91,114,245,0.15); backdrop-filter:blur(10px);
    border:1px solid rgba(91,114,245,0.35); border-radius:50px;
    padding:0.28rem 1rem; font-size:0.72rem; font-weight:600;
    color:#A5B4FC; letter-spacing:0.10em; margin-bottom:1rem; position:relative; z-index:1;
}}
.hero-title {{
    font-family:'Space Grotesk',sans-serif; font-size:2.6rem; font-weight:700;
    background: linear-gradient(90deg,#FFFFFF 0%,#C4B5FD 60%,#F0ABDC 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin:0 0 0.5rem 0; line-height:1.15; position:relative; z-index:1;
}}
.hero-sub {{ font-size:0.98rem; color:#8896D8; max-width:600px; line-height:1.65; position:relative; z-index:1; }}
.mrow {{ display:flex; gap:1rem; margin-bottom:1.8rem; flex-wrap:wrap; }}
.mcard {{
    flex:1; min-width:140px;
    background: linear-gradient(145deg, {C_CARD} 0%, {C_CARD2} 100%);
    border:1px solid {C_BORDER}; border-radius:16px; padding:1.2rem 1.4rem;
    position:relative; overflow:hidden; box-shadow:0 4px 20px rgba(0,0,0,0.35);
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
}}
.mcard:hover {{ transform:translateY(-3px); box-shadow:0 8px 30px rgba(91,114,245,0.25); border-color:rgba(91,114,245,0.45); }}
.mcard::before {{ content:''; position:absolute; top:0; left:0; right:0; height:3px; background: {C_GRAD}; }}
.mlabel {{ font-size:0.66rem; font-weight:700; letter-spacing:0.12em; color:{C_TEXT3}; text-transform:uppercase; margin-bottom:0.45rem; }}
.mvalue {{ font-family:'Space Grotesk',sans-serif; font-size:1.55rem; font-weight:700; color:#FFFFFF; line-height:1; white-space:nowrap; }}
.msub {{ font-size:0.68rem; color:{C_TEXT3}; margin-top:0.35rem; }}
[data-testid="stTabs"] {{ margin-top:0.5rem; }}
button[data-baseweb="tab"] {{
    font-family:'Space Grotesk',sans-serif !important; font-size:0.90rem !important;
    font-weight:600 !important; color:{C_TEXT2} !important; background:transparent !important;
    border:none !important; padding:0.7rem 1.4rem !important; border-radius:10px 10px 0 0 !important;
    transition: color 0.2s !important;
}}
button[data-baseweb="tab"]:hover {{ color:{C_TEXT} !important; }}
button[data-baseweb="tab"][aria-selected="true"] {{
    color:#FFFFFF !important;
    background: linear-gradient(135deg, rgba(91,114,245,0.20), rgba(139,99,240,0.15)) !important;
    border-bottom:2px solid {C_VIOLET} !important;
}}
[data-testid="stTabsContent"] {{
    background: {C_CARD}; border:1px solid {C_BORDER};
    border-radius:0 16px 16px 16px; padding:2rem 2rem;
}}
[data-testid="stFileUploadDropzone"] {{
    background: linear-gradient(160deg, #0D0F28 0%, #130B30 100%) !important;
    border: 2px dashed {C_BORDER2} !important; border-radius: 18px !important;
    min-height: 130px !important; transition: border-color 0.3s, box-shadow 0.3s !important;
}}
[data-testid="stFileUploadDropzone"]:hover {{
    border-color: {C_VIOLET} !important; box-shadow: 0 0 25px rgba(139,99,240,0.20) !important;
}}
[data-testid="stFileUploadDropzone"] * {{ color:{C_TEXT2} !important; }}
[data-testid="stFileUploadDropzone"] button {{
    background: linear-gradient(135deg, {C_BLUE}, {C_VIOLET}) !important;
    color:#FFFFFF !important; border:none !important; border-radius:10px !important;
    padding:0.5rem 1.5rem !important; font-weight:600 !important;
    box-shadow: 0 4px 15px rgba(91,114,245,0.35) !important;
}}
.icard {{
    background: linear-gradient(145deg, #0E1130 0%, #141840 100%);
    border: 1px solid {C_BORDER}; border-radius:16px; padding:1.4rem 1.6rem;
    margin-bottom:1rem; box-shadow:0 3px 16px rgba(0,0,0,0.3);
    transition: border-color 0.2s, box-shadow 0.2s;
}}
.icard:hover {{ border-color:rgba(91,114,245,0.4); box-shadow:0 6px 24px rgba(91,114,245,0.15); }}
.icard-title {{
    font-family:'Space Grotesk',sans-serif; font-size:0.95rem; font-weight:600;
    color:{C_BLUE}; margin-bottom:0.7rem; display:flex; align-items:center; gap:0.5rem;
}}
.icard p {{ color:{C_TEXT2}; font-size:0.85rem; margin:0.25rem 0; line-height:1.6; }}
.res-diad {{
    background: linear-gradient(135deg, #2C0914 0%, #3D1020 100%);
    border:2px solid #E05577; border-radius:18px; padding:1.8rem 2rem;
    text-align:center; animation:glow-red 2.5s ease-in-out infinite;
}}
@keyframes glow-red {{ 0%,100% {{ box-shadow:0 0 25px rgba(224,85,119,0.20); }} 50% {{ box-shadow:0 0 50px rgba(224,85,119,0.40); }} }}
.res-safe {{
    background: linear-gradient(135deg, #10082E 0%, #180C40 100%);
    border:2px solid {C_VIOLET}; border-radius:18px; padding:1.8rem 2rem;
    text-align:center; animation:glow-violet 2.5s ease-in-out infinite;
}}
@keyframes glow-violet {{ 0%,100% {{ box-shadow:0 0 25px rgba(139,99,240,0.20); }} 50% {{ box-shadow:0 0 50px rgba(139,99,240,0.40); }} }}
.res-icon {{ font-size:3rem; margin-bottom:0.4rem; }}
.res-label {{ font-family:'Space Grotesk',sans-serif; font-size:1.9rem; font-weight:800; margin:0.3rem 0; }}
.res-desc {{ font-size:0.86rem; color:{C_TEXT2}; margin-top:0.5rem; line-height:1.6; }}
.pbar {{ margin:0.8rem 0; }}
.pbar-lbl {{ display:flex; justify-content:space-between; font-size:0.78rem; color:{C_TEXT2}; margin-bottom:0.3rem; }}
.pbar-bg {{ background:#141830; border-radius:50px; height:10px; overflow:hidden; }}
.pbar-diad {{ height:100%; border-radius:50px; background:linear-gradient(90deg,#E05577,#FF7DA0); transition:width 0.8s ease; }}
.pbar-safe {{ height:100%; border-radius:50px; background: {C_GRAD}; transition:width 0.8s ease; }}
.sec-title {{
    font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:700;
    color:{C_VIOLET}; border-left:4px solid {C_BLUE}; padding-left:0.8rem; margin:1.6rem 0 1rem 0;
}}
.styled-table {{ width:100%; border-collapse:collapse; font-size:0.85rem; margin:1rem 0; }}
.styled-table th {{
    background: linear-gradient(90deg, rgba(91,114,245,0.25), rgba(139,99,240,0.20));
    color:{C_TEXT}; font-weight:600; padding:0.75rem 1rem;
    text-align:left; border-bottom:2px solid {C_BORDER};
}}
.styled-table td {{ padding:0.65rem 1rem; color:{C_TEXT2}; border-bottom:1px solid {C_BORDER}; }}
.styled-table tr:hover td {{ background:rgba(91,114,245,0.06); }}
.badge-best {{
    background: linear-gradient(135deg,#5B72F5,#8B63F0);
    color:#FFF; font-size:0.68rem; font-weight:700;
    padding:0.15rem 0.55rem; border-radius:50px; letter-spacing:0.05em;
}}
.badge-fs {{
    background:rgba(197,106,212,0.20); color:#D4A0E5; font-size:0.70rem; font-weight:600;
    padding:0.15rem 0.5rem; border-radius:50px; border:1px solid rgba(197,106,212,0.35);
}}
[data-testid="stDataFrame"] {{ border-radius:12px; overflow:hidden; }}
[data-testid="stExpander"] {{
    background:{C_CARD}; border:1px solid {C_BORDER} !important; border-radius:14px !important; margin-bottom:0.7rem;
}}
summary {{ color:{C_TEXT} !important; font-weight:500 !important; }}
::-webkit-scrollbar {{ width:6px; height:6px; }}
::-webkit-scrollbar-track {{ background:{C_BG}; }}
::-webkit-scrollbar-thumb {{ background:{C_BORDER}; border-radius:3px; }}
::-webkit-scrollbar-thumb:hover {{ background:{C_BLUE}; }}
hr {{ border-color:{C_BORDER} !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  FUNGSI UTILITAS
# ─────────────────────────────────────────────────────────────────────────────
def fix_european(df):
    """Bersihkan format angka Eropa (koma → titik) dan konversi ke numerik."""
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  MUAT MODEL DARI PKL (diekspor dari NB02)
#  Tidak melatih ulang — wajib jalankan cell ekspor di NB02 terlebih dahulu.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_or_load_model():
    """
    Memuat model dan semua artefak dari file pkl yang diekspor dari NB02.

    File yang dibutuhkan (semua di folder model/):
      - xgb_model.pkl         → XGBClassifier dengan HP hasil GridSearchCV
      - selected_features.pkl → list nama fitur yang lolos MI (94 fitur)
      - model_metrics.pkl     → dict metrik: MCC, ROC-AUC, Accuracy, F1
      - scaler.pkl            → StandardScaler yang di-fit pada Xtr80 raw (196 fitur)

    Alur prediksi:
      input (196 fitur) → scaler → pilih selected_features → xgb_model → output

    Cara menghasilkan file ini: jalankan cell ekspor di NB02 sebelum deploy.
    """
    required = {
        "xgb_model.pkl"        : MODEL_PATH,
        "selected_features.pkl": FEATURES_PATH,
        "model_metrics.pkl"    : METRICS_PATH,
        "scaler.pkl"           : SCALER_PATH,
    }
    missing_files = [name for name, path in required.items()
                     if not os.path.exists(path)]

    if missing_files:
        st.error(
            "❌ **File model belum lengkap.** "
            "Jalankan cell ekspor di `NB02` terlebih dahulu untuk menghasilkan:\n\n"
            + "\n".join(f"- `model/{f}`" for f in missing_files)
            + "\n\nLetakkan semua file tersebut di folder `model/` "
            "sebelum menjalankan aplikasi."
        )
        st.stop()

    with open(MODEL_PATH,    "rb") as f: model             = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f: selected_features = pickle.load(f)
    with open(METRICS_PATH,  "rb") as f: metrics           = pickle.load(f)
    with open(SCALER_PATH,   "rb") as f: scaler            = pickle.load(f)

    metrics.setdefault("skenario",    "Full Data 80:20 Holdout")
    metrics.setdefault("fs_method",   "Mutual Information")
    metrics.setdefault("n_fitur_mi",  len(selected_features))
    metrics.setdefault("n_fitur_raw", N_FEATURES_TOTAL)

    return model, selected_features, metrics, scaler


# ─────────────────────────────────────────────────────────────────────────────
#  MUAT SHAP VALUES DARI PKL (diekspor dari NB02)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def compute_shap_values(_model, _selected_features):
    """
    Memuat SHAP values dari file pkl yang diekspor langsung dari NB02.
    Plot identik 100% dengan notebook karena menggunakan sv & Xsv yang sama.

    File yang dibutuhkan: model/shap_values.pkl
    """
    if not os.path.exists(SHAP_PATH):
        st.error(
            "❌ **File `model/shap_values.pkl` belum ditemukan.**\n\n"
            "Jalankan cell ekspor SHAP di `NB02` untuk menghasilkannya:\n"
            "```python\n"
            "with open('model/shap_values.pkl', 'wb') as f:\n"
            "    pickle.dump({'sv': sv, 'Xsv': Xsv, 'version': 'from_notebook'}, f)\n"
            "```"
        )
        st.stop()

    with open(SHAP_PATH, "rb") as f:
        shap_data = pickle.load(f)

    if shap_data.get("sv") is None or shap_data.get("Xsv") is None:
        st.error(
            "❌ File `shap_values.pkl` tidak valid atau rusak.\n\n"
            "Hapus file tersebut dan jalankan ulang cell ekspor di NB02."
        )
        st.stop()

    return shap_data


# ─────────────────────────────────────────────────────────────────────────────
#  MATPLOTLIB SHAP RENDERER
# ─────────────────────────────────────────────────────────────────────────────
_SHAP_RC = {
    "figure.facecolor": "#090B20", "axes.facecolor": "#090B20",
    "text.color": "#DDE3FF",       "axes.labelcolor": "#9CA8E8",
    "xtick.color": "#4A5280",      "ytick.color": "#4A5280",
    "axes.edgecolor": "#1C2050",   "grid.color": "#161A38",
    "grid.alpha": 0.35,            "axes.grid": True,
    "savefig.facecolor": "#090B20",
}

def render_shap_bar(sv, Xsv, title="SHAP Importance — XGBoost + Mutual Information (Top 20)"):
    plt.close("all")
    with matplotlib.rc_context(_SHAP_RC):
        shap.summary_plot(sv, Xsv, plot_type="bar", max_display=20, show=False)
        fig = plt.gcf()
        fig.patch.set_facecolor("#090B20")
        fig.set_size_inches(7, 7)
        for ax in fig.axes:
            ax.set_facecolor("#090B20")
            ax.tick_params(colors="#9CA8E8", labelsize=9)
            ax.xaxis.label.set_color("#9CA8E8")
            ax.yaxis.label.set_color("#9CA8E8")
            for spine in ax.spines.values():
                spine.set_edgecolor("#1C2050")
        fig.axes[0].set_title(title, fontsize=11, color="#DDE3FF", pad=10)
        plt.tight_layout()
    return fig


def render_shap_beeswarm(sv, Xsv, title="SHAP — XGBoost + Mutual Information (Top 20)"):
    plt.close("all")
    with matplotlib.rc_context(_SHAP_RC):
        shap.summary_plot(sv, Xsv, max_display=20, show=False)
        fig = plt.gcf()
        fig.patch.set_facecolor("#090B20")
        fig.set_size_inches(7, 7)
        for ax in fig.axes:
            ax.set_facecolor("#090B20")
            ax.tick_params(colors="#9CA8E8", labelsize=9)
            ax.xaxis.label.set_color("#9CA8E8")
            ax.yaxis.label.set_color("#9CA8E8")
            for spine in ax.spines.values():
                spine.set_edgecolor("#1C2050")
        for ax in fig.axes[1:]:
            ax.tick_params(colors="#9CA8E8", labelsize=8)
        fig.axes[0].set_title(title, fontsize=11, color="#DDE3FF", pad=10)
        plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  CHART HELPER
# ─────────────────────────────────────────────────────────────────────────────
PLOT_BG  = "#090B20"
GRID_COL = "#161A38"
AXIS_COL = "#4A5280"

def chart_layout(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Grotesk", size=15, color=C_VIOLET)),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=C_TEXT2, family="Inter"),
        margin=dict(l=20, r=20, t=50, b=40), height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C_TEXT2)),
        hoverlabel=dict(bgcolor="#12163A", font_color=C_TEXT, bordercolor=C_BLUE),
    )
    fig.update_xaxes(gridcolor=GRID_COL, zerolinecolor=GRID_COL, tickfont=dict(color=AXIS_COL))
    fig.update_yaxes(gridcolor=GRID_COL, zerolinecolor=GRID_COL, tickfont=dict(color=AXIS_COL))
    return fig


def make_contribution_chart(model, selected_features, compound_values, top_n=15):
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"fitur": selected_features, "importance": importances})
    vals    = np.array([compound_values.get(f, 0.0) for f in selected_features], dtype=float)
    max_v   = np.abs(vals).max()
    norm    = np.abs(vals) / max_v if max_v > 0 else np.ones(len(vals))
    feat_df["kontribusi"] = feat_df["importance"] * norm
    feat_df = feat_df.nlargest(top_n, "kontribusi").sort_values("kontribusi")
    n       = len(feat_df)
    colors  = [f"rgba({int(91+(197-91)*(i/max(n-1,1)))},{int(114+(106-114)*(i/max(n-1,1)))},{int(245+(212-245)*(i/max(n-1,1)))},0.88)" for i in range(n)]
    fig = go.Figure(go.Bar(
        x=feat_df["kontribusi"].values, y=feat_df["fitur"].values, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(91,114,245,0.3)", width=0.5)),
        text=[f"{v:.4f}" for v in feat_df["kontribusi"].values],
        textposition="outside", textfont=dict(size=9, color=C_TEXT2),
        hovertemplate="<b>%{y}</b><br>Kontribusi: %{x:.5f}<extra></extra>",
    ))
    return chart_layout(fig, "🔬 Kontribusi Fitur — Senyawa Ini", height=max(360, top_n * 26))


def make_global_importance_chart(model, selected_features, top_n=20):
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"fitur": selected_features, "importance": importances})
    feat_df = feat_df.nlargest(top_n, "importance").sort_values("importance")
    n       = len(feat_df)
    colors  = [f"rgba({int(91+(197-91)*(i/max(n-1,1)))},{int(114+(106-114)*(i/max(n-1,1)))},{int(245+(212-245)*(i/max(n-1,1)))},0.90)" for i in range(n)]
    fig = go.Figure(go.Bar(
        x=feat_df["importance"].values, y=feat_df["fitur"].values, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(91,114,245,0.3)", width=0.4)),
        text=[f"{v:.4f}" for v in feat_df["importance"].values],
        textposition="outside", textfont=dict(size=9, color=C_TEXT2),
        hovertemplate="<b>%{y}</b><br>Skor: %{x:.5f}<extra></extra>",
    ))
    return chart_layout(fig, f"📊 {top_n} Fitur Terpenting (Gain XGBoost)", height=max(420, top_n * 26))


def make_comparison_chart(metric_col, metric_label):
    colors = ["#4A90D9", C_BLUE, C_VIOLET, C_PINK, "#E8A838"]
    fig    = go.Figure()
    for i, fs in enumerate(COMPARISON_DATA["Seleksi Fitur"].unique()):
        sub = COMPARISON_DATA[COMPARISON_DATA["Seleksi Fitur"] == fs]
        fig.add_trace(go.Bar(
            name=fs, x=sub["Model"].values, y=sub[metric_col].values,
            marker=dict(color=colors[i % len(colors)], opacity=0.85,
                        line=dict(color="rgba(255,255,255,0.1)", width=0.5)),
            text=[f"{v:.4f}" for v in sub[metric_col].values],
            textposition="outside", textfont=dict(size=9, color=C_TEXT2),
            hovertemplate=f"<b>%{{x}}</b><br>{metric_label}: %{{y:.4f}}<br>FS: {fs}<extra></extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0.05)
    return chart_layout(fig, f"📈 Perbandingan {metric_label} — Model & Teknik Seleksi Fitur", height=420)


# ─────────────────────────────────────────────────────────────────────────────
#  RENDER HASIL PREDIKSI
# ─────────────────────────────────────────────────────────────────────────────
def render_predictions(uploaded_file, model, selected_features, scaler):
    try:
        df_input = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Tidak dapat membaca file: {e}")
        return

    id_cols = [c for c in ["Label", "SMILES", "Name", "Compound", "Nama"] if c in df_input.columns]
    id_data = df_input[id_cols].copy() if id_cols else pd.DataFrame()
    X_raw   = df_input.drop(columns=id_cols, errors="ignore")
    X_raw   = X_raw.apply(lambda c: c.astype(str).str.replace(",", ".", regex=False)
                          if c.dtype == object else c)
    X_raw   = X_raw.apply(pd.to_numeric, errors="coerce").fillna(0)

    missing = [f for f in selected_features if f not in X_raw.columns]
    if missing:
        st.error(
            f"⚠️ File tidak memiliki **{len(missing)}** kolom deskriptor yang dibutuhkan. "
            f"Contoh kolom hilang: `{'`, `'.join(missing[:5])}`\n\n"
            "Pastikan file CSV memiliki deskriptor RDKit lengkap."
        )
        return

    # ── 1. StandardScaler (fit dari Xtr80 raw, semua kolom) ──────────────────
    scaler_cols = (scaler.feature_names_in_.tolist()
                   if hasattr(scaler, "feature_names_in_") else X_raw.columns.tolist())
    X_for_scale = X_raw.reindex(columns=scaler_cols, fill_value=0)
    X_scaled    = pd.DataFrame(scaler.transform(X_for_scale), columns=scaler_cols)

    # ── 2. Pilih fitur hasil Mutual Information ───────────────────────────────
    X_sel  = X_scaled[selected_features]
    y_pred = model.predict(X_sel)
    y_prob = model.predict_proba(X_sel)
    n      = len(y_pred)

    n_buang = N_FEATURES_TOTAL - len(selected_features)
    st.markdown(
        f"<p style='color:{C_TEXT2};font-size:0.87rem;margin-bottom:1rem;'>"
        f"✅ Berhasil menganalisis <strong style='color:{C_VIOLET};'>{n} senyawa</strong> "
        f"menggunakan <strong>{len(selected_features)} fitur</strong> "
        f"(setelah Mutual Information — {n_buang} fitur dibuang).</p>",
        unsafe_allow_html=True,
    )

    for idx in range(n):
        pred      = int(y_pred[idx])
        prob_diad = float(y_prob[idx][1])
        prob_safe = float(y_prob[idx][0])
        label_str = "DIAD" if pred == 1 else "Non-DIAD"
        cname     = (str(id_data.iloc[idx]["Name"]) if "Name" in id_data.columns else f"Senyawa {idx+1}")

        with st.expander(
            f"{'🔴' if pred==1 else '🔵'} {cname}  —  **{label_str}**  "
            f"(Keyakinan: {max(prob_diad, prob_safe)*100:.1f}%)",
            expanded=(idx == 0),
        ):
            colA, colB = st.columns([1, 1.1])
            with colA:
                if pred == 1:
                    st.markdown(f"""
                    <div class="res-diad">
                        <div class="res-icon">⚠️</div>
                        <div class="res-label" style="color:#E05577;">{label_str}</div>
                        <p style="color:#FFAABB;font-size:0.88rem;margin:0.3rem 0;">
                            Risiko autoimunitas akibat obat <strong>terdeteksi</strong>
                        </p>
                        <div class="res-desc">
                            Senyawa ini diprediksi memiliki risiko signifikan<br>
                            dalam memicu respons autoimun.
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="res-safe">
                        <div class="res-icon">✅</div>
                        <div class="res-label" style="color:#A78BFA;">{label_str}</div>
                        <p style="color:#C4B5FD;font-size:0.88rem;margin:0.3rem 0;">
                            Tidak ada risiko autoimunitas signifikan <strong>terdeteksi</strong>
                        </p>
                        <div class="res-desc">
                            Senyawa ini diprediksi aman<br>
                            dari sudut pandang drug-induced autoimmunity.
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="pbar" style="margin-top:1.3rem;">
                    <div class="pbar-lbl">
                        <span>🔴 Probabilitas DIAD</span>
                        <span style="font-weight:600;color:#E05577;">{prob_diad*100:.1f}%</span>
                    </div>
                    <div class="pbar-bg"><div class="pbar-diad" style="width:{prob_diad*100:.1f}%;"></div></div>
                </div>
                <div class="pbar" style="margin-top:0.7rem;">
                    <div class="pbar-lbl">
                        <span>🔵 Probabilitas Non-DIAD</span>
                        <span style="font-weight:600;color:{C_VIOLET};">{prob_safe*100:.1f}%</span>
                    </div>
                    <div class="pbar-bg"><div class="pbar-safe" style="width:{prob_safe*100:.1f}%;"></div></div>
                </div>
                """, unsafe_allow_html=True)

            with colB:
                fig = make_contribution_chart(model, selected_features,
                                              X_sel.iloc[idx].to_dict(), top_n=15)
                st.plotly_chart(fig, use_container_width=True, key=f"contrib_{idx}")

    if n > 1:
        st.markdown('<div class="sec-title">📋 Tabel Ringkasan Prediksi</div>', unsafe_allow_html=True)
        summary = []
        for idx in range(n):
            pred      = int(y_pred[idx])
            prob_diad = float(y_prob[idx][1])
            prob_safe = float(y_prob[idx][0])
            summary.append({
                "Senyawa":            f"Senyawa {idx+1}",
                "Prediksi":           "DIAD" if pred == 1 else "Non-DIAD",
                "Prob. DIAD (%)":     round(prob_diad * 100, 2),
                "Prob. Non-DIAD (%)": round(prob_safe * 100, 2),
                "Keyakinan (%)":      round(max(prob_diad, prob_safe) * 100, 2),
            })
        st.dataframe(
            pd.DataFrame(summary).style.map(
                lambda v: ("color:#E05577;font-weight:bold" if v == "DIAD"
                           else ("color:#A78BFA;font-weight:bold" if v == "Non-DIAD" else "")),
                subset=["Prediksi"],
            ),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    with st.spinner("🔮 Memuat model — harap tunggu…"):
        model, selected_features, metrics, scaler = train_or_load_model()

    n_fitur = metrics.get("n_fitur_mi", len(selected_features))
    n_buang = N_FEATURES_TOTAL - n_fitur

    # ── Hero Banner ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-badge">⚡ MACHINE LEARNING &nbsp;·&nbsp; KEMOINFORMATIKA &nbsp;·&nbsp; KEAMANAN OBAT</div>
        <div class="hero-title">Prediktor Penyakit Autoimun<br>Akibat Obat (DIAD)</div>
        <div class="hero-sub">
            Prediksi risiko autoimunitas menggunakan <strong>XGBoost</strong> dan seleksi fitur
            <strong>Mutual Information</strong> pada {N_FEATURES_TOTAL} deskriptor molekuler RDKit.
            Model dilatih dengan strategi validasi <strong>Full Data 80:20 Holdout</strong>
            dari dataset DIA (597 senyawa).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Strip Metrik ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f"""<div class="mcard">
        <div class="mlabel">MCC</div>
        <div class="mvalue">{metrics['mcc']:.4f}</div>
        <div class="msub">Matthews Corr. Coef.</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="mcard">
        <div class="mlabel">ROC-AUC</div>
        <div class="mvalue">{metrics['roc_auc']:.4f}</div>
        <div class="msub">Holdout 80:20</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="mcard">
        <div class="mlabel">Akurasi</div>
        <div class="mvalue">{metrics['accuracy']*100:.2f}%</div>
        <div class="msub">Holdout 80:20</div></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="mcard">
        <div class="mlabel">Model</div>
        <div class="mvalue" style="font-size:1.1rem;">XGBoost</div>
        <div class="msub">Gradient Boosting</div></div>""", unsafe_allow_html=True)
    c5.markdown(f"""<div class="mcard">
        <div class="mlabel">Seleksi Fitur</div>
        <div class="mvalue" style="font-size:0.85rem;">Mutual Info</div>
        <div class="msub">{n_fitur} dari {N_FEATURES_TOTAL} fitur</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📤  Upload & Prediksi",
        "📊  Informasi Model",
        "🔍  Feature Importance",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 1 — UPLOAD & PREDIKSI
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(160deg, #0D0F28 0%, #130B30 100%);
            border:2px dashed {C_BORDER2}; border-radius:18px;
            padding:2rem 2rem 0.5rem 2rem; text-align:center; margin-bottom:-4px;
        ">
            <div style="font-size:3rem; margin-bottom:0.5rem;">🧪</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.2rem; font-weight:700; color:{C_TEXT}; margin-bottom:0.3rem;">
                Upload Deskriptor Senyawa Anda
            </div>
            <div style="color:{C_TEXT2}; font-size:0.86rem; margin-bottom:1rem;">
                Seret &amp; letakkan atau pilih file CSV dengan {N_FEATURES_TOTAL} deskriptor RDKit
                untuk memulai analisis prediksi
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload file CSV", type=["csv"],
            label_visibility="collapsed", key="main_uploader",
        )

        if uploaded is None:
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""<div class="icard">
                    <div class="icard-title">⚗️ Persyaratan Data</div>
                    <p>• Format CSV dengan <strong style='color:{C_VIOLET};'>{N_FEATURES_TOTAL} deskriptor RDKit 2D</strong></p>
                    <p>• Kolom sesuai nama deskriptor RDKit standar</p>
                    <p>• Satu baris per senyawa; banyak senyawa didukung</p>
                    <p>• Kolom <code>SMILES</code> atau <code>Name</code> opsional untuk pelabelan</p>
                    <p>• Seleksi fitur MI ({n_buang} kolom dibuang) diterapkan otomatis</p>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class="icard">
                    <div class="icard-title">🎯 Output Prediksi</div>
                    <p>• <strong style='color:#E05577;'>DIAD</strong> – Risiko autoimunitas akibat obat terdeteksi</p>
                    <p>• <strong style='color:{C_VIOLET};'>Non-DIAD</strong> – Tidak ada risiko autoimunitas signifikan</p>
                    <p>• Skor probabilitas untuk setiap kelas prediksi</p>
                    <p>• Grafik kontribusi fitur per senyawa</p>
                    <p>• Fitur digunakan: <strong>{n_fitur}</strong> (setelah Mutual Information)</p>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="text-align:center; padding:1rem; color:{C_TEXT3}; font-size:0.82rem;">
                💡 Hitung deskriptor RDKit via
                <a href='http://www.scbdd.com/rdk_desc/index/' target='_blank'
                   style='color:{C_BLUE};'>SCBDD RDKit Descriptor Calculator</a>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            render_predictions(uploaded, model, selected_features, scaler)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 2 — INFORMASI MODEL
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="sec-title">🤖 Ringkasan Model Terbaik</div>', unsafe_allow_html=True)

        mc1, mc2, mc3 = st.columns(3)
        mc1.markdown(f"""<div class="icard">
            <div class="icard-title">🏆 Algoritma Terpilih</div>
            <p><strong style='color:{C_VIOLET};font-size:1.1rem;'>XGBoost Classifier</strong></p>
            <p>Metode: <strong>Gradient Boosting</strong></p>
            <p>scale_pos_weight: <strong>otomatis dari ytr80</strong></p>
            <p>eval_metric: <strong>logloss</strong></p>
            <p>HP Tuning: <strong>GridSearchCV (108 kombinasi)</strong></p>
            <p>Inner CV: <strong>StratifiedKFold(3), scoring=MCC</strong></p>
        </div>""", unsafe_allow_html=True)

        mc2.markdown(f"""<div class="icard">
            <div class="icard-title">🔬 Seleksi Fitur Terbaik</div>
            <p><strong style='color:{C_VIOLET};font-size:1.1rem;'>Mutual Information</strong></p>
            <p>Threshold MI Score: <strong>&gt; 0.01</strong></p>
            <p>Fitur dibuang: <strong>{n_buang}</strong> dari {N_FEATURES_TOTAL}</p>
            <p>Fitur terpilih: <strong>{n_fitur}</strong></p>
            <p>Skenario: <strong>Full Data 80:20 Holdout</strong></p>
            <p>Metode: <strong>mutual_info_classif (sklearn)</strong></p>
        </div>""", unsafe_allow_html=True)

        mc3.markdown(f"""<div class="icard">
            <div class="icard-title">📈 Performa Model (Holdout 80:20)</div>
            <p>MCC: <strong style='color:{C_VIOLET};'>{metrics['mcc']:.4f}</strong></p>
            <p>ROC-AUC: <strong style='color:{C_VIOLET};'>{metrics['roc_auc']:.4f}</strong></p>
            <p>Akurasi: <strong style='color:{C_VIOLET};'>{metrics['accuracy']*100:.2f}%</strong></p>
            <p>F1-Score (Makro): <strong style='color:{C_VIOLET};'>{metrics['f1']:.4f}</strong></p>
            <p>Data training: <strong>477 sampel (80%)</strong></p>
            <p>Data test: <strong>120 sampel (20%)</strong></p>
        </div>""", unsafe_allow_html=True)

        # ── Tabel Jumlah Fitur per Metode FS ──────────────────────────────────
        st.markdown('<div class="sec-title">📋 Jumlah Fitur Hasil Seleksi per Metode</div>',
                    unsafe_allow_html=True)
        rows_html = ""
        for _, row in FS_TABLE.iterrows():
            is_best = row["Metode"] == "Mutual Information"
            badge   = ' <span class="badge-best">★ TERBAIK</span>' if is_best else ""
            rows_html += f"""
            <tr>
                <td><span class="badge-fs">{row['Metode']}</span>{badge}</td>
                <td style="text-align:center;color:{'#A78BFA' if is_best else C_TEXT2};font-weight:{'700' if is_best else '400'};">{row['Fitur Terpilih']}</td>
                <td style="text-align:center;">{row['Total']}</td>
                <td style="text-align:center;">{row['Rasio']}</td>
                <td>{row['Keterangan']}</td>
            </tr>"""

        st.markdown(f"""
        <table class="styled-table">
            <thead><tr>
                <th>Metode Seleksi Fitur</th>
                <th style="text-align:center;">Fitur Terpilih</th>
                <th style="text-align:center;">Total Fitur</th>
                <th style="text-align:center;">Rasio</th>
                <th>Keterangan</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        # ── Tabel Perbandingan Lengkap ─────────────────────────────────────────
        st.markdown('<div class="sec-title">🗂️ Tabel Perbandingan Performa — Full Data 80:20 Holdout</div>',
                    unsafe_allow_html=True)

        tbl_rows = ""
        for _, row in COMPARISON_DATA.iterrows():
            is_best = row["Model"] == "XGBoost" and row["Seleksi Fitur"] == "Mutual Information"
            badge   = ' <span class="badge-best">★ TERBAIK</span>' if is_best else ""
            row_bg  = "background:rgba(91,114,245,0.06);" if is_best else ""
            tbl_rows += f"""
            <tr style="{row_bg}">
                <td><strong style="color:{C_TEXT};">{row['Model']}</strong>{badge}</td>
                <td><span class="badge-fs">{row['Seleksi Fitur']}</span></td>
                <td style="text-align:center;">{row['Jumlah Fitur']}</td>
                <td style="text-align:center;color:{'#A78BFA' if is_best else C_TEXT2};">{row['Akurasi']:.4f}</td>
                <td style="text-align:center;color:{'#A78BFA' if is_best else C_TEXT2};">{row['MCC']:.4f}</td>
                <td style="text-align:center;">{row['F1-Score']:.4f}</td>
                <td style="text-align:center;">{row['ROC-AUC']:.4f}</td>
            </tr>"""

        st.markdown(f"""
        <table class="styled-table">
            <thead><tr>
                <th>Model</th><th>Seleksi Fitur</th>
                <th style="text-align:center;">Jml. Fitur</th>
                <th style="text-align:center;">Akurasi</th>
                <th style="text-align:center;">MCC</th>
                <th style="text-align:center;">F1-Score</th>
                <th style="text-align:center;">ROC-AUC</th>
            </tr></thead>
            <tbody>{tbl_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

        # ── Chart Perbandingan ─────────────────────────────────────────────────
        st.markdown('<div class="sec-title">📊 Visualisasi Perbandingan Performa</div>',
                    unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)
        with ch1: st.plotly_chart(make_comparison_chart("Akurasi", "Akurasi"), use_container_width=True, key="cmp_acc")
        with ch2: st.plotly_chart(make_comparison_chart("MCC", "MCC"),         use_container_width=True, key="cmp_mcc")
        ch3, ch4 = st.columns(2)
        with ch3: st.plotly_chart(make_comparison_chart("F1-Score", "F1-Score"), use_container_width=True, key="cmp_f1")
        with ch4: st.plotly_chart(make_comparison_chart("ROC-AUC", "ROC-AUC"),   use_container_width=True, key="cmp_auc")

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 3 — SHAP FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown(f"""
        <div class="icard" style="margin-bottom:1.5rem;">
            <div class="icard-title">ℹ️ SHAP Feature Importance — XGBoost + Mutual Information</div>
            <p>Analisis menggunakan <strong style='color:{C_VIOLET};'>SHAP (SHapley Additive exPlanations)</strong>
            konsisten dengan <code>02_Main.ipynb</code>:
            model dilatih pada <strong>Xtr80</strong> (split 80:20, 477 sampel)
            dengan <strong>Mutual Information (threshold MI &gt; 0.01)</strong> sebagai seleksi fitur,
            menghasilkan <strong>{n_fitur} fitur</strong> yang digunakan.</p>
            <p>Explainer: <code>shap.TreeExplainer</code> + <code>shap.summary_plot(max_display=20)</code>,
            identik dengan fungsi <code>compute_shap()</code> dan <code>plot_shap_*()</code> di <code>utils.py</code>.</p>
            <p>• <strong style='color:{C_BLUE};'>SHAP Bar</strong> — mean |SHAP value| per fitur (kepentingan global).</p>
            <p>• <strong style='color:{C_BLUE};'>SHAP Beeswarm</strong> — distribusi SHAP per sampel;
            <span style='color:#E84040;'>merah</span> = nilai fitur tinggi,
            <span style='color:#4472CA;'>biru</span> = nilai fitur rendah.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("⚗️ Menghitung SHAP values — harap tunggu…"):
            shap_data = compute_shap_values(model, selected_features)
        sv  = shap_data["sv"]
        Xsv = shap_data["Xsv"]

        st.markdown(
            f'<div class="sec-title">📊 SHAP Bar Chart & Beeswarm Plot — Top 20 Fitur ({n_fitur} fitur aktif)</div>',
            unsafe_allow_html=True,
        )
        col_bar, col_bee = st.columns(2)
        with col_bar:
            fig_bar = render_shap_bar(sv, Xsv, title=f"SHAP Importance — XGBoost + Mutual Information (Top 20)")
            st.pyplot(fig_bar, use_container_width=True)
            plt.close(fig_bar)
        with col_bee:
            fig_bee = render_shap_beeswarm(sv, Xsv, title=f"SHAP — XGBoost + Mutual Information (Top 20)")
            st.pyplot(fig_bee, use_container_width=True)
            plt.close(fig_bee)

        st.markdown(
            f'<div class="sec-title">📋 Tabel Peringkat SHAP — {len(Xsv.columns)} Fitur (Xtr80 + MI)</div>',
            unsafe_allow_html=True,
        )
        mean_abs_shap = np.abs(sv).mean(axis=0)
        shap_df = pd.DataFrame({
            "Rank"        : range(1, len(Xsv.columns) + 1),
            "Nama Fitur"  : Xsv.columns.tolist(),
            "Mean |SHAP|" : np.round(mean_abs_shap, 6),
            "Mean SHAP"   : np.round(sv.mean(axis=0), 6),
            "Max |SHAP|"  : np.round(np.abs(sv).max(axis=0), 6),
        }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
        shap_df["Rank"] = range(1, len(shap_df) + 1)
        st.dataframe(
            shap_df.style.background_gradient(subset=["Mean |SHAP|"], cmap="Blues"),
            use_container_width=True, height=440,
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<p style='text-align:center;font-size:0.74rem;color:{C_TEXT3};'>"
        f"DIAD Predictor &nbsp;·&nbsp; XGBoost + Mutual Information &nbsp;·&nbsp; "
        f"{N_FEATURES_TOTAL} Deskriptor Molekuler RDKit &nbsp;·&nbsp; "
        f"Full Data 80:20 Holdout &nbsp;·&nbsp; Dibangun dengan Streamlit 🔮"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
