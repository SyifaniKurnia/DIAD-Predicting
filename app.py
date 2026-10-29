"""
═══════════════════════════════════════════════════════════════════════════════
 APLIKASI PREDIKSI RISIKO DIAD (Drug-Induced Autoimmune Disease)
═══════════════════════════════════════════════════════════════════════════════

 MODEL YANG DIPAKAI
   XGBoost + Mutual Information + holdout 80:20
   MCC = 0,6181 | ROC-AUC = 0,8967 | Akurasi = 85,83%
   (terpilih dari 210 kombinasi pada 14 skenario validasi)

 ALUR PREDIKSI 
 ─────────────────────────────────────────────────────────────────────────────

   Senyawa (misal: Amoxicillin)
      │
      │ 
      ▼
   SMILES: CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O
      │
      │  ← RDKit menghitung deskriptor, DI LUAR aplikasi
      ▼
   CSV berisi 196 angka  (MolWt=365.4, SlogP_VSA10=12.3, fr_NH2=1, ...)
      │
      │  
      ▼
   TAHAP 1 — Buang kolom identitas (Label/SMILES/Name), bersihkan angka
      ▼
   TAHAP 2 — StandardScaler: z = (x - mean) / std
             mean & std diambil dari 477 senyawa DATA LATIH (scaler.pkl)
      ▼
   TAHAP 3 — Ambil 95 fitur hasil Mutual Information (selected_features.pkl)
      ▼
   TAHAP 4 — XGBoost menghitung P(DIAD) = 0,912
      ▼
   Ambang 0,5  →  0,912 > 0,5  →  "DIAD"

 ─────────────────────────────────────────────────────────────────────────────
 TIGA HAL YANG SERING DITANYAKAN

 1. Di mana ambang 0,5 berada?
    Tersembunyi di dalam model.predict() — perilaku bawaan XGBoost.
    Tidak ada baris kode yang menuliskannya secara eksplisit.

 2. Apa yang sebenarnya "dilihat" model?
    Hanya 95 angka. Model tidak tahu apa itu Amoxicillin, tidak melihat
    strukturnya, dan tidak memahami kimia. Ia hanya mengenali pola angka.

 3. Mengapa scaler memakai .transform() dan bukan .fit_transform()?
    Agar mean dan std tetap berasal dari data latih. Menghitung ulang dari
    data baru akan menyebabkan kebocoran data (data leakage).
═══════════════════════════════════════════════════════════════════════════════
"""

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

import concurrent.futures.thread   
import concurrent.futures.process 

import streamlit as st
import pandas as pd
import numpy as np
import shap

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

# ─────────────────────────────────────────────────────────────────────────────
#  JUMLAH FITUR
# ─────────────────────────────────────────────────────────────────────────────
#  196 = total deskriptor RDKit 2D per senyawa, setelah kolom SMILES dan Label
#        dibuang. Inilah "kartu identitas numerik" sebuah molekul: MolWt,
#        SlogP_VSA10, fr_NH2, BalabanJ, dan seterusnya.
#
#   95 = fitur yang lolos seleksi Mutual Information (ambang MI > 0.01).
#        101 deskriptor sisanya dibuang karena skor MI-nya terhadap label DIAD
#        berada di bawah ambang, artinya nyaris tidak membawa informasi
#        pembeda antara kelas DIAD dan Non-DIAD.
#  Aplikasi hanya membaca daftar nama fitur tersebut dari selected_features.pkl.
# ─────────────────────────────────────────────────────────────────────────────
N_FEATURES_TOTAL = 196   # total deskriptor RDKit (setelah drop SMILES/Label)
N_FEATURES_MI    = 95    # fitur tersisa setelah Mutual Information (MI > 0.01)

TARGET    = "Label"
DROP_COLS = ["Label", "SMILES"]
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  DATA PERBANDINGAN MODEL
#  Sumber: Tabel 4.4 skripsi (= 01_Validation_Strategy.ipynb), skenario 80:20 holdout
#  15 kombinasi: 3 Model × 5 Teknik Seleksi Fitur
#  CATATAN: angka di bawah disalin verbatim dari Tabel 4.4. Kolom "F1-Score"
#  adalah F1 makro. Jumlah fitur mengikuti hasil seleksi pada data latih 80:20:
#  Baseline 196, VT 169, ZDE 112, Spearman 131, MI 95.
# ─────────────────────────────────────────────────────────────────────────────
COMPARISON_DATA = pd.DataFrame([
    # ── SVM ───────────────────────────────────────────────────────────────────
    {"Model": "SVM", "Seleksi Fitur": "Baseline",            "Jumlah Fitur": 196, "Akurasi": 0.8250, "MCC": 0.5495, "F1-Score": 0.7740, "ROC-AUC": 0.8674},
    {"Model": "SVM", "Seleksi Fitur": "Variance Threshold",  "Jumlah Fitur": 169, "Akurasi": 0.8083, "MCC": 0.5064, "F1-Score": 0.7525, "ROC-AUC": 0.8385},
    {"Model": "SVM", "Seleksi Fitur": "Zero Dominant",       "Jumlah Fitur": 112, "Akurasi": 0.8167, "MCC": 0.5459, "F1-Score": 0.7701, "ROC-AUC": 0.8652},
    {"Model": "SVM", "Seleksi Fitur": "Spearman",            "Jumlah Fitur": 131, "Akurasi": 0.8167, "MCC": 0.5339, "F1-Score": 0.7656, "ROC-AUC": 0.8600},
    {"Model": "SVM", "Seleksi Fitur": "Mutual Information",  "Jumlah Fitur":  95, "Akurasi": 0.7833, "MCC": 0.4758, "F1-Score": 0.7332, "ROC-AUC": 0.8487},
    # ── Random Forest ─────────────────────────────────────────────────────────
    {"Model": "Random Forest", "Seleksi Fitur": "Baseline",           "Jumlah Fitur": 196, "Akurasi": 0.8000, "MCC": 0.4228, "F1-Score": 0.7054, "ROC-AUC": 0.8830},
    {"Model": "Random Forest", "Seleksi Fitur": "Variance Threshold", "Jumlah Fitur": 169, "Akurasi": 0.8083, "MCC": 0.4620, "F1-Score": 0.7287, "ROC-AUC": 0.8793},
    {"Model": "Random Forest", "Seleksi Fitur": "Zero Dominant",      "Jumlah Fitur": 112, "Akurasi": 0.7833, "MCC": 0.3730, "F1-Score": 0.6809, "ROC-AUC": 0.8726},
    {"Model": "Random Forest", "Seleksi Fitur": "Spearman",           "Jumlah Fitur": 131, "Akurasi": 0.8250, "MCC": 0.4799, "F1-Score": 0.7025, "ROC-AUC": 0.9196},
    {"Model": "Random Forest", "Seleksi Fitur": "Mutual Information", "Jumlah Fitur":  95, "Akurasi": 0.8250, "MCC": 0.4835, "F1-Score": 0.6911, "ROC-AUC": 0.9052},
    # ── XGBoost ───────────────────────────────────────────────────────────────
    {"Model": "XGBoost", "Seleksi Fitur": "Baseline",            "Jumlah Fitur": 196, "Akurasi": 0.8250, "MCC": 0.5282, "F1-Score": 0.7640, "ROC-AUC": 0.8989},
    {"Model": "XGBoost", "Seleksi Fitur": "Variance Threshold",  "Jumlah Fitur": 169, "Akurasi": 0.8417, "MCC": 0.5646, "F1-Score": 0.7814, "ROC-AUC": 0.9070},
    {"Model": "XGBoost", "Seleksi Fitur": "Zero Dominant",       "Jumlah Fitur": 112, "Akurasi": 0.8333, "MCC": 0.5766, "F1-Score": 0.7869, "ROC-AUC": 0.8978},
    {"Model": "XGBoost", "Seleksi Fitur": "Spearman",            "Jumlah Fitur": 131, "Akurasi": 0.8167, "MCC": 0.5222, "F1-Score": 0.7608, "ROC-AUC": 0.8878},
    {"Model": "XGBoost", "Seleksi Fitur": "Mutual Information",  "Jumlah Fitur":  95, "Akurasi": 0.8583, "MCC": 0.6181, "F1-Score": 0.8090, "ROC-AUC": 0.8967},  # ← TERBAIK
])

# Tabel ringkasan seleksi fitur
FS_TABLE = pd.DataFrame([
    {"Metode": "Baseline",           "Fitur Terpilih": 196, "Total": 196, "Rasio": "100.0%", "Keterangan": "Semua deskriptor RDKit digunakan tanpa seleksi"},
    {"Metode": "Variance Threshold", "Fitur Terpilih": 169, "Total": 196, "Rasio": "86.2%",  "Keterangan": "Eliminasi fitur dengan variance ≤ 0.01 (pada data training)"},
    {"Metode": "Zero Dominant",      "Fitur Terpilih": 112, "Total": 196, "Rasio": "57.1%",  "Keterangan": "Eliminasi fitur dengan proporsi nilai dominan > 80%"},
    {"Metode": "Spearman",           "Fitur Terpilih": 131, "Total": 196, "Rasio": "66.8%",  "Keterangan": "Eliminasi salah satu dari pasangan fitur berkorelasi Spearman > 0.85"},
    {"Metode": "Mutual Information", "Fitur Terpilih":  95, "Total": 196, "Rasio": "48.5%",  "Keterangan": "Pertahankan fitur dengan MI Score > 0.01 terhadap label ★ TERBAIK"},
])

# ─────────────────────────────────────────────────────────────────────────────
#  WARNA TEMA
# ─────────────────────────────────────────────────────────────────────────────
C_BG      = "#F5FBF6"
C_CARD    = "#FFFFFF"
C_CARD2   = "#EAF7EE"
C_BORDER  = "#D3EDD9"
C_BORDER2 = "#A9DFB7"
C_BLUE    = "#63B583"
C_VIOLET  = "#3F9E6B"
C_PINK    = "#9BE0B0"
C_TEXT    = "#1F3A28"
C_TEXT2   = "#4B6B54"
C_TEXT3   = "#7C9C86"
C_GRAD    = "linear-gradient(135deg, #63B583 0%, #3F9E6B 50%, #9BE0B0 100%)"

# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; background-color: {C_BG}; color: {C_TEXT}; }}
.stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stHeader"],
[data-testid="stBottom"],
[data-testid="stToolbar"] {{
    background-color: {C_BG} !important;
    background-image: none !important;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stDecoration"] {{ background: transparent !important; display:none !important; }}
.main .block-container {{ padding: 1.5rem 2.5rem 2rem 2.5rem; max-width: 1300px; }}
[data-testid="collapsedControl"] {{ display: none; }}
[data-testid="stSidebar"] {{ display: none; }}
.hero-wrap {{
    background: linear-gradient(135deg, #EAF7EE 0%, #E1F3E7 40%, #D8EEE0 70%, #EFFAF2 100%);
    border-radius: 22px; padding: 2.8rem 3.5rem; margin-bottom: 1.6rem;
    position: relative; overflow: hidden; border: 1px solid rgba(63,158,107,0.20);
    box-shadow: 0 8px 30px rgba(99,181,131,0.12), 0 0 60px rgba(155,224,176,0.15);
}}
.hero-wrap::before {{
    content:''; position:absolute; inset:0;
    background: radial-gradient(ellipse at 78% 40%, rgba(155,224,176,0.25) 0%, transparent 55%),
                radial-gradient(ellipse at 20% 80%, rgba(99,181,131,0.14) 0%, transparent 50%);
}}
.hero-wrap::after {{
    content:''; position:absolute; top:-80px; right:-80px; width:380px; height:380px;
    border-radius:50%; background: radial-gradient(circle, rgba(63,158,107,0.12) 0%, transparent 65%);
}}
.hero-badge {{
    display:inline-flex; align-items:center; gap:0.4rem;
    background: rgba(63,158,107,0.12); backdrop-filter:blur(10px);
    border:1px solid rgba(63,158,107,0.30); border-radius:50px;
    padding:0.28rem 1rem; font-size:0.72rem; font-weight:600;
    color:#2F7D52; letter-spacing:0.10em; margin-bottom:1rem; position:relative; z-index:1;
}}
.hero-title {{
    font-family:'Space Grotesk',sans-serif; font-size:2.6rem; font-weight:700;
    background: linear-gradient(90deg,#1F3A28 0%,#3F9E6B 60%,#63B583 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin:0 0 0.5rem 0; line-height:1.15; position:relative; z-index:1;
}}
.hero-sub {{ font-size:0.98rem; color:#4B6B54; max-width:600px; line-height:1.65; position:relative; z-index:1; }}
.mrow {{ display:flex; gap:1rem; margin-bottom:1.8rem; flex-wrap:wrap; }}
.mcard {{
    flex:1; min-width:140px;
    background: linear-gradient(145deg, {C_CARD} 0%, {C_CARD2} 100%);
    border:1px solid {C_BORDER}; border-radius:16px; padding:1.2rem 1.4rem;
    position:relative; overflow:hidden; box-shadow:0 4px 16px rgba(99,181,131,0.10);
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
}}
.mcard:hover {{ transform:translateY(-3px); box-shadow:0 8px 26px rgba(63,158,107,0.20); border-color:rgba(63,158,107,0.40); }}
.mcard::before {{ content:''; position:absolute; top:0; left:0; right:0; height:3px; background: {C_GRAD}; }}
.mlabel {{ font-size:0.66rem; font-weight:700; letter-spacing:0.12em; color:{C_TEXT3}; text-transform:uppercase; margin-bottom:0.45rem; }}
.mvalue {{ font-family:'Space Grotesk',sans-serif; font-size:1.55rem; font-weight:700; color:{C_TEXT}; line-height:1; white-space:nowrap; }}
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
    color:{C_VIOLET} !important;
    background: linear-gradient(135deg, rgba(99,181,131,0.18), rgba(63,158,107,0.12)) !important;
    border-bottom:2px solid {C_VIOLET} !important;
}}
[data-testid="stTabsContent"] {{
    background: {C_CARD}; border:1px solid {C_BORDER};
    border-radius:0 16px 16px 16px; padding:2rem 2rem;
}}
[data-testid="stFileUploadDropzone"] {{
    background: linear-gradient(160deg, #F5FBF6 0%, #EAF7EE 100%) !important;
    border: 2px dashed {C_BORDER2} !important; border-radius: 18px !important;
    min-height: 130px !important; transition: border-color 0.3s, box-shadow 0.3s !important;
}}
[data-testid="stFileUploadDropzone"]:hover {{
    border-color: {C_VIOLET} !important; box-shadow: 0 0 20px rgba(63,158,107,0.18) !important;
}}
[data-testid="stFileUploadDropzone"] * {{ color:{C_TEXT2} !important; }}
[data-testid="stFileUploadDropzone"] button {{
    background: linear-gradient(135deg, {C_BLUE}, {C_VIOLET}) !important;
    color:#FFFFFF !important; border:none !important; border-radius:10px !important;
    padding:0.5rem 1.5rem !important; font-weight:600 !important;
    box-shadow: 0 4px 15px rgba(63,158,107,0.30) !important;
}}
.icard {{
    background: linear-gradient(145deg, #FFFFFF 0%, #F1FAF3 100%);
    border: 1px solid {C_BORDER}; border-radius:16px; padding:1.4rem 1.6rem;
    margin-bottom:1rem; box-shadow:0 3px 14px rgba(99,181,131,0.08);
    transition: border-color 0.2s, box-shadow 0.2s;
}}
.icard:hover {{ border-color:rgba(63,158,107,0.35); box-shadow:0 6px 20px rgba(63,158,107,0.14); }}
.icard-title {{
    font-family:'Space Grotesk',sans-serif; font-size:0.95rem; font-weight:600;
    color:{C_VIOLET}; margin-bottom:0.7rem; display:flex; align-items:center; gap:0.5rem;
}}
.icard p {{ color:{C_TEXT2}; font-size:0.85rem; margin:0.25rem 0; line-height:1.6; }}
.res-diad {{
    background: linear-gradient(135deg, #FDEEF0 0%, #FBE1E6 100%);
    border:2px solid #E05577; border-radius:18px; padding:1.8rem 2rem;
    text-align:center; animation:glow-red 2.5s ease-in-out infinite;
}}
@keyframes glow-red {{ 0%,100% {{ box-shadow:0 0 18px rgba(224,85,119,0.15); }} 50% {{ box-shadow:0 0 34px rgba(224,85,119,0.30); }} }}
.res-safe {{
    background: linear-gradient(135deg, #EAFBF0 0%, #DFF6E6 100%);
    border:2px solid {C_VIOLET}; border-radius:18px; padding:1.8rem 2rem;
    text-align:center; animation:glow-violet 2.5s ease-in-out infinite;
}}
@keyframes glow-violet {{ 0%,100% {{ box-shadow:0 0 18px rgba(63,158,107,0.15); }} 50% {{ box-shadow:0 0 34px rgba(63,158,107,0.30); }} }}
.res-icon {{ font-size:3rem; margin-bottom:0.4rem; }}
.res-label {{ font-family:'Space Grotesk',sans-serif; font-size:1.9rem; font-weight:800; margin:0.3rem 0; }}
.res-desc {{ font-size:0.86rem; color:{C_TEXT2}; margin-top:0.5rem; line-height:1.6; }}
.pbar {{ margin:0.8rem 0; }}
.pbar-lbl {{ display:flex; justify-content:space-between; font-size:0.78rem; color:{C_TEXT2}; margin-bottom:0.3rem; }}
.pbar-bg {{ background:#E7F3EA; border-radius:50px; height:10px; overflow:hidden; }}
.pbar-diad {{ height:100%; border-radius:50px; background:linear-gradient(90deg,#E05577,#FF7DA0); transition:width 0.8s ease; }}
.pbar-safe {{ height:100%; border-radius:50px; background: {C_GRAD}; transition:width 0.8s ease; }}
.sec-title {{
    font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:700;
    color:{C_VIOLET}; border-left:4px solid {C_BLUE}; padding-left:0.8rem; margin:1.6rem 0 1rem 0;
}}
.styled-table {{ width:100%; border-collapse:collapse; font-size:0.85rem; margin:1rem 0; }}
.styled-table th {{
    background: linear-gradient(90deg, rgba(99,181,131,0.20), rgba(63,158,107,0.14));
    color:{C_TEXT}; font-weight:600; padding:0.75rem 1rem;
    text-align:left; border-bottom:2px solid {C_BORDER};
}}
.styled-table td {{ padding:0.65rem 1rem; color:{C_TEXT2}; border-bottom:1px solid {C_BORDER}; }}
.styled-table tr:hover td {{ background:rgba(99,181,131,0.06); }}
.badge-best {{
    background: linear-gradient(135deg,#63B583,#3F9E6B);
    color:#FFF; font-size:0.68rem; font-weight:700;
    padding:0.15rem 0.55rem; border-radius:50px; letter-spacing:0.05em;
}}
.badge-fs {{
    background:rgba(63,158,107,0.14); color:#2F7D52; font-size:0.70rem; font-weight:600;
    padding:0.15rem 0.5rem; border-radius:50px; border:1px solid rgba(63,158,107,0.30);
}}
[data-testid="stDataFrame"] {{ border-radius:12px; overflow:hidden; }}
[data-testid="stExpander"] {{
    background:{C_CARD}; border:1px solid {C_BORDER} !important; border-radius:14px !important; margin-bottom:0.7rem;
}}
[data-testid="stExpander"] summary {{ color: white !important; font-weight:700 !important; }}
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
    """
    Membersihkan format angka Eropa/Indonesia (koma sebagai desimal, titik
    sebagai pemisah ribuan) menjadi format yang dikenali Python, lalu
    mengonversi seluruh kolom numerik ke tipe angka.

    Fungsi ini SENGAJA dibuat identik dengan fix_european() di utils.py agar
    representasi numerik deskriptor sama persis antara tahap pelatihan dan
    tahap prediksi. Bila salah satu diubah, ubah pula yang lain.
    """
    df = df.copy()

    for col in df.columns:
        # Lewati kolom yang memang sudah numerik; tidak perlu disentuh.
        # is_numeric_dtype() berlaku di semua versi pandas (2.x maupun 3.x).
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        try:
            teks = df[col].astype(str)

            # LANGKAH 1: buang titik pemisah ribuan, TETAPI hanya bila kolom
            # ini memang memakai koma sebagai desimal. Tanda pengenalnya:
            # ada koma di dalam teksnya. Tanpa syarat ini, angka biasa seperti
            # "365.4" akan dirusak menjadi "3654".
            if teks.str.contains(",", na=False).any():
                teks = teks.str.replace(".", "", regex=False)   # 1.234,5 → 1234,5

            # LANGKAH 2: koma desimal menjadi titik desimal.
            teks = teks.str.replace(",", ".", regex=False)      # 1234,5  → 1234.5

            # LANGKAH 3: ubah teks menjadi angka.
            # errors='coerce': nilai yang tidak bisa dikonversi (misalnya
            # kolom SMILES atau Name) menjadi NaN, bukan menghentikan program.
            konversi = pd.to_numeric(teks, errors='coerce')

            # PENJAGA: hanya terapkan bila kolom ini memang kolom angka.
            # Kolom teks murni (SMILES, Name) menghasilkan NaN seluruhnya —
            # kolom seperti itu dibiarkan utuh, tidak dirusak menjadi NaN.
            if konversi.notna().any():
                df[col] = konversi

        except Exception:
            pass  # kolom ini bukan angka — biarkan apa adanya

    return df

# ─────────────────────────────────────────────────────────────────────────────
#  MUAT MODEL DARI PKL (diekspor dari NB02)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_or_load_model():
    """
    Memuat model dan semua artefak dari file pkl yang diekspor dari NB02.

    File yang dibutuhkan (semua di folder model/):
      - xgb_model.pkl         → XGBClassifier dengan HP hasil GridSearchCV
      - selected_features.pkl → list nama fitur yang lolos MI (95 fitur)
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
    "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF",
    "text.color": "#1F3A28",       "axes.labelcolor": "#4B6B54",
    "xtick.color": "#7C9C86",      "ytick.color": "#7C9C86",
    "axes.edgecolor": "#D3EDD9",   "grid.color": "#DCEEDC",
    "grid.alpha": 0.6,             "axes.grid": True,
    "savefig.facecolor": "#FFFFFF",
}

def render_shap_bar(sv, Xsv, title="SHAP Importance — XGBoost + Mutual Information (Top 20)"):
    plt.close("all")
    with matplotlib.rc_context(_SHAP_RC):
        shap.summary_plot(sv, Xsv, plot_type="bar", max_display=20, show=False)
        fig = plt.gcf()
        fig.patch.set_facecolor("#FFFFFF")
        fig.set_size_inches(7, 7)
        for ax in fig.axes:
            ax.set_facecolor("#FFFFFF")
            ax.tick_params(colors="#4B6B54", labelsize=9)
            ax.xaxis.label.set_color("#4B6B54")
            ax.yaxis.label.set_color("#4B6B54")
            for spine in ax.spines.values():
                spine.set_edgecolor("#D3EDD9")
        fig.axes[0].set_title(title, fontsize=11, color="#1F3A28", pad=10)
        plt.tight_layout()
    return fig


def render_shap_beeswarm(sv, Xsv, title="SHAP — XGBoost + Mutual Information (Top 20)"):
    plt.close("all")
    with matplotlib.rc_context(_SHAP_RC):
        shap.summary_plot(sv, Xsv, max_display=20, show=False)
        fig = plt.gcf()
        fig.patch.set_facecolor("#FFFFFF")
        fig.set_size_inches(7, 7)
        for ax in fig.axes:
            ax.set_facecolor("#FFFFFF")
            ax.tick_params(colors="#4B6B54", labelsize=9)
            ax.xaxis.label.set_color("#4B6B54")
            ax.yaxis.label.set_color("#4B6B54")
            for spine in ax.spines.values():
                spine.set_edgecolor("#D3EDD9")
        for ax in fig.axes[1:]:
            ax.tick_params(colors="#4B6B54", labelsize=8)
        fig.axes[0].set_title(title, fontsize=11, color="#1F3A28", pad=10)
        plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  CHART HELPER
# ─────────────────────────────────────────────────────────────────────────────
PLOT_BG  = "#FFFFFF"
GRID_COL = "#E3F2E6"
AXIS_COL = "#7C9C86"

def chart_layout(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Grotesk", size=15, color=C_VIOLET)),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=C_TEXT2, family="Inter"),
        margin=dict(l=20, r=20, t=50, b=40), height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C_TEXT2)),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=C_TEXT, bordercolor=C_BLUE),
    )
    fig.update_xaxes(gridcolor=GRID_COL, zerolinecolor=GRID_COL, tickfont=dict(color=AXIS_COL))
    fig.update_yaxes(gridcolor=GRID_COL, zerolinecolor=GRID_COL, tickfont=dict(color=AXIS_COL))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  SHAP LOKAL — kontribusi fitur untuk SATU senyawa
# ─────────────────────────────────────────────────────────────────────────────
#  Rumus SHAP:  f(x) = base_value + Σ φᵢ   (i = 1..95 deskriptor)
#    φᵢ > 0 → deskriptor mendorong prediksi ke arah DIAD      (bar merah)
#    φᵢ < 0 → deskriptor mendorong prediksi ke arah Non-DIAD  (bar hijau)
#  Jumlah seluruh φᵢ persis sama dengan selisih keluaran model terhadap
#  nilai dasarnya. Sifat aditif inilah yang membuat SHAP sah secara matematis.
#  TreeExplainer dipakai karena XGBoost berbasis pohon: perhitungannya eksak.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_tree_explainer(_model):
    """Bangun TreeExplainer sekali, simpan di cache agar tidak diulang tiap senyawa."""
    return shap.TreeExplainer(_model)


def hitung_shap_lokal(model, X_sel):
    """
    Menghitung nilai SHAP untuk SELURUH senyawa sekaligus.
    Mengembalikan (sv, base_value):
      sv         : ndarray (n_senyawa × n_fitur) — nilai SHAP bertanda
      base_value : float — nilai dasar model
    """
    explainer = get_tree_explainer(model)
    sv = explainer.shap_values(X_sel)

    # Seragamkan bentuk keluaran antar versi pustaka shap → ambil kelas positif (DIAD)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    base = explainer.expected_value
    base = float(np.array(base).flatten()[-1]) if isinstance(base, (list, np.ndarray)) else float(base)
    return sv, base


def make_contribution_chart(selected_features, shap_row, top_n=15):
    """
    Bar chart SHAP lokal untuk SATU senyawa, lengkap dengan ARAH dorongan.
    Nilai mutlak dipakai hanya untuk MEMILIH top_n, bukan untuk MENAMPILKAN —
    sehingga tanda (+/−) tetap terlihat.
    """
    feat_df = pd.DataFrame({
        "fitur": selected_features,
        "shap" : np.asarray(shap_row, dtype=float),
    })
    feat_df["abs"] = feat_df["shap"].abs()
    feat_df = feat_df.nlargest(top_n, "abs").sort_values("shap")

    # Merah = mendorong ke DIAD; hijau = mendorong ke Non-DIAD
    colors = ["rgba(224,85,119,0.90)" if v > 0 else "rgba(63,158,107,0.90)"
              for v in feat_df["shap"].values]

    fig = go.Figure(go.Bar(
        x=feat_df["shap"].values, y=feat_df["fitur"].values, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(120,120,120,0.30)", width=0.5)),
        text=[f"{v:+.4f}" for v in feat_df["shap"].values],
        textposition="outside", textfont=dict(size=9, color=C_TEXT2),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.5f}<br><i>%{customdata}</i><extra></extra>",
        customdata=["mendorong ke DIAD" if v > 0 else "mendorong ke Non-DIAD"
                    for v in feat_df["shap"].values],
    ))
    fig.add_vline(x=0, line_width=1.2, line_color="#9AA5A0")

    return chart_layout(
        fig,
        "🔬 Kontribusi SHAP — Senyawa Ini  (merah → DIAD, hijau → Non-DIAD)",
        height=max(360, top_n * 26),
    )


def make_global_importance_chart(model, selected_features, top_n=20):
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"fitur": selected_features, "importance": importances})
    feat_df = feat_df.nlargest(top_n, "importance").sort_values("importance")
    n       = len(feat_df)
    colors  = [f"rgba({int(155+(63-155)*(i/max(n-1,1)))},{int(224+(158-224)*(i/max(n-1,1)))},{int(176+(107-176)*(i/max(n-1,1)))},0.92)" for i in range(n)]
    fig = go.Figure(go.Bar(
        x=feat_df["importance"].values, y=feat_df["fitur"].values, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(63,158,107,0.30)", width=0.4)),
        text=[f"{v:.4f}" for v in feat_df["importance"].values],
        textposition="outside", textfont=dict(size=9, color=C_TEXT2),
        hovertemplate="<b>%{y}</b><br>Skor: %{x:.5f}<extra></extra>",
    ))
    return chart_layout(fig, f"📊 {top_n} Fitur Terpenting (Gain XGBoost)", height=max(420, top_n * 26))


def make_comparison_chart(metric_col, metric_label):
    colors = ["#B7E4C7", C_BLUE, C_VIOLET, C_PINK, "#C9E4A0"]
    fig    = go.Figure()
    for i, fs in enumerate(COMPARISON_DATA["Seleksi Fitur"].unique()):
        sub = COMPARISON_DATA[COMPARISON_DATA["Seleksi Fitur"] == fs]
        fig.add_trace(go.Bar(
            name=fs, x=sub["Model"].values, y=sub[metric_col].values,
            marker=dict(color=colors[i % len(colors)], opacity=0.90,
                        line=dict(color="rgba(63,158,107,0.20)", width=0.5)),
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

    # ═════════════════════════════════════════════════════════════════════════
    #  ALUR PREDIKSI — 4 TAHAP
    #  CSV (196 angka) → StandardScaler → pilih 95 fitur → XGBoost → DIAD/Non-DIAD
    # ═════════════════════════════════════════════════════════════════════════

    # ── TAHAP 1: Pisahkan kolom identitas dari kolom deskriptor ──────────────
    # Kolom Label/SMILES/Name bukan deskriptor. Kolom ini disimpan terpisah (id_data) hanya untuk ditampilkan pada hasil, dan TIDAK ikut masuk model.
    # Kalau kolom Label ikut masuk, model akan "mengintip" jawabannya — itu kebocoran data (data leakage) yang paling fatal.
    id_cols = [c for c in ["Label", "SMILES", "Name", "Compound", "Nama"] if c in df_input.columns]
    id_data = df_input[id_cols].copy() if id_cols else pd.DataFrame()
    X_raw   = df_input.drop(columns=id_cols, errors="ignore")

    # Sebagian file CSV memakai format angka Eropa (koma sebagai desimal:
    # "3,14" bukan "3.14"). Baris ini menyeragamkannya menjadi format titik.
    X_raw   = fix_european(X_raw).fillna(0)

    # ── Validasi: pastikan seluruh 95 fitur yang dibutuhkan model tersedia ───
    missing = [f for f in selected_features if f not in X_raw.columns]
    if missing:
        st.error(
            f"⚠️ File tidak memiliki **{len(missing)}** kolom deskriptor yang dibutuhkan. "
            f"Contoh kolom hilang: `{'`, `'.join(missing[:5])}`\n\n"
            "Pastikan file CSV memiliki deskriptor RDKit lengkap."
        )
        return

    # ═════════════════════════════════════════════════════════════════════════
    #  TAHAP 2: STANDARDISASI Z-SCORE (StandardScaler)
    # ═════════════════════════════════════════════════════════════════════════
    #  Deskriptor RDKit punya skala yang sangat timpang. 
    #  MolWt bernilai 300–500, sedangkan MaxPartialCharge bernilai 0,01–0,3. 
    #  Tanpa penyeragaman, deskriptor berangka besar akan mendominasi model semata karena besarnya, bukan karena kepentingannya.
    #
    #  RUMUSNYA:  z = (x - mean) / std
    #
    #  DARI MANA mean DAN std?
    #  Dari 477 senyawa DATA LATIH (80% dari 597) — bukan dari CSV yang diunggah. 
    #  Nilai mean dan std itu sudah "dibekukan" di dalam scaler.pkl saat pelatihan di NB02.
    #
    #  CATATAN : yang dipanggil adalah .transform(), BUKAN .fit_transform().
    #  - .fit_transform() = hitung ulang mean/std dari data baru  → KEBOCORAN DATA
    #  - .transform()     = pakai mean/std data latih yang sudah ada  → BENAR
    #  Inilah prinsip anti-kebocoran yang dijelaskan pada BAB III.
    #
    #  URUTAN PENTING: scaler bekerja pada SELURUH 196 kolom, baru setelah itu
    #  95 fitur dipilih. Tidak boleh dibalik — scaler dilatih pada 196 kolom,
    #  sehingga akan error bila diberi jumlah kolom yang berbeda.
    # ─────────────────────────────────────────────────────────────────────────
    scaler_cols = (scaler.feature_names_in_.tolist()
                   if hasattr(scaler, "feature_names_in_") else X_raw.columns.tolist())
    # reindex: susun ulang kolom agar urutannya persis sama dengan saat pelatihan.
    # Urutan kolom penting karena scaler menyimpan mean/std per posisi kolom.
    X_for_scale = X_raw.reindex(columns=scaler_cols, fill_value=0)
    X_scaled    = pd.DataFrame(scaler.transform(X_for_scale), columns=scaler_cols)

    # ═════════════════════════════════════════════════════════════════════════
    #  TAHAP 3: SELEKSI FITUR — ambil 95 dari 196
    # ═════════════════════════════════════════════════════════════════════════
    #  selected_features berisi daftar nama 95 deskriptor yang lolos ambang
    #  Mutual Information (MI > 0,01) pada data latih. Sisanya (101 deskriptor)
    #  dibuang karena tidak membawa informasi pembeda DIAD vs Non-DIAD.
    X_sel = X_scaled[selected_features]

    # ═════════════════════════════════════════════════════════════════════════
    #  TAHAP 4: PREDIKSI
    # ═════════════════════════════════════════════════════════════════════════
    #  model.predict()       → menghasilkan 0 (Non-DIAD) atau 1 (DIAD)
    #  model.predict_proba() → menghasilkan [P(Non-DIAD), P(DIAD)], jumlah = 1,0
    #
    #  DI MANA AMBANG (THRESHOLD) 0,5 BERADA?
    #  Ambang 0,5 TIDAK tertulis di kode ini. Ia adalah perilaku bawaan
    #  XGBoost yang tersembunyi di dalam .predict(). Secara internal:
    #
    #      prob_diad = predict_proba(X)[:, 1]
    #      prediksi  = 1 jika prob_diad > 0.5, selain itu 0
    #
    #  KETERBATASAN (dibahas pada BAB IV & disarankan pada BAB V):
    #  Ambang 0,5 belum tentu optimal untuk data timpang 3:1. Bila ingin
    #  menerapkan threshold tuning, ganti dua baris di bawah dengan:
    #
    #      prob_diad = model.predict_proba(X_sel)[:, 1]
    #      y_pred    = (prob_diad > 0.35).astype(int)   # ambang bebas
    #
    #  Penelitian ini sengaja mempertahankan ambang 0,5 agar perbandingan
    #  210 kombinasi model berlangsung pada kondisi yang setara.
    # ─────────────────────────────────────────────────────────────────────────
    y_pred = model.predict(X_sel)        # array 0/1 — hasil pemotongan pada ambang 0,5
    y_prob = model.predict_proba(X_sel)  # array [P(kelas 0), P(kelas 1)] per senyawa
    n      = len(y_pred)

    # ── TAHAP 5: SHAP lokal ──────────────────────────────────────────────────
    # Dihitung SEKALI untuk seluruh senyawa (jauh lebih cepat daripada per senyawa).
    # sv_lokal[i] = vektor 95 nilai SHAP bertanda untuk senyawa ke-i.
    # Bila SHAP gagal, prediksi tetap ditampilkan — hanya grafiknya yang dilewati.
    try:
        sv_lokal, shap_base = hitung_shap_lokal(model, X_sel)
    except Exception as e:
        sv_lokal, shap_base = None, None
        st.warning(f"Grafik SHAP tidak dapat dihitung: {e}")

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
        # y_prob[idx] adalah pasangan [P(Non-DIAD), P(DIAD)] untuk senyawa ke-idx.
        # Indeks 0 = kelas negatif (Non-DIAD); indeks 1 = kelas positif (DIAD).
        # Keduanya selalu berjumlah 1,0. Inilah angka yang tampil di Tabel 4.12.
        # Contoh: prob_diad = 0,912 berarti model 91,2% yakin senyawa ini DIAD.
        prob_diad = float(y_prob[idx][1])
        prob_safe = float(y_prob[idx][0])
        # label_str hanya menerjemahkan angka 0/1 dari .predict() menjadi teks.
        # Keputusannya sendiri sudah diambil di dalam .predict() pada ambang 0,5.
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
                        <p style="color:#B23A54;font-size:0.88rem;margin:0.3rem 0;">
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
                        <div class="res-label" style="color:#3F9E6B;">{label_str}</div>
                        <p style="color:#4B6B54;font-size:0.88rem;margin:0.3rem 0;">
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
                if sv_lokal is not None:
                    fig = make_contribution_chart(selected_features,
                                                  sv_lokal[idx], top_n=15)
                    st.plotly_chart(fig, use_container_width=True, key=f"contrib_{idx}")
                else:
                    st.info("Grafik kontribusi SHAP tidak tersedia.")

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
                           else ("color:#2F7D52;font-weight:bold" if v == "Non-DIAD" else "")),
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
        <div class="hero-badge">⚡ MACHINE LEARNING &nbsp;·&nbsp; Cheminformatics &nbsp;·&nbsp; KEAMANAN OBAT</div>
        <div class="hero-title">Prediction <br> Drug Induced Autoimmune Disease (DIAD)</div>
        <div class="hero-sub">
            Prediksi risiko autoimunitas menggunakan <strong>XGBoost</strong> dan seleksi fitur
            <strong>Mutual Information</strong> pada {N_FEATURES_TOTAL} deskriptor molekuler RDKit.
            Model dilatih dengan strategi validasi <strong>80:20 </strong>
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
            background: linear-gradient(160deg, #F5FBF6 0%, #EAF7EE 100%);
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
                <td style="text-align:center;color:{'#2F7D52' if is_best else C_TEXT2};font-weight:{'700' if is_best else '400'};">{row['Fitur Terpilih']}</td>
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
            row_bg  = "background:rgba(63,158,107,0.08);" if is_best else ""
            tbl_rows += f"""
            <tr style="{row_bg}">
                <td><strong style="color:{C_TEXT};">{row['Model']}</strong>{badge}</td>
                <td><span class="badge-fs">{row['Seleksi Fitur']}</span></td>
                <td style="text-align:center;">{row['Jumlah Fitur']}</td>
                <td style="text-align:center;color:{'#2F7D52' if is_best else C_TEXT2};">{row['Akurasi']:.4f}</td>
                <td style="text-align:center;color:{'#2F7D52' if is_best else C_TEXT2};">{row['MCC']:.4f}</td>
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