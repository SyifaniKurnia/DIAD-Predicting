# =============================================================================
# utils.py
# Berisi seluruh fungsi pendukung pipeline prediksi risiko DIAD:
# preprocessing, seleksi fitur, tuning hyperparameter (GridSearchCV),
# evaluasi holdout & CV, visualisasi, dan SHAP.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
# shap: library untuk menghitung kontribusi tiap fitur ke prediksi (interpretasi model)
import shap

# -----------------------------------------------------------------------------
# PENGATURAN PERINGATAN
# -----------------------------------------------------------------------------
# Peringatan kosmetik disembunyikan agar keluaran notebook bersih.
# ConvergenceWarning SENGAJA TIDAK disembunyikan: peringatan itu memberi tahu
# bahwa solver SVM berhenti sebelum konvergen (max_iter=5000 tercapai).
# Kalau dibungkam, model tetap memberi prediksi — tetapi dari solusi yang
# belum konvergen, tanpa ada yang menyadarinya.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.model_selection import (
    StratifiedKFold, KFold,
    GridSearchCV,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, make_scorer
)


# =============================================================================
# Konstanta Global
# =============================================================================

METRICS         = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'ROC-AUC']
TARGET          = 'Label'
DROP            = ['Label', 'SMILES']
MODEL_NAMES     = ['SVM', 'Random Forest', 'XGBoost']
# Bungkus MCC jadi objek scorer sklearn (dipakai bila butuh scorer, mis. cross_validate)
MCC_SCORER      = make_scorer(matthews_corrcoef)
F1_MACRO_SCORER = make_scorer(f1_score, average='macro', zero_division=0)
# Jumlah fold untuk inner CV di dalam GridSearchCV (tuning), bukan outer CV
INNER_CV_FOLDS  = 3
# Seed tetap supaya hasil bisa direproduksi (angka sama tiap dijalankan)
RANDOM_STATE    = 42

# Grid hyperparameter untuk GridSearchCV
# SVM  : 30 kombinasi × 3 inner fold = 90 inner fits
# RF   : 36 kombinasi × 3 inner fold = 108 inner fits
# XGB  : 108 kombinasi × 3 inner fold = 324 inner fits
# Total per teknik FS per skenario: 90 + 108 + 324 = 522 inner fits

PARAM_GRID_SVM = [
    # ── Kernel Linear ───────────────────────────────────────────────────────
    # 5 kombinasi
    {
        'kernel': ['linear'],
        'C'     : [0.01, 0.1, 1, 10, 100],
    },
    # ── Kernel RBF ──────────────────────────────────────────────────────────
    # 10 kombinasi
    {
        'kernel': ['rbf'],
        'C'     : [0.01, 0.1, 1, 10, 100],
        'gamma' : ['scale', 'auto'],
    },
    # ── Kernel Polynomial ───────────────────────────────────────────────────
    # 15 kombinasi
    {
        'kernel': ['poly'],
        'C'     : [0.01, 0.1, 1, 10, 100],
        'degree': [2, 3, 4],
    },
]
# Total SVM: 5 + 10 + 15 = 30 kombinasi × 3 inner fold = 90 inner fits

PARAM_GRID_RF = {
    'n_estimators'     : [100, 200, 300],
    'max_depth'        : [5, 10, None],
    'max_features'     : ['sqrt', 'log2'],
    'min_samples_split': [2, 5],
}

PARAM_GRID_XGB = {
    'learning_rate'   : [0.01, 0.1, 0.3],
    'max_depth'       : [3, 6, 9],
    'n_estimators'    : [100, 200, 300],
    'subsample'       : [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0],
}

_MODEL_COLORS = [
    '#2196F3', '#FF9800', '#4CAF50',
    '#9C27B0', '#F44336', '#00BCD4',
    '#795548', '#607D8B', '#E91E63',
]


# =============================================================================
# Preprocessing
# =============================================================================

def fix_european(df):
    """
    Membersihkan format angka Eropa/Indonesia (koma sebagai desimal, titik
    sebagai pemisah ribuan) menjadi format yang dikenali Python, lalu
    mengonversi seluruh kolom numerik ke tipe angka.

    MENGAPA PERLU?
    CSV deskriptor yang disimpan lewat Excel berpengaturan lokal Indonesia
    menuliskan desimal dengan koma: "365,4" bukan "365.4". Python membaca
    koma sebagai teks, bukan angka, sehingga kolom deskriptor gagal
    dikonversi dan model tidak dapat memprosesnya.

    APA YANG SALAH PADA VERSI LAMA?
      (a) Versi lama HANYA membuang titik pemisah ribuan lewat regex
          r'\\.(?=\\d{3}\\.)' dan TIDAK PERNAH mengonversi koma menjadi titik.
          Akibatnya "365,4" berubah menjadi NaN, bukan 365.4.
      (b) Regex tersebut juga keliru: '1.234.567' menjadi 1234.567,
          bukan 1234567.
      (c) Syarat `df[col].dtype == object` hanya benar pada pandas 2.x.
          Sejak pandas 3.x kolom teks bertipe 'str', sehingga syarat itu
          tidak pernah terpenuhi dan konversi dilewati DIAM-DIAM.

    Fungsi ini identik dengan yang dipakai di app.py, sehingga representasi
    numerik deskriptor sama persis antara tahap pelatihan dan tahap prediksi.
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
            # Kolom teks murni (SMILES, Name) akan menghasilkan NaN seluruhnya;
            # kolom seperti itu dibiarkan utuh, tidak dirusak menjadi NaN.
            if konversi.notna().any():
                df[col] = konversi

        except Exception:
            pass  # kolom ini bukan angka — biarkan apa adanya

    return df


# =============================================================================
# Transformer Seleksi Fitur (sklearn Pipeline-compatible)
# =============================================================================

class ZeroDominantSelector(BaseEstimator, TransformerMixin):
    """
    Menghapus fitur yang memiliki satu nilai dominan melebihi threshold.
    Fitur dengan proporsi nilai terbanyak > threshold (default 80%) dianggap
    tidak informatif dan dibuang. fit() hanya dipanggil pada data training
    untuk mencegah data leakage.

    transformer diletakkan SEBELUM StandardScaler di dalam
    pipeline, karena proporsi nilai dominan harus dihitung dari data MENTAH.
    Setelah scaling, nilai-nilai yang semula identik tetap identik, tetapi
    interpretasinya menjadi kabur.
    """
    def __init__(self, threshold=0.80):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_ = pd.DataFrame(X)
        self.keep_mask_ = np.array([
            # True jika proporsi nilai terbanyak fitur <= 80% -> fitur DIPERTAHANKAN
            X_.iloc[:, i].value_counts(normalize=True).max() <= self.threshold
            for i in range(X_.shape[1])
        ])
        return self

    def transform(self, X, y=None):
        # Ambil hanya kolom yang lolos mask (keep_mask_ = True)
        return np.array(X)[:, self.keep_mask_]


class SpearmanSelector(BaseEstimator, TransformerMixin):
    """
    Menghapus salah satu fitur dari setiap pasangan yang memiliki korelasi
    Spearman absolut melebihi threshold (default 0.85). Fitur dengan varians
    nol ditangani secara internal sebelum perhitungan korelasi. fit() hanya
    dipanggil pada data training untuk mencegah data leakage.

    Korelasi Spearman berbasis PERINGKAT, sehingga tidak terpengaruh
    oleh transformasi monoton seperti Z-score. Menempatkannya sesudah
    StandardScaler karena itu aman — hasilnya identik dengan bila dihitung
    pada data mentah.
    """
    def __init__(self, threshold=0.85):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_df   = pd.DataFrame(X)
        n_orig = X_df.shape[1]

        # Tandai fitur yang variansnya > 0; fitur konstan diabaikan agar korelasi tidak error
        nonzero_var = X_df.var() > 0
        nz_indices  = [i for i, v in enumerate(nonzero_var) if v]
        X_nz        = X_df.loc[:, nonzero_var]

        # Matriks korelasi Spearman absolut antar fitur non-konstan
        cu = X_nz.corr(method='spearman').abs()
        # Ambil segitiga atas saja supaya tiap pasangan tidak dihitung dua kali
        cu = cu.where(np.triu(np.ones(cu.shape), k=1).astype(bool))
        drop_local = {j for j in range(X_nz.shape[1])
                      # Tandai fitur yang punya pasangan berkorelasi > 0,85 untuk dibuang
                      if any(cu.iloc[:, j] > self.threshold)}

        keep_orig = {nz_indices[j] for j in range(len(nz_indices))
                     if j not in drop_local}

        # Susun mask akhir mengikuti urutan kolom asli (termasuk yang konstan = dibuang)
        self.keep_mask_ = np.array([i in keep_orig for i in range(n_orig)])
        return self

    def transform(self, X, y=None):
        return np.array(X)[:, self.keep_mask_]


class MutualInfoSelector(BaseEstimator, TransformerMixin):
    """
    Menghapus fitur yang memiliki skor Mutual Information terhadap label
    di bawah atau sama dengan threshold (default 0.01). Skor MI mengukur
    seberapa besar informasi yang dibagi antara fitur dengan variabel target.
    fit() menerima label y dari training fold sehingga aman digunakan dalam
    Pipeline CV tanpa data leakage.

    MI juga berbasis distribusi, bukan skala, sehingga aman ditempatkan sesudah StandardScaler.
    """
    def __init__(self, threshold=0.01, random_state=RANDOM_STATE):
        self.threshold    = threshold
        self.random_state = random_state

    def fit(self, X, y):
        # Hitung skor MI tiap fitur terhadap label (butuh y, makanya fit menerima y)
        mi = mutual_info_classif(X, y, random_state=self.random_state)
        # Pertahankan fitur dengan MI > 0,01 (di atas ambang informatif)
        self.keep_mask_ = mi > self.threshold
        return self

    def transform(self, X, y=None):
        return np.array(X)[:, self.keep_mask_]


# =============================================================================
# Evaluasi
# =============================================================================

def evaluate(y_true, y_pred, y_proba=None):
    """
    Menghitung enam metrik evaluasi klasifikasi: Accuracy, Precision (macro),
    Recall (macro), F1-Score (macro), MCC, dan ROC-AUC. ROC-AUC hanya dihitung
    jika probabilitas prediksi (y_proba) tersedia.

    PENTING — average='macro':
    Precision, Recall, dan F1 dihitung sebagai RATA-RATA KEDUA KELAS, bukan
    hanya kelas DIAD. Jadi Recall = 0,8056 pada model terbaik adalah
    macro-recall, sedangkan recall kelas DIAD saja bernilai sekitar 70%.
    Perbedaan ini WAJIB dinyatakan di naskah agar tidak dikira salah salin.

    MCC dipilih sebagai metrik utama karena rasio kelas timpang (~3:1);
    MCC memperhitungkan keempat sel confusion matrix sekaligus.
    """
    res = {
        'Accuracy' : accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall'   : recall_score(y_true, y_pred, average='macro', zero_division=0),
        'F1-Score' : f1_score(y_true, y_pred, average='macro', zero_division=0),
        'MCC'      : matthews_corrcoef(y_true, y_pred),
    }
    # ROC-AUC hanya bisa dihitung kalau probabilitas prediksi tersedia
    if y_proba is not None:
        res['ROC-AUC'] = roc_auc_score(y_true, y_proba)
    return res


def diagnose_fit(train_score, test_score, gap_thr=0.10, low_thr=0.75):
    """
    Mendiagnosis kondisi fitting model berdasarkan selisih skor train dan test.
    Mengembalikan salah satu dari tiga kondisi: 'Good Fit', 'Overfitting',
    atau 'Underfitting'.
    """
    # Gap = selisih performa latih vs uji; makin besar makin overfitting
    gap = train_score - test_score
    # Kalau dua-duanya rendah (<0,75) -> model underfitting (belum belajar)
    if train_score < low_thr and test_score < low_thr:
        return 'Underfitting'
    if gap > gap_thr:
        return 'Overfitting'
    return 'Good Fit'


# =============================================================================
# Tuning Hyperparameter — GridSearchCV
# =============================================================================

def _inner_cv():
    """
    Membuat objek StratifiedKFold untuk inner CV pada proses tuning
    hyperparameter. Menggunakan jumlah fold dari konstanta INNER_CV_FOLDS.
    """
    # Inner CV stratified: jaga rasio kelas di tiap fold saat tuning
    return StratifiedKFold(n_splits=INNER_CV_FOLDS,
                           shuffle=True, random_state=RANDOM_STATE)


def tune_svm(X, y):
    """
    Mencari hyperparameter terbaik untuk SVM menggunakan GridSearchCV dengan
    inner StratifiedKFold(3) dan scoring MCC. Mengevaluasi 30 kombinasi
    (5 linear + 10 RBF + 15 Poly).

    PERHATIAN — max_iter=5000:
    Bila solver belum konvergen dalam 5.000 iterasi, sklearn akan menerbitkan
    ConvergenceWarning. Peringatan itu SENGAJA TIDAK dibungkam pada versi
    utils.py ini. Bila peringatan tersebut muncul, artinya SVM memberi
    prediksi dari solusi yang belum konvergen — dan hal itu perlu dilaporkan
    sebagai keterbatasan, bukan disembunyikan.
    """
    base_clf = SVC(
        # Bobot kelas dibalik proporsional agar kelas minoritas DIAD tidak terabaikan.
        # class_weight bekerja pada penalti hinge loss / kriteria margin,
        # BUKAN pada ambang keputusan akhir (yang tetap 0,5).
        class_weight='balanced',
        probability=True,
        random_state=RANDOM_STATE,
        max_iter=5000,
    )
    gs = GridSearchCV(
        estimator  = base_clf,
        param_grid = PARAM_GRID_SVM,
        cv         = _inner_cv(),
        # Hyperparameter dipilih berdasarkan MCC, bukan akurasi (cocok data timpang)
        scoring    = 'matthews_corrcoef',
        n_jobs     = -1,
        refit      = True,
    )
    # Jalankan grid search: coba semua kombinasi lalu fit ulang yang terbaik (refit=True)
    gs.fit(X, y)
    # Kembalikan model dengan kombinasi hyperparameter terbaik
    return gs.best_estimator_


def tune_rf(X, y):
    """
    Mencari hyperparameter terbaik untuk Random Forest menggunakan GridSearchCV
    dengan inner StratifiedKFold(3) dan scoring MCC. Mengevaluasi 36 kombinasi.
    """
    base_clf = RandomForestClassifier(
        # class_weight='balanced' membobot impuritas Gini saat pohon tumbuh.
        # Ia TIDAK menggeser ambang keputusan 0,5. Inilah yang menjelaskan
        # mengapa RF mencatat ROC-AUC tertinggi tetapi MCC lebih rendah:
        # peringkat probabilitasnya baik, tetapi titik potongnya kurang tepat.
        class_weight='balanced',
        random_state=RANDOM_STATE,
    )
    gs = GridSearchCV(
        estimator  = base_clf,
        param_grid = PARAM_GRID_RF,
        cv         = _inner_cv(),
        scoring    = 'matthews_corrcoef',
        n_jobs     = -1,
        refit      = True,
    )
    gs.fit(X, y)
    return gs.best_estimator_


def tune_xgb(X, y):
    """
    Mencari hyperparameter terbaik untuk XGBoost menggunakan GridSearchCV
    dengan inner StratifiedKFold(3) dan scoring MCC. Mengevaluasi 108 kombinasi.

    scale_pos_weight DIHITUNG ULANG dari y yang diterima fungsi ini. Karena
    tune_model() dipanggil dengan (X_tr, y_tr) di dalam setiap outer fold,
    maka scale_pos_weight benar-benar dihitung PER FOLD PELATIHAN — bukan
    sekali di awal. Klaim ini dinyatakan pada BAB IV dan kodenya konsisten.
    """
    # Rasio kelas negatif:positif (~3:1). Dipakai XGBoost untuk menyeimbangkan kelas.
    # Berbeda dari class_weight pada RF/SVM: scale_pos_weight masuk ke GRADIEN
    # logistic loss di setiap iterasi boosting, sehingga distribusi probabilitas
    # keluaran ikut bergeser — bukan hanya kriteria pemisahan simpul.
    scale_pw = float((y == 0).sum()) / float((y == 1).sum())

    base_clf = XGBClassifier(
        scale_pos_weight=scale_pw,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    gs = GridSearchCV(
        estimator  = base_clf,
        param_grid = PARAM_GRID_XGB,
        cv         = _inner_cv(),
        scoring    = 'matthews_corrcoef',
        n_jobs     = -1,
        refit      = True,
    )
    gs.fit(X, y)
    return gs.best_estimator_


def tune_model(model_name, X, y):
    """
    Dispatcher yang memanggil fungsi tuning sesuai nama model ('SVM',
    'Random Forest', atau 'XGBoost'). Seluruh proses tuning dan fitting
    dilakukan hanya pada X dan y yang diberikan — yaitu data latih saja.
    """
    if model_name == 'SVM':
        return tune_svm(X, y)
    elif model_name == 'Random Forest':
        return tune_rf(X, y)
    elif model_name == 'XGBoost':
        return tune_xgb(X, y)
    else:
        # Cegah salah ketik nama model dengan melempar error yang jelas
        raise ValueError(f'Model tidak dikenal: {model_name}')


# =============================================================================
# Tabel Per-Fold
# =============================================================================

def show_fold_table(fold_details, label):
    """
    Menampilkan tabel nilai metrik per fold untuk setiap kombinasi teknik
    seleksi fitur dan model. Setiap tabel memiliki baris per fold, baris
    Mean, dan baris Std. Secara otomatis menggunakan display() jika berjalan
    di lingkungan IPython/Colab/Kaggle, atau print() jika tidak.

    Parameter
    ----------
    fold_details : dict — keluaran kedua dari run_cv_tuned(),
                   berformat {(fs_name, model_name): pd.DataFrame}
    label        : str  — nama skenario, misal 'S3: 5-Fold CV'
    """
    try:
        from IPython.display import display as _ipy_display
        _use_display = True
    except ImportError:
        _use_display = False

    sep = '=' * 72
    print(f'\n{sep}')
    print(f'  TABEL PER-FOLD — {label}')
    print(f'{sep}')

    for (fs_name, model_name), fold_df in fold_details.items():
        print(f'\n{"-" * 72}')
        print(f'  [{fs_name}]  ×  [{model_name}]')
        print(f'{"-" * 72}')
        if _use_display:
            _ipy_display(fold_df.style.format('{:.4f}', na_rep='N/A')
                         .set_caption(f'{fs_name} × {model_name}')
                         .set_table_styles([{
                             'selector': 'caption',
                             'props': [('font-weight', 'bold'), ('font-size', '12px')]
                         }]))
        else:
            print(fold_df.to_string(float_format=lambda x: f'{x:.4f}'))
        print()


# =============================================================================
# Evaluasi Holdout dengan Tuning GridSearchCV
# =============================================================================

def run_nocv_tuned(fs_dict, y_train, y_test, label):
    """
    Menjalankan evaluasi strategi holdout dengan tuning hyperparameter via
    GridSearchCV. Untuk setiap kombinasi teknik seleksi fitur dan model,
    fungsi ini melakukan tuning pada data training saja, kemudian mengevaluasi
    model terbaik pada data test.

    Parameter
    ----------
    fs_dict : dict — output dari make_feature_sets_split(),
              berformat {nama_fs: (Xtr, Xte)}
    y_train : pd.Series — label training
    y_test  : pd.Series — label test
    label   : str       — nama skenario untuk kolom 'Skenario'
    """
    rows = []
    for fs_name, (Xtr, Xte) in fs_dict.items():
        for model_name in MODEL_NAMES:
            # Tuning + fit HANYA di data latih (data uji tidak disentuh -> anti-leakage)
            clf = tune_model(model_name, Xtr, y_train)

            y_pred_tr  = clf.predict(Xtr)
            y_pred_te  = clf.predict(Xte)
            # Ambil probabilitas kelas 1 (DIAD) untuk ROC-AUC, jika model mendukungnya
            y_prob_te  = clf.predict_proba(Xte)[:, 1] \
                         if hasattr(clf, 'predict_proba') else None

            train_acc = accuracy_score(y_train, y_pred_tr)
            test_acc  = accuracy_score(y_test,  y_pred_te)
            train_f1  = f1_score(y_train, y_pred_tr, average='macro', zero_division=0)
            test_f1   = f1_score(y_test,  y_pred_te, average='macro', zero_division=0)
            eval_dict = evaluate(y_test, y_pred_te, y_prob_te)

            rows.append({
                'Skenario'          : label,
                'Feature Selection' : fs_name,
                'Model'             : model_name,
                'Train Accuracy'    : round(train_acc, 4),
                'Test Accuracy'     : round(test_acc,  4),
                'Gap Accuracy'      : round(train_acc - test_acc, 4),
                'Train Macro F1'    : round(train_f1,  4),
                'Test Macro F1'     : round(test_f1,   4),
                'Gap Macro F1'      : round(train_f1 - test_f1, 4),
                # PERBAIKAN: diagnose_fit() kini benar-benar dipakai, tidak lagi
                # menjadi kode mati. Hasilnya masuk ke kolom 'Diagnosis'.
                'Diagnosis'         : diagnose_fit(train_acc, test_acc),
                # Gabungkan semua metrik evaluate() ke dalam baris hasil, dibulatkan 4 desimal
                **{k: round(v, 4) for k, v in eval_dict.items()}
            })
    return rows


# =============================================================================
# Evaluasi CV dengan Tuning GridSearchCV (Nested CV)
# =============================================================================

def run_cv_tuned(X, fs_dict, y, n_splits, label, stratified=True):
    """
    Menjalankan evaluasi cross-validation dengan tuning hyperparameter
    GridSearchCV di setiap fold menggunakan struktur Nested CV yang bebas
    data leakage. Alur per fold: (1) split outer, (2) preprocessing di-fit
    hanya pada training fold, (3) GridSearchCV dengan inner 3-fold CV,
    (4) evaluasi model terbaik pada validation fold.

    Parameter stratified mengontrol jenis outer CV:
      True  → StratifiedKFold (S9–S14)
      False → KFold biasa (S3–S8)

    Parameter
    ----------
    X          : pd.DataFrame — matriks fitur mentah
    fs_dict    : dict         — output dari make_feature_sets_cv(),
                 berformat {nama_fs: (Pipeline, X_raw)}
    y          : pd.Series    — label target
    n_splits   : int          — jumlah fold outer CV
    label      : str          — nama skenario untuk kolom 'Skenario'
    stratified : bool         — True untuk StratifiedKFold, False untuk KFold

    Returns
    -------
    rows         : list of dict   — ringkasan metrik rata-rata per kombinasi
    fold_details : dict           — {(fs_name, model_name): pd.DataFrame}
                   DataFrame berisi metrik per fold + baris Mean dan Std
    """
    # Outer CV: stratified atau biasa, tergantung parameter stratified
    outer_cv = (StratifiedKFold(n_splits=n_splits, shuffle=True,
                                random_state=RANDOM_STATE)
                if stratified
                else KFold(n_splits=n_splits, shuffle=True,
                           random_state=RANDOM_STATE))
    rows         = []
    fold_details = {}

    for fs_name, (pipe, _) in fs_dict.items():
        for model_name in MODEL_NAMES:

            tr_acc_list  = []
            val_acc_list = []
            tr_f1_list   = []
            val_f1_list  = []
            mcc_list     = []
            auc_list     = []
            prec_list    = []
            rec_list     = []
            fold_records = []

            # StratifiedKFold butuh y untuk menjaga rasio kelas; KFold biasa cukup X
            split_args = (X, y) if stratified else (X,)

            for fold_idx, (train_idx, val_idx) in enumerate(
                    outer_cv.split(*split_args)):

                # Ambil data latih untuk fold ini berdasarkan indeks dari outer_cv
                X_tr_raw  = X.iloc[train_idx].reset_index(drop=True)
                X_val_raw = X.iloc[val_idx].reset_index(drop=True)
                y_tr      = y.iloc[train_idx].reset_index(drop=True)
                y_val     = y.iloc[val_idx].reset_index(drop=True)

                # Salin pipeline bersih tiap fold agar scaler/seleksi fitur di-fit ulang dari nol
                p     = clone(pipe)
                # Fit + transform pipeline HANYA di training fold (kunci anti-leakage)
                X_tr  = p.fit_transform(X_tr_raw, y_tr)
                # Validation fold cuma di-transform pakai parameter dari training fold
                X_val = p.transform(X_val_raw)

                # Tuning hyperparameter (inner CV) di dalam training fold ini -> Nested CV.
                # scale_pos_weight untuk XGBoost juga dihitung ulang di sini,
                # berdasarkan komposisi kelas pada training fold ini saja.
                clf = tune_model(model_name, X_tr, y_tr)

                y_pred_tr  = clf.predict(X_tr)
                y_pred_val = clf.predict(X_val)
                y_prob_val = (clf.predict_proba(X_val)[:, 1]
                              if hasattr(clf, 'predict_proba') else None)

                acc_tr = accuracy_score(y_tr,  y_pred_tr)
                f1_tr  = f1_score(y_tr,  y_pred_tr,  average='macro',
                                  zero_division=0)

                acc_val  = accuracy_score(y_val, y_pred_val)
                f1_val   = f1_score(y_val, y_pred_val, average='macro',
                                    zero_division=0)
                mcc_val  = matthews_corrcoef(y_val, y_pred_val)
                prec_val = precision_score(y_val, y_pred_val,
                                           average='macro', zero_division=0)
                rec_val  = recall_score(y_val, y_pred_val,
                                        average='macro', zero_division=0)

                # Default AUC = NaN; diisi kalau probabilitas tersedia dan perhitungannya sukses
                auc_val = np.nan
                if y_prob_val is not None:
                    try:
                        auc_val = roc_auc_score(y_val, y_prob_val)
                    except Exception:
                        pass

                tr_acc_list.append(acc_tr)
                val_acc_list.append(acc_val)
                tr_f1_list.append(f1_tr)
                val_f1_list.append(f1_val)
                mcc_list.append(mcc_val)
                prec_list.append(prec_val)
                rec_list.append(rec_val)
                if not np.isnan(auc_val):
                    auc_list.append(auc_val)

                fold_records.append({
                    'Fold'     : fold_idx + 1,
                    'Accuracy' : round(acc_val,  4),
                    'Precision': round(prec_val, 4),
                    'Recall'   : round(rec_val,  4),
                    'F1-Score' : round(f1_val,   4),
                    'MCC'      : round(mcc_val,  4),
                    'ROC-AUC'  : (round(auc_val, 4)
                                  if not np.isnan(auc_val) else np.nan),
                })

            fold_df   = pd.DataFrame(fold_records).set_index('Fold')
            mean_row  = fold_df.mean().round(4).to_frame().T
            # Std antar-fold (ddof=1 = sample std) sebagai ukuran kestabilan metrik
            std_row   = fold_df.std(ddof=1).round(4).to_frame().T
            mean_row.index = pd.Index(['Mean'], name='Fold')
            std_row.index  = pd.Index(['Std'],  name='Fold')
            fold_df = pd.concat([fold_df, mean_row, std_row])
            fold_details[(fs_name, model_name)] = fold_df

            tr_acc  = float(np.mean(tr_acc_list))
            val_acc = float(np.mean(val_acc_list))
            tr_f1   = float(np.mean(tr_f1_list))
            val_f1  = float(np.mean(val_f1_list))

            rows.append({
                'Skenario'          : label,
                'Feature Selection' : fs_name,
                'Model'             : model_name,
                'Train Accuracy'    : round(tr_acc,  4),
                'Test Accuracy'     : round(val_acc, 4),
                'Gap Accuracy'      : round(tr_acc - val_acc, 4),
                'Train Macro F1'    : round(tr_f1,  4),
                'Test Macro F1'     : round(val_f1, 4),
                'Gap Macro F1'      : round(tr_f1 - val_f1, 4),
                # PERBAIKAN: diagnose_fit() kini dipakai juga pada jalur CV.
                'Diagnosis'         : diagnose_fit(tr_acc, val_acc),
                # Kolom 'Accuracy' sengaja diisi akurasi validasi agar konsisten dengan skema holdout
                'Accuracy'          : round(val_acc, 4),
                'Precision'         : round(float(np.mean(prec_list)), 4),
                'Recall'            : round(float(np.mean(rec_list)),  4),
                'F1-Score'          : round(val_f1, 4),
                'MCC'               : round(float(np.mean(mcc_list)),  4),
                'ROC-AUC'           : (round(float(np.mean(auc_list)), 4)
                                       if auc_list else np.nan),
            })

    return rows, fold_details


# =============================================================================
# Pembuatan Feature Sets — Holdout
# =============================================================================

def make_feature_sets_split(Xtr_raw, Xte_raw, ytr):
    """
    Membuat lima set fitur untuk skenario holdout, masing-masing menggunakan
    teknik seleksi fitur yang berbeda. Seluruh proses fit (StandardScaler,
    VarianceThreshold, SpearmanSelector, MutualInfoSelector) hanya dilakukan
    pada Xtr_raw untuk mencegah data leakage dari data test.

    URUTAN OPERASI PER TEKNIK — dan alasannya:
      Baseline            : StandardScaler saja
      Variance Threshold  : VT pada data MENTAH → StandardScaler
                            (varians harus dihitung sebelum diseragamkan;
                             setelah Z-score semua varians menjadi 1)
      Zero Dominant       : ZDE pada data MENTAH → StandardScaler
                            (proporsi nilai dominan bermakna pada data asli)
      Spearman            : StandardScaler → SpearmanSelector
                            (Spearman berbasis peringkat, tidak terpengaruh skala)
      Mutual Information  : StandardScaler → MI threshold > 0.01
                            (MI berbasis distribusi, tidak terpengaruh skala)

    Urutan ini IDENTIK dengan pipeline pada make_feature_sets_cv().

    Mengembalikan dictionary {nama_fs: (Xtr_seleksi, Xte_seleksi)}.
    """
    sc  = StandardScaler()
    # fit_transform di data latih: scaler BELAJAR mean & std lalu menstandarkan
    Xtr = pd.DataFrame(sc.fit_transform(Xtr_raw), columns=Xtr_raw.columns)
    # transform saja di data uji: pakai mean & std dari data latih (anti-leakage)
    Xte = pd.DataFrame(sc.transform(Xte_raw),     columns=Xte_raw.columns)

    vt_    = VarianceThreshold(threshold=0.01)
    # VarianceThreshold dihitung dari data MENTAH (sebelum scaling)
    vt_.fit(Xtr_raw)
    sel_vt = Xtr_raw.columns[vt_.get_support()].tolist()
    Xtr_vt = Xtr[sel_vt].reset_index(drop=True)
    Xte_vt = Xte[sel_vt].reset_index(drop=True)

    # Daftar fitur zero-dominant: >80% nilainya didominasi satu angka -> dibuang
    zd = [c for c in Xtr_raw.columns
          if Xtr_raw[c].value_counts(normalize=True).max() > 0.80]

    sp_sel = SpearmanSelector(threshold=0.85)
    # Spearman dihitung SETELAH scaling (aman: berbasis peringkat)
    sp_sel.fit(Xtr.values)
    Xtr_sp = Xtr.iloc[:, sp_sel.keep_mask_].reset_index(drop=True)
    Xte_sp = Xte.iloc[:, sp_sel.keep_mask_].reset_index(drop=True)

    # Skor MI dihitung dari data latih ter-scale terhadap label latih.
    # RANDOM_STATE dipakai (bukan angka 42 yang dituliskan langsung) agar
    # seed tetap konsisten bila konstanta globalnya suatu saat diubah.
    mi_    = mutual_info_classif(Xtr, ytr, random_state=RANDOM_STATE)
    sel_mi = Xtr.columns[mi_ > 0.01].tolist()

    return {
        'Baseline'          : (Xtr.copy(), Xte.copy()),
        'Variance Threshold': (Xtr_vt, Xte_vt),
        'Zero Dominant'     : (Xtr.drop(columns=zd).reset_index(drop=True),
                               Xte.drop(columns=zd).reset_index(drop=True)),
        'Spearman'          : (Xtr_sp, Xte_sp),
        'Mutual Information': (Xtr[sel_mi].reset_index(drop=True),
                               Xte[sel_mi].reset_index(drop=True)),
    }


# =============================================================================
# Pembuatan Feature Sets — CV (Pipeline-based)
# =============================================================================

def make_feature_sets_cv(X_raw):
    """
    Membuat lima Pipeline sklearn untuk digunakan dalam skenario cross-validation.
    Berbeda dari make_feature_sets_split(), fungsi ini mengembalikan Pipeline
    yang siap di-fit per fold di dalam run_cv_tuned(), sehingga seleksi fitur
    hanya menggunakan informasi dari training fold dan bebas data leakage.

    Urutan pipeline per teknik — IDENTIK dengan make_feature_sets_split():
      Baseline            : StandardScaler
      Variance Threshold  : VarianceThreshold(0.01) → StandardScaler
      Zero Dominant       : ZeroDominantSelector(0.80) → StandardScaler
      Spearman            : StandardScaler → SpearmanSelector(0.85)
      Mutual Information  : StandardScaler → MutualInfoSelector(0.01)

    Mengembalikan dictionary {nama_fs: (Pipeline, X_raw)}.
    """
    pipelines = {
        'Baseline'          : Pipeline([
            ('scaler', StandardScaler())
        ]),
        'Variance Threshold': Pipeline([
            ('vt',     VarianceThreshold(0.01)),
            ('scaler', StandardScaler())
        ]),
        'Zero Dominant'     : Pipeline([
            ('zd',     ZeroDominantSelector(0.80)),
            ('scaler', StandardScaler())
        ]),
        'Spearman'          : Pipeline([
            ('scaler', StandardScaler()),
            ('sp',     SpearmanSelector(0.85))
        ]),
        'Mutual Information': Pipeline([
            ('scaler', StandardScaler()),
            ('mi',     MutualInfoSelector(0.01))
        ]),
    }
    # Kembalikan pipeline (belum di-fit) + data mentah; fit dilakukan per fold nanti
    return {name: (pipe, X_raw) for name, pipe in pipelines.items()}


# =============================================================================
# Visualisasi
# =============================================================================

def plot_mcc_heatmap(df_results, models):
    """
    Membuat heatmap MCC untuk setiap skenario validasi. Sumbu baris adalah
    teknik seleksi fitur dan sumbu kolom adalah nama model. Nilai MCC
    ditampilkan di setiap sel dengan skala warna RdYlGn.
    """
    scen_list = list(df_results['Skenario'].unique())
    n_scen    = len(scen_list)
    fig, axes = plt.subplots(1, n_scen,
                             figsize=(5.5 * n_scen, max(5, len(models) * 0.7)))
    if n_scen == 1:
        axes = [axes]
    for ax, scen in zip(axes, scen_list):
        # Susun ulang jadi tabel: baris = teknik FS, kolom = model, isi = MCC
        pivot = df_results[df_results['Skenario'] == scen].pivot_table(
            index='Feature Selection', columns='Model', values='MCC')
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn',
                    ax=ax, linewidths=0.5, vmin=0, vmax=1,
                    cbar_kws={'label': 'MCC'})
        ax.set_title(scen, fontsize=10, fontweight='bold')
        ax.set_xlabel(''); ax.tick_params(axis='x', rotation=35)
    plt.suptitle('Perbandingan MCC per Skenario & Model',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.show()


def plot_mcc_bar(df_results, models):
    """
    Membuat bar chart MCC terbaik per skenario untuk setiap model. Menampilkan
    nilai MCC tertinggi yang dicapai masing-masing model di setiap skenario
    validasi, terlepas dari teknik seleksi fitur yang digunakan.
    """
    scen_list   = list(df_results['Skenario'].unique())
    model_names = list(models.keys())
    n_models    = len(model_names)
    colors      = _MODEL_COLORS[:n_models]
    # Ambil MCC tertinggi tiap (skenario, model), lepas dari teknik FS-nya
    best_per    = (df_results.groupby(['Skenario', 'Model'])['MCC']
                   .max().reset_index())
    x = np.arange(len(scen_list))
    w = min(0.8 / n_models, 0.25)
    fig, ax = plt.subplots(figsize=(max(13, len(scen_list) * 1.5), 5))
    for i, (m_name, c) in enumerate(zip(model_names, colors)):
        subset = best_per[best_per['Model'] == m_name]
        vals   = [subset[subset['Skenario'] == s]['MCC'].values[0]
                  if len(subset[subset['Skenario'] == s]) > 0 else 0
                  for s in scen_list]
        bars_ = ax.bar(x + i * w, vals, w, label=m_name,
                       color=c, edgecolor='white', alpha=0.88)
        for b, v in zip(bars_, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.007,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x + w * (n_models - 1) / 2)
    ax.set_xticklabels(scen_list, rotation=20, ha='right')
    ax.set_ylabel('MCC Terbaik per Teknik FS')
    ax.set_title('Perbandingan MCC Terbaik — Seluruh Skenario & Model',
                 fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.legend(title='Model', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3); plt.tight_layout(); plt.show()


def plot_gap_analysis(df_results, metric='Accuracy'):
    """
    Membuat bar chart analisis gap (selisih train-test) untuk setiap kombinasi
    teknik seleksi fitur dan model per skenario. Warna bar merepresentasikan
    kondisi fitting: hijau (Good Fit), merah (Overfitting), oranye (Underfitting).
    """
    gap_col = f'Gap {metric}' if metric == 'Accuracy' else 'Gap Macro F1'
    if gap_col not in df_results.columns:
        print(f"Kolom '{gap_col}' tidak ditemukan."); return

    # Tentukan warna bar: merah=overfitting, oranye=underfitting, hijau=good fit
    def _color(gap):
        if gap > 0.10:    return '#F44336'
        elif gap < -0.05: return '#FF9800'
        else:             return '#4CAF50'

    n_scen = df_results['Skenario'].nunique()
    fig, axes = plt.subplots(1, n_scen, figsize=(5 * n_scen, 4), sharey=False)
    if n_scen == 1: axes = [axes]
    for ax, (scen, sub) in zip(axes, df_results.groupby('Skenario')):
        x      = np.arange(len(sub))
        colors = [_color(g) for g in sub[gap_col]]
        ax.bar(x, sub[gap_col], color=colors, edgecolor='white', alpha=0.88)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{r['Feature Selection']}\n{r['Model']}" for _, r in sub.iterrows()],
            fontsize=7, rotation=45, ha='right')
        ax.set_title(scen, fontsize=9, fontweight='bold')
        ax.set_ylabel(f'Gap {metric} (Train - Test)')
        ax.grid(axis='y', alpha=0.3)
    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor='#4CAF50', label='Good Fit  (gap ≤ 0.10)'),
        Patch(facecolor='#F44336', label='Overfitting (gap > 0.10)'),
        Patch(facecolor='#FF9800', label='Underfitting (gap < −0.05)'),
    ], loc='upper right', title='Diagnosis', bbox_to_anchor=(1.01, 1))
    plt.suptitle(f'Analisis Gap — {metric}',
                 fontsize=11, fontweight='bold', y=1.04)
    plt.tight_layout(); plt.show()


# =============================================================================
# SHAP
# =============================================================================

def compute_shap(model_shap, Xtr_shap, best_model):
    """
    Menghitung SHAP values untuk model terbaik guna interpretasi fitur.
    Untuk model berbasis pohon (Random Forest, XGBoost) digunakan TreeExplainer.
    Untuk SVM digunakan KernelExplainer dengan subsample 200 sampel dan
    100 background. Mengembalikan tuple (sv, Xsv) berisi SHAP values dan
    data yang digunakan untuk menghitungnya.

    INTERPRETASI:
    SHAP menunjukkan SEBERAPA BESAR suatu deskriptor menggeser keluaran model,
    BUKAN mengapa secara biologis. Nilai SHAP tinggi tidak membuktikan
    hubungan kausal antara deskriptor dan mekanisme DIAD.
    """
    # Cek apakah model berbasis pohon; menentukan jenis explainer SHAP
    is_tree = any(t in best_model for t in ['Random Forest', 'XGBoost'])
    if is_tree:
        # TreeExplainer: cepat & eksak untuk model pohon (dipakai model terbaik)
        explainer   = shap.TreeExplainer(model_shap)
        shap_values = explainer.shap_values(Xtr_shap)
        # Ambil SHAP untuk kelas positif (DIAD); tangani beda format antar versi shap
        sv  = shap_values[1] if isinstance(shap_values, list) else shap_values
        if np.array(sv).ndim == 3:
            sv = np.array(sv)[:, :, 1]
        Xsv = Xtr_shap.reset_index(drop=True)
    else:
        Xsv        = Xtr_shap.iloc[:200].reset_index(drop=True)
        background = shap.sample(Xtr_shap, 100, random_state=RANDOM_STATE)
        # KernelExplainer untuk SVM (non-pohon): lebih lambat, pakai 100 sampel background
        explainer  = shap.KernelExplainer(model_shap.predict_proba, background)
        sv = np.array(explainer.shap_values(Xsv, nsamples=100))
        if sv.ndim == 3:   sv = sv[:, :, 1]
        elif sv.ndim == 1: sv = sv[1]
    print(f"sv shape : {sv.shape} | Xsv shape: {Xsv.shape}")
    return sv, Xsv


def plot_shap_beeswarm(sv, Xsv, best_model, best_fs,
                       save_path='shap_beeswarm.png'):
    """
    Membuat SHAP Beeswarm Plot yang menampilkan distribusi nilai SHAP per fitur.
    Setiap titik mewakili satu sampel; warna menunjukkan nilai fitur asli.
    Hanya menampilkan 20 fitur dengan kontribusi SHAP terbesar.
    Gambar disimpan ke save_path dengan resolusi 300 DPI.
    """
    # Beeswarm 20 fitur teratas: tiap titik = 1 senyawa, warna = nilai fitur asli
    shap.summary_plot(sv, Xsv, max_display=20, show=False)
    plt.title(f"SHAP — {best_model} + {best_fs} (Top 20)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(); print(f"  -> {save_path}")


def plot_shap_bar(sv, Xsv, best_model, best_fs, save_path='shap_bar.png'):
    """
    Membuat SHAP Bar Chart yang menampilkan rata-rata nilai SHAP absolut per
    fitur sebagai ukuran kepentingan global. Hanya menampilkan 20 fitur dengan
    kontribusi terbesar. Gambar disimpan ke save_path dengan resolusi 300 DPI.
    """
    # Bar chart: ranking fitur berdasarkan rata-rata |SHAP| (kepentingan global)
    shap.summary_plot(sv, Xsv, plot_type='bar', max_display=20, show=False)
    plt.title(f"SHAP Importance — {best_model} + {best_fs} (Top 20)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(); print(f"  -> {save_path}")