"""
═══════════════════════════════════════════════════════════════════════════════
 SKRIP 1 — PENGAMBILAN SMILES DARI PUBCHEM (PUG-REST API)
═══════════════════════════════════════════════════════════════════════════════
 Skrip ini mengambil struktur kimia (notasi SMILES) untuk 20 senyawa obat
 yang akan dipakai sebagai HIMPUNAN VALIDASI EKSTERNAL — yaitu senyawa yang
 sama sekali tidak pernah dilihat model, baik saat pelatihan maupun pengujian
 internal.

 Model tidak bisa membaca nama obat. Model hanya membaca angka. Rantai
 perubahannya seperti ini:

     "Amoxicillin"  ← nama, tidak berarti apa-apa bagi model
          │  ← SKRIP INI (PubChem PUG-REST API)
          ▼
     CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O    ← SMILES
          │  ← SKRIP 2 (RDKit)
          ▼
     196 angka: MolWt=365.4, SlogP_VSA10=12.3, fr_NH2=1, ...
          │  ← app.py
          ▼
     P(DIAD) = 0,912  →  "DIAD"

 SMILES adalah jembatan antara nama obat dan angka. Tanpa langkah ini,
 deskriptor RDKit tidak bisa dihitung sama sekali.

 MENGAPA PUBCHEM, BUKAN MENGETIK SENDIRI?
 Mengetik SMILES secara manual sangat rawan salah — satu kurung keliru saja
 sudah menghasilkan molekul yang berbeda. 
 PubChem adalah basis data kimia milik NIH (Amerika Serikat) yang datanya terkurasi. 
 Mengambilnya lewat API menjamin struktur yang dipakai benar dan dapat ditelusuri kembali lewat CID.

 Skrip ini menyimpan kolom CID (PubChem Compound ID). 
 CID adalah nomor unik permanen. 
 Siapa pun yang ingin memverifikasi penelitian ini dapat membuka
 https://pubchem.ncbi.nlm.nih.gov/compound/<CID> dan mendapatkan struktur
 yang persis sama.
═══════════════════════════════════════════════════════════════════════════════
"""

import time       # untuk memberi jeda antarpermintaan agar tidak diblokir PubChem
import requests   # untuk mengirim permintaan HTTP ke server PubChem
import pandas as pd   # untuk menyusun hasil menjadi tabel dan menyimpannya ke CSV


# ═════════════════════════════════════════════════════════════════════════════
#  DAFTAR SENYAWA VALIDASI EKSTERNAL
# ═════════════════════════════════════════════════════════════════════════════
#  20 senyawa: 10 berlabel DIAD-positif, 10 berlabel Non-DIAD.
# ─────────────────────────────────────────────────────────────────────────────
LABEL_AWAL = {
    # --- 10 senyawa DIAD-positif ---
    "Cilazapril":   "DIAD",      # penghambat ACE
    "Atenolol":     "DIAD",      # beta-blocker
    "Doxycycline":  "DIAD",      # antibiotik tetrasiklin
    "Ethosuximide": "DIAD",      # antiepilepsi
    "Propafenone":  "DIAD",      # antiaritmia
    "Fluphenazine": "DIAD",      # antipsikotik
    "Griseofulvin": "DIAD",      # antijamur
    "Indapamide":   "DIAD",      # diuretik
    "Labetalol":    "DIAD",      # beta-blocker
    "Naproxen":     "DIAD",      # NSAID

    # --- 10 senyawa Non-DIAD ---
    "Loperamide":   "Non-DIAD",  # antidiare
    "Losartan":     "Non-DIAD",  # penghambat reseptor angiotensin
    "Pyridoxine":   "Non-DIAD",  # vitamin B6
    "Amoxicillin":  "Non-DIAD",  # antibiotik beta-laktam
    "Warfarin":     "Non-DIAD",  # antikoagulan
    "Ondansetron":  "Non-DIAD",  # antiemetik
    "Clopidogrel":  "Non-DIAD",  # antiplatelet
    "Domperidone":  "Non-DIAD",  # prokinetik
    "Ranitidine":   "Non-DIAD",  # penghambat H2
    "Pioglitazone": "Non-DIAD",  # antidiabetes
}

# Daftar nama obat diturunkan langsung dari kunci LABEL_AWAL.
DAFTAR_OBAT = list(LABEL_AWAL.keys())


# ═════════════════════════════════════════════════════════════════════════════
#  ALAMAT API PUBCHEM (PUG-REST)
# ═════════════════════════════════════════════════════════════════════════════
#  Struktur alamatnya dibaca seperti ini:
#
#    .../compound/name/{nama}/property/SMILES,ConnectivitySMILES,.../JSON
#       └──┬───┘ └─┬┘  └─┬─┘          └────────────┬────────────┘  └─┬─┘
#          │       │     │                         │                 │
#     cari senyawa │  nama obat            properti yang           format
#                  │                       diminta                 keluaran
#            dicari berdasarkan NAMA
#
#  Properti yang diminta:
#    SMILES             → SMILES isomerik (memuat informasi stereokimia,
#                         misalnya @ dan @@ untuk pusat kiral)
#    ConnectivitySMILES → SMILES konektivitas (tanpa stereokimia); dipakai
#                         untuk mencocokkan tumpang tindih dengan dataset asli,
#                         karena dua senyawa yang sama bisa ditulis berbeda
#                         bila stereokimianya disertakan
#    MolecularFormula   → rumus molekul, untuk pemeriksaan silang manual
#    IUPACName          → nama sistematis, untuk memastikan obat yang terambil
#                         benar-benar yang dimaksud (nama dagang kadang ambigu)
#
# ─────────────────────────────────────────────────────────────────────────────
PUG_REST_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{nama}/"
    "property/SMILES,ConnectivitySMILES,MolecularFormula,IUPACName/JSON"
)


def ambil_data_pubchem(nama_obat: str, jeda: float = 0.3) -> dict:
    """
    Mengambil SMILES, rumus molekul, dan nama IUPAC untuk SATU obat.

    Parameter
    ---------
    nama_obat : nama obat yang dicari, misalnya "Amoxicillin"
    jeda      : waktu tunggu (detik) setelah permintaan dikirim.
                PubChem membatasi maksimal 5 permintaan per detik. Jeda 0,3
                detik menjaga laju di sekitar 3 permintaan/detik — aman.
                Tanpa jeda, alamat IP dapat diblokir sementara.

    Mengembalikan
    -------------
    dict berisi hasil, atau dict berisi status gagal bila pengambilan
    tidak berhasil. Skrip tidak pernah berhenti di tengah jalan; kegagalan
    dicatat pada kolom Status agar dapat ditelusuri belakangan.
    """
    # requests.utils.quote() mengubah karakter khusus menjadi bentuk aman-URL.
    # Contoh: spasi menjadi %20. Tanpa ini, nama obat bersuku kata banyak
    # akan merusak alamat permintaan.
    url = PUG_REST_URL.format(nama=requests.utils.quote(nama_obat))

    try:
        # timeout=10 → bila server tidak menjawab dalam 10 detik, permintaan
        # dibatalkan. Tanpa timeout, skrip bisa menggantung selamanya.
        resp = requests.get(url, timeout=10)
        time.sleep(jeda)  # jaga jarak antarpermintaan (lihat penjelasan di atas)

        # Kode 200 berarti permintaan berhasil. 
        # Kode lain (404 = tidak ditemukan,
        # 503 = server sibuk) berarti gagal.
        if resp.status_code == 200:
            # Jawaban PubChem berbentuk JSON bersarang:
            #   { "PropertyTable": { "Properties": [ { "CID": 33613, ... } ] } }
            # Baris di bawah menembus tiga lapis untuk mengambil kamus properti.
            # Indeks [0] dipakai karena PubChem mengembalikan daftar; untuk
            # pencarian satu nama, isinya hanya satu unsur.
            prop = resp.json()["PropertyTable"]["Properties"][0]

            return {
                "Obat":                nama_obat,
                "Label_Awal":          LABEL_AWAL.get(nama_obat, ""),
                "CID":                 prop.get("CID", ""),                 # nomor unik permanen PubChem
                "SMILES_Isomerik":     prop.get("SMILES", ""),              # dengan stereokimia
                "SMILES_Konektivitas": prop.get("ConnectivitySMILES", ""),  # tanpa stereokimia
                "Formula_Molekul":     prop.get("MolecularFormula", ""),
                "Nama_IUPAC":          prop.get("IUPACName", ""),
                "Status":              "DITEMUKAN",
            }

        # Senyawa tidak ditemukan di PubChem (biasanya karena salah eja nama).
        # Baris tetap dikembalikan dengan kolom kosong agar tidak hilang dari
        # tabel akhir — kegagalan harus terlihat, bukan tersembunyi.
        return {
            "Obat": nama_obat, "Label_Awal": LABEL_AWAL.get(nama_obat, ""),
            "CID": "", "SMILES_Isomerik": "", "SMILES_Konektivitas": "",
            "Formula_Molekul": "", "Nama_IUPAC": "",
            "Status": f"TIDAK DITEMUKAN (HTTP {resp.status_code})",
        }

    # Menangkap kegagalan jaringan: internet putus, DNS gagal, 
    # server tidak merespons dalam batas timeout. 
    # Ditangkap agar satu kegagalan tidak membatalkan pengambilan 19 senyawa lainnya.
    except requests.exceptions.RequestException as e:
        return {
            "Obat": nama_obat, "Label_Awal": LABEL_AWAL.get(nama_obat, ""),
            "CID": "", "SMILES_Isomerik": "", "SMILES_Konektivitas": "",
            "Formula_Molekul": "", "Nama_IUPAC": "",
            "Status": f"GAGAL KONEKSI ({e})",
        }


def main():
    """Menjalankan pengambilan untuk seluruh senyawa, lalu menyimpannya ke CSV."""
    print(f"Mengambil data untuk {len(DAFTAR_OBAT)} senyawa dari PubChem PUG-REST...\n")

    hasil = []
    # enumerate(..., start=1) membuat penomoran dimulai dari 1, bukan 0,
    # sehingga tampilan progres terbaca wajar bagi manusia.
    for i, obat in enumerate(DAFTAR_OBAT, start=1):
        print(f"  [{i:2d}/{len(DAFTAR_OBAT)}] {obat} ...", end=" ")
        data = ambil_data_pubchem(obat)   # satu permintaan HTTP per senyawa
        print(data["Status"])             # tampilkan hasilnya seketika
        hasil.append(data)

    # Ubah daftar dict menjadi tabel, lalu simpan ke CSV.
    # File inilah yang menjadi masukan bagi Skrip 2 (perhitungan deskriptor RDKit).
    df = pd.DataFrame(hasil)
    output_path = "daftar_smiles_validasi_eksternal.csv"
    df.to_csv(output_path, index=False)   # index=False agar nomor baris tidak ikut tersimpan

    # Laporan akhir. Setiap kegagalan harus dilaporkan secara eksplisit —
    # senyawa yang gagal diambil tidak boleh diam-diam hilang dari analisis.
    n_gagal = (df["Status"] != "DITEMUKAN").sum()
    print(f"\nSelesai. Hasil disimpan ke: {output_path}")
    print(f"Berhasil ditemukan : {len(df) - n_gagal}/{len(df)}")
    if n_gagal > 0:
        print(f"Perlu dicek manual : {n_gagal} baris (lihat kolom 'Status')")
        print(df.loc[df["Status"] != "DITEMUKAN", "Obat"].to_string(index=False))

    return df


# Blok ini hanya berjalan bila berkas dieksekusi langsung (python ambil_smiles.py),
# dan tidak berjalan bila berkas ini diimpor sebagai modul dari berkas lain.
if __name__ == "__main__":
    main()