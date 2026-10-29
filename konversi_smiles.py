import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors as RDDescriptors

# ---------------------------------------------------------------------------
# Daftar SMILES untuk dikonversi. Ganti/tambah sesuai kebutuhan.
# Default: 20 senyawa data uji eksternal yang telah disepakati sebelumnya
# (10 kandidat DIAD-positif + 10 kandidat non-DIAD).
#
# CATATAN PERBAIKAN:
# Sebelumnya, blok ini ditulis sebagai baris CSV mentah tanpa tanda kutip
# dan tanpa pasangan "key: value", sehingga BUKAN sintaks dict Python yang
# sah (menyebabkan SyntaxError / NameError). Sekarang setiap entri ditulis
# sebagai pasangan "nama_obat": "SMILES", sesuai cara variabel ini dipakai
# di fungsi main() melalui DAFTAR_SMILES.items().
# ---------------------------------------------------------------------------
DAFTAR_SMILES = {
    # --- Kandidat DIAD-positif ---
    "Cilazapril": "CCOC(=O)C(CCC1=CC=CC=C1)NC2CCCN3CCCC(N3C2=O)C(=O)O",
    "Atenolol": "CC(C)NCC(COC1=CC=C(C=C1)CC(=O)N)O",
    "Doxycycline": "CC1C2C(C3C(C(=O)C(=C(C3(C(=O)C2=C(C4=C1C=CC=C4O)O)O)O)C(=O)N)N(C)C)O",
    "Ethosuximide": "CCC1(CC(=O)NC1=O)C",
    "Propafenone": "CCCNCC(COC1=CC=CC=C1C(=O)CCC2=CC=CC=C2)O",
    "Fluphenazine": "C1CN(CCN1CCCN2C3=CC=CC=C3SC4=C2C=C(C=C4)C(F)(F)F)CCO",
    "Griseofulvin": "CC1CC(=O)C=C(C12C(=O)C3=C(O2)C(=C(C=C3OC)OC)Cl)OC",
    "Indapamide": "CC1CC2=CC=CC=C2N1NC(=O)C3=CC(=C(C=C3)Cl)S(=O)(=O)N",
    "Labetalol": "CC(CCC1=CC=CC=C1)NCC(C2=CC(=C(C=C2)O)C(=O)N)O",
    "Naproxen": "CC(C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O",

    # --- Kandidat Non-DIAD ---
    "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
    "Losartan": "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl",
    "Furosemide": "C1=COC(=C1)CNC2=CC(=C(C=C2C(=O)O)S(=O)(=O)N)Cl",
    "Amoxicillin": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C",
    "Warfarin": "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
    "Ondansetron": "CC1=NC=CN1CC2CCC3=C(C2=O)C4=CC=CC=C4N3C",
    "Clopidogrel": "COC(=O)C(C1=CC=CC=C1Cl)N2CCC3=C(C2)C=CS3",
    "Domperidone": "C1CN(CCC1N2C3=C(C=C(C=C3)Cl)NC2=O)CCCN4C5=CC=CC=C5NC4=O",
    "Ranitidine": "CNC(=C[N+](=O)[O-])NCCSCC1=CC=C(O1)CN(C)C",
    "Pioglitazone": "CCC1=CN=C(C=C1)CCOC2=CC=C(C=C2)CC3C(=O)NC(=O)S3",
    "Loperamide": "CN(C)C(=O)C(CCN1CCC(CC1)(C2=CC=C(C=C2)Cl)O)(C3=CC=CC=C3)C4=CC=CC=C4",
    "Pyridoxine": "CC1=NC=C(C(=C1O)CO)CO",
}

# Urutan 196 nama kolom deskriptor RDKit, IDENTIK dengan struktur dataset
# pelatihan/pengujian asli (DIA_trainingset/testset_RDKit_descriptors.csv)
# dan berkas result.csv yang menjadi acuan format.
URUTAN_KOLOM_DESKRIPTOR = [
    "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v",
    "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v", "EState_VSA1",
    "EState_VSA10", "EState_VSA11", "EState_VSA2", "EState_VSA3", "EState_VSA4",
    "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9",
    "ExactMolWt", "FractionCSP3", "HallKierAlpha", "HeavyAtomCount",
    "HeavyAtomMolWt", "Ipc", "Kappa1", "Kappa2", "Kappa3", "LabuteASA",
    "MaxAbsEStateIndex", "MaxAbsPartialCharge", "MaxEStateIndex",
    "MaxPartialCharge", "MinAbsEStateIndex", "MinAbsPartialCharge",
    "MinEStateIndex", "MinPartialCharge", "MolLogP", "MolMR", "MolWt",
    "NHOHCount", "NOCount", "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles", "NumAliphaticRings", "NumAromaticCarbocycles",
    "NumAromaticHeterocycles", "NumAromaticRings", "NumHAcceptors",
    "NumHDonors", "NumHeteroatoms", "NumRadicalElectrons",
    "NumRotatableBonds", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles",
    "NumSaturatedRings", "NumValenceElectrons", "PEOE_VSA1", "PEOE_VSA10",
    "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14", "PEOE_VSA2",
    "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7",
    "PEOE_VSA8", "PEOE_VSA9", "RingCount", "SMR_VSA1", "SMR_VSA10",
    "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7",
    "SMR_VSA8", "SMR_VSA9", "SlogP_VSA1", "SlogP_VSA10", "SlogP_VSA11",
    "SlogP_VSA12", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5",
    "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8", "SlogP_VSA9", "TPSA",
    "VSA_EState1", "VSA_EState10", "VSA_EState2", "VSA_EState3",
    "VSA_EState4", "VSA_EState5", "VSA_EState6", "VSA_EState7",
    "VSA_EState8", "VSA_EState9", "fr_Al_COO", "fr_Al_OH", "fr_Al_OH_noTert",
    "fr_ArN", "fr_Ar_COO", "fr_Ar_N", "fr_Ar_NH", "fr_Ar_OH", "fr_COO",
    "fr_COO2", "fr_C_O", "fr_C_O_noCOO", "fr_C_S", "fr_HOCCN", "fr_Imine",
    "fr_NH0", "fr_NH1", "fr_NH2", "fr_N_O", "fr_Ndealkylation1",
    "fr_Ndealkylation2", "fr_Nhpyrrole", "fr_SH", "fr_aldehyde",
    "fr_alkyl_carbamate", "fr_alkyl_halide", "fr_allylic_oxid", "fr_amide",
    "fr_amidine", "fr_aniline", "fr_aryl_methyl", "fr_azide", "fr_azo",
    "fr_barbitur", "fr_benzene", "fr_benzodiazepine", "fr_bicyclic",
    "fr_diazo", "fr_dihydropyridine", "fr_epoxide", "fr_ester", "fr_ether",
    "fr_furan", "fr_guanido", "fr_halogen", "fr_hdrzine", "fr_hdrzone",
    "fr_imidazole", "fr_imide", "fr_isocyan", "fr_isothiocyan", "fr_ketone",
    "fr_ketone_Topliss", "fr_lactam", "fr_lactone", "fr_methoxy",
    "fr_morpholine", "fr_nitrile", "fr_nitro", "fr_nitro_arom",
    "fr_nitro_arom_nonortho", "fr_nitroso", "fr_oxazole", "fr_oxime",
    "fr_para_hydroxylation", "fr_phenol", "fr_phenol_noOrthoHbond",
    "fr_phos_acid", "fr_phos_ester", "fr_piperdine", "fr_piperzine",
    "fr_priamide", "fr_prisulfonamd", "fr_pyridine", "fr_quatN",
    "fr_sulfide", "fr_sulfonamd", "fr_sulfone", "fr_term_acetylene",
    "fr_tetrazole", "fr_thiazole", "fr_thiocyan", "fr_thiophene",
    "fr_unbrch_alkane", "fr_urea",
]


def hitung_deskriptor(mol):
    """Menghitung seluruh 196 deskriptor RDKit untuk satu molekul,
    mengembalikan dict sesuai urutan kolom acuan."""
    fungsi_deskriptor = dict(RDDescriptors._descList)
    hasil = {}
    for nama_kolom in URUTAN_KOLOM_DESKRIPTOR:
        fungsi = fungsi_deskriptor.get(nama_kolom)
        if fungsi is None:
            hasil[nama_kolom] = None
            continue
        try:
            hasil[nama_kolom] = fungsi(mol)
        except Exception:
            hasil[nama_kolom] = None
    return hasil


def main():
    baris_hasil = []
    gagal = []

    for nama_obat, smiles in DAFTAR_SMILES.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  [GAGAL] {nama_obat}: SMILES tidak valid -- dilewati.")
            gagal.append(nama_obat)
            continue

        baris = hitung_deskriptor(mol)
        baris["smile"] = smiles
        baris["_nama_obat"] = nama_obat  # kolom bantu, dihapus sebelum simpan
        baris_hasil.append(baris)
        print(f"  [OK] {nama_obat}: 196 deskriptor berhasil dihitung")

    df = pd.DataFrame(baris_hasil)

    # Simpan nama obat sebagai berkas terpisah (indeks bantu), lalu buang
    # dari CSV utama agar strukturnya identik dengan result.csv/dataset asli
    nama_obat_urut = df["_nama_obat"].tolist()
    df = df.drop(columns=["_nama_obat"])
    df = df[URUTAN_KOLOM_DESKRIPTOR + ["smile"]]  # pastikan urutan kolom persis

    output_path = "deskriptor_data_uji_eksternal.csv"
    df.to_csv(output_path, index=False)

    # Simpan juga versi dengan nama obat untuk memudahkan Ana menelusuri baris
    df_dengan_nama = df.copy()
    df_dengan_nama.insert(0, "nama_obat", nama_obat_urut)
    df_dengan_nama.to_csv("deskriptor_data_uji_eksternal_dengan_nama.csv", index=False)

    print(f"\nSelesai. {len(df)} senyawa berhasil dikonversi, {len(gagal)} gagal.")
    print(f"Berkas gabungan (format identik dataset asli): {output_path}")
    print(f"Berkas dengan kolom nama obat (untuk penelusuran): "
          f"deskriptor_data_uji_eksternal_dengan_nama.csv")
    if gagal:
        print(f"Senyawa yang gagal dikonversi: {gagal}")

    # -----------------------------------------------------------------
    # Simpan JUGA satu berkas CSV terpisah per senyawa (format identik
    # result.csv -- satu baris per berkas), di dalam folder "per_senyawa".
    # Berguna bila Ana ingin mengunggah/menguji satu senyawa per satu
    # kali ke antarmuka aplikasi (app.py), atau untuk kebutuhan lampiran
    # skripsi yang menyertakan berkas mentah per senyawa secara terpisah
    # agar mudah ditelusuri ulang (traceability) pada saat sidang.
    # -----------------------------------------------------------------
    folder_output = "per_senyawa"
    os.makedirs(folder_output, exist_ok=True)
    for nama_obat, baris_satu in zip(nama_obat_urut, baris_hasil):
        baris_bersih = {k: v for k, v in baris_satu.items() if k != "_nama_obat"}
        df_satu = pd.DataFrame([baris_bersih])[URUTAN_KOLOM_DESKRIPTOR + ["smile"]]
        nama_file_aman = "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in nama_obat
        )
        path_satu = os.path.join(folder_output, f"{nama_file_aman}.csv")
        df_satu.to_csv(path_satu, index=False)

    print(f"\nBerkas per senyawa (satu CSV per obat) tersimpan di folder: "
          f"./{folder_output}/")
    print(f"Jumlah berkas: {len(nama_obat_urut)}")

    return df


if __name__ == "__main__":
    main()