import pickle

import pandas as pd
import streamlit as st

# ============================================================
# Konfigurasi halaman
# ============================================================
st.set_page_config(
    page_title="Prediksi Tingkat Risiko Banjir DKI Jakarta",
    page_icon="🌊",
    layout="centered",
)

WARNA_RISIKO = {
    "Risiko Sangat Rendah": "#2e7d32",
    "Risiko Rendah": "#66bb6a",
    "Risiko Sedang": "#fbc02d",
    "Risiko Tinggi": "#fb8c00",
    "Risiko Sangat Tinggi": "#c62828",
}


# ============================================================
# Load artefak model (di-cache supaya tidak reload tiap interaksi)
# ============================================================
@st.cache_resource
def load_artefak(path: str = "model_risiko_banjir.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


artefak = load_artefak()

model_risiko = artefak["model_risiko"]
kolom_fitur = artefak["kolom_fitur_risiko"]
fitur_kategorikal = artefak["fitur_kategorikal_risiko"]
fitur_numerik = artefak["fitur_numerik_risiko"]
daftar_wilayah = artefak["daftar_wilayah"]
daftar_kecamatan = artefak["daftar_kecamatan"]
wilayah_per_kecamatan = artefak["wilayah_per_kecamatan"]
riwayat_kecamatan_terakhir = artefak["riwayat_kecamatan_terakhir"]
urutan_label = artefak["urutan_label_risiko"]


# ============================================================
# Fungsi bantu: susun fitur input jadi format yang sama persis
# dengan waktu training (one-hot encoding + urutan kolom identik)
# ============================================================
def buat_fitur_input(kecamatan: str, bulan: int, riwayat_kecamatan: float, tinggi_air: float) -> pd.DataFrame:
    wilayah = wilayah_per_kecamatan.get(kecamatan, daftar_wilayah[0])

    baris = {
        "bulan": bulan,
        "riwayat_kecamatan": riwayat_kecamatan,
        "tinggi_air_rata_cm": tinggi_air,
    }
    for w in daftar_wilayah:
        baris[f"wilayah_{w}"] = 1 if w == wilayah else 0
    for k in daftar_kecamatan:
        baris[f"kecamatan_{k}"] = 1 if k == kecamatan else 0

    X_input = pd.DataFrame([baris])
    # Pastikan urutan & kelengkapan kolom identik dengan saat training
    X_input = X_input.reindex(columns=kolom_fitur, fill_value=0)
    return X_input


# ============================================================
# UI
# ============================================================
st.title("🌊 Prediksi Tingkat Risiko Banjir")
st.caption(
    "Model Random Forest yang memprediksi tingkat risiko dampak banjir "
    "(bukan sekadar cluster) berdasarkan lokasi, waktu, riwayat kejadian, dan tinggi air."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    kecamatan = st.selectbox("Kecamatan", options=daftar_kecamatan)
    wilayah_terdeteksi = wilayah_per_kecamatan.get(kecamatan, "-")
    st.text_input("Wilayah (otomatis)", value=wilayah_terdeteksi, disabled=True)

with col2:
    bulan = st.selectbox(
        "Bulan",
        options=list(range(1, 13)),
        format_func=lambda b: [
            "Januari", "Februari", "Maret", "April", "Mei", "Juni",
            "Juli", "Agustus", "September", "Oktober", "November", "Desember",
        ][b - 1],
    )
    tinggi_air = st.number_input("Perkiraan tinggi genangan air (cm)", min_value=0.0, max_value=300.0, value=30.0, step=5.0)

default_riwayat = float(riwayat_kecamatan_terakhir.get(kecamatan, 0))
riwayat_kecamatan = st.number_input(
    "Riwayat jumlah kejadian banjir di kecamatan ini (tahun-tahun sebelumnya)",
    min_value=0.0,
    value=default_riwayat,
    step=1.0,
    help="Terisi otomatis dari data historis kecamatan terpilih. Bisa diubah manual jika ada data lebih baru.",
)

st.divider()

if st.button("Prediksi Tingkat Risiko", type="primary", use_container_width=True):
    X_input = buat_fitur_input(kecamatan, bulan, riwayat_kecamatan, tinggi_air)
    prediksi = model_risiko.predict(X_input)[0]
    proba = model_risiko.predict_proba(X_input)[0]
    proba_df = pd.DataFrame({
        "Tingkat Risiko": model_risiko.classes_,
        "Probabilitas": proba,
    }).set_index("Tingkat Risiko").reindex(urutan_label).dropna()

    warna = WARNA_RISIKO.get(prediksi, "#616161")
    st.markdown(
        f"""
        <div style="padding:1rem;border-radius:0.5rem;background-color:{warna}22;
                    border:1px solid {warna};text-align:center;">
            <span style="font-size:0.9rem;">Prediksi Tingkat Risiko</span><br>
            <span style="font-size:1.6rem;font-weight:700;color:{warna};">{prediksi}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("Distribusi Probabilitas per Kelas")
    st.bar_chart(proba_df)

    st.caption(
        "Catatan: model ini paling andal membedakan risiko rendah vs risiko tinggi secara umum. "
        "Untuk kelas risiko tinggi/sangat tinggi, akurasi model lebih rendah -- gunakan hasil ini "
        "sebagai alat skrining awal, bukan satu-satunya dasar keputusan mitigasi."
    )

st.divider()
with st.expander("Tentang model ini"):
    st.write(
        """
        Model ini dilatih dari data kejadian banjir DKI Jakarta 2015-2020, dengan target
        **tingkat risiko** (kombinasi tinggi genangan air dan jumlah jiwa terdampak, disederhanakan
        ke 5 kategori ordinal). Fitur yang dipakai untuk prediksi: wilayah, kecamatan, bulan,
        riwayat kejadian di kecamatan tersebut pada tahun-tahun sebelumnya, dan tinggi air --
        semuanya informasi yang secara wajar tersedia lebih awal, bukan hasil dampak itu sendiri.

        Algoritma: Random Forest Classifier, dievaluasi dengan pembagian data berbasis waktu
        (latih 2015-2018, uji 2019-2020).
        """
    )
