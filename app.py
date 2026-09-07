import pickle
import numpy as np
import streamlit as st

with open("model_banjir.pkl", "rb") as f:
    artefak = pickle.load(f)

scaler = artefak["scaler"]
kmeans = artefak["kmeans"]
fitur = artefak["fitur"]
kolom_log = artefak["kolom_log"]
nama_cluster = artefak["nama_cluster"]
mapping_risiko = artefak["mapping_risiko"]
batas_air = artefak["batas_air"]
batas_jiwa = artefak["batas_jiwa"]


def hitung_level(nilai, batas):
    if nilai <= batas["level1_max"]:
        return 1
    elif nilai <= batas["level2_max"]:
        return 2
    else:
        return 3


def prediksi_banjir(tinggi_air, rw_terdampak, kk_terdampak, jiwa_terdampak, pengungsi):
    nilai = {
        "tinggi_air_rata_cm": tinggi_air,
        "jumlah_rw_terdampak": rw_terdampak,
        "jumlah_kk_terdampak": kk_terdampak,
        "jumlah_jiwa_terdampak": jiwa_terdampak,
        "jumlah_pengungsi": pengungsi,
    }
    X_baru = np.array([[nilai[f] for f in fitur]])
    for i, f in enumerate(fitur):
        if f in kolom_log:
            X_baru[0, i] = np.log1p(X_baru[0, i])
    X_baru_scaled = scaler.transform(X_baru)
    cluster = int(kmeans.predict(X_baru_scaled)[0])
    label_cluster = nama_cluster.get(cluster, f"Cluster {cluster}")

    level_air = hitung_level(tinggi_air, batas_air)
    level_jiwa = hitung_level(jiwa_terdampak, batas_jiwa)
    skor = level_air + level_jiwa
    tingkat_risiko = mapping_risiko.get(skor, "Tidak diketahui")

    return cluster, label_cluster, tingkat_risiko, level_air, level_jiwa


st.set_page_config(page_title="Prediksi Banjir DKI Jakarta", page_icon="\U0001F30A")
st.title("Prediksi Pola & Tingkat Risiko Banjir DKI Jakarta (2015-2020)")
st.write(
    "Masukkan estimasi data satu kejadian banjir untuk melihat kelompok pola "
    "(cluster) dan tingkat risikonya, berdasarkan model yang dilatih dari data "
    "kejadian banjir DKI Jakarta 2015-2020."
)

col1, col2 = st.columns(2)
with col1:
    tinggi_air = st.number_input("Tinggi Genangan Air (cm)", min_value=0.0, value=35.0)
    rw_terdampak = st.number_input("Jumlah RW Terdampak", min_value=0.0, value=2.0)
    kk_terdampak = st.number_input("Jumlah KK Terdampak", min_value=0.0, value=10.0)
with col2:
    jiwa_terdampak = st.number_input("Jumlah Jiwa Terdampak", min_value=0.0, value=40.0)
    pengungsi = st.number_input("Jumlah Pengungsi", min_value=0.0, value=0.0)

if st.button("Prediksi", type="primary"):
    cluster, label_cluster, tingkat_risiko, level_air, level_jiwa = prediksi_banjir(
        tinggi_air, rw_terdampak, kk_terdampak, jiwa_terdampak, pengungsi
    )
    st.success(f"**Kelompok Pola (K-Means):** Cluster {cluster} \u2014 {label_cluster}")
    st.info(
        f"**Tingkat Risiko:** {tingkat_risiko}  \n"
        f"(Level tinggi air: {level_air}/3, Level dampak jiwa: {level_jiwa}/3)"
    )

st.caption("Model dilatih dari data kejadian banjir DKI Jakarta 2015-2020 (SatuData Jakarta).")
