import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Estimasi Risiko Banjir DKI Jakarta", page_icon="🌧️", layout="centered")

@st.cache_resource
def load_artifact():
    with open("model_risiko_banjir.pkl", "rb") as f:
        return pickle.load(f)

art = load_artifact()
model = art["pipeline_risiko"]

st.title("🌧️ Estimasi Tingkat Risiko Banjir DKI Jakarta")
st.caption("Model klasifikasi risiko saat informasi tinggi genangan telah tersedia.")

kecamatan = st.selectbox("Kecamatan", art["daftar_kecamatan"])
wilayah_default = art["wilayah_per_kecamatan"].get(kecamatan, art["daftar_wilayah"][0])
st.text_input("Wilayah", value=wilayah_default, disabled=True)

bulan = st.selectbox("Bulan", list(range(1, 13)), index=0)
riwayat_default = float(art["riwayat_kecamatan_terakhir"].get(kecamatan, 0))
riwayat = st.number_input("Riwayat catatan banjir kecamatan", min_value=0.0, value=riwayat_default, step=1.0)
tinggi_air = st.number_input("Rata-rata tinggi genangan (cm)", min_value=0.0, value=30.0, step=1.0)

if st.button("Estimasi Risiko", type="primary"):
    input_df = pd.DataFrame([{
        "wilayah": wilayah_default,
        "kecamatan": kecamatan,
        "bulan": bulan,
        "riwayat_kecamatan": riwayat,
        "tinggi_air_rata_cm": tinggi_air,
    }])
    pred = model.predict(input_df)[0]
    st.success(f"Hasil estimasi: **{pred}**")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        kelas = model.named_steps["model"].classes_
        prob_df = pd.DataFrame({"Kelas": kelas, "Probabilitas": proba}).sort_values("Probabilitas", ascending=False)
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

st.divider()
st.info("Catatan: indeks risiko bersifat data-driven dan bukan klasifikasi resmi BPBD/BNPB. Model bukan forecasting banjir jangka jauh karena tinggi genangan digunakan sebagai input.")
