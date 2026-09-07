import streamlit as st
import pandas as pd
import pickle
import sys
import sklearn
import numpy as np

st.set_page_config(
    page_title="Estimasi Risiko Banjir DKI Jakarta",
    page_icon="🌧️",
    layout="centered"
)

@st.cache_resource
def load_artifact():
    with open("model_risiko_banjir.pkl", "rb") as f:
        return pickle.load(f)

try:
    art = load_artifact()
except Exception as e:
    st.error("Model gagal dimuat.")
    st.exception(e)
    st.stop()

model = art["pipeline_risiko"]

st.title("🌧️ Estimasi Tingkat Risiko Banjir DKI Jakarta")

st.caption(
    "Model klasifikasi risiko saat informasi tinggi genangan "
    "telah tersedia."
)

# =========================================================
# INPUT
# =========================================================

kecamatan = st.selectbox(
    "Kecamatan",
    art["daftar_kecamatan"]
)

wilayah_default = art["wilayah_per_kecamatan"].get(
    kecamatan,
    art["daftar_wilayah"][0]
)

st.text_input(
    "Wilayah",
    value=wilayah_default,
    disabled=True
)

bulan = st.selectbox(
    "Bulan",
    list(range(1, 13)),
    index=0
)

riwayat_default = float(
    art["riwayat_kecamatan_terakhir"].get(kecamatan, 0)
)

riwayat = st.number_input(
    "Riwayat catatan banjir kecamatan",
    min_value=0.0,
    value=riwayat_default,
    step=1.0
)

tinggi_air = st.number_input(
    "Rata-rata tinggi genangan (cm)",
    min_value=0.0,
    value=30.0,
    step=1.0
)

# =========================================================
# PREDIKSI
# =========================================================

if st.button("Estimasi Risiko", type="primary"):

    input_df = pd.DataFrame([{
        "wilayah": wilayah_default,
        "kecamatan": kecamatan,
        "bulan": int(bulan),
        "riwayat_kecamatan": float(riwayat),
        "tinggi_air_rata_cm": float(tinggi_air),
    }])

    try:
        pred = model.predict(input_df)[0]

        st.success(
            f"Hasil estimasi tingkat risiko: **{pred}**"
        )

        if hasattr(model, "predict_proba"):

            proba = model.predict_proba(input_df)[0]

            kelas = model.named_steps["model"].classes_

            prob_df = pd.DataFrame({
                "Kelas": kelas,
                "Probabilitas": proba
            })

            prob_df["Probabilitas"] = (
                prob_df["Probabilitas"] * 100
            ).round(2)

            prob_df = prob_df.sort_values(
                "Probabilitas",
                ascending=False
            )

            prob_df["Probabilitas"] = (
                prob_df["Probabilitas"]
                .astype(str) + "%"
            )

            st.subheader("Probabilitas Prediksi")

            st.dataframe(
                prob_df,
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:
        st.error("Terjadi kesalahan saat melakukan prediksi.")
        st.exception(e)

# =========================================================
# INFORMASI MODEL
# =========================================================

st.divider()

st.info(
    "Catatan: indeks risiko bersifat data-driven dan bukan "
    "klasifikasi resmi BPBD/BNPB. Model ini bukan forecasting "
    "banjir jangka jauh karena tinggi genangan digunakan "
    "sebagai salah satu input model."
)

with st.expander("Informasi teknis model"):

    if "environment_info" in art:

        env = art["environment_info"]

        st.write(
            "Model dibuat menggunakan:"
        )

        st.write(
            f"- scikit-learn: "
            f"{env.get('scikit_learn_version', '-')}"
        )

        st.write(
            f"- pandas: "
            f"{env.get('pandas_version', '-')}"
        )

        st.write(
            f"- numpy: "
            f"{env.get('numpy_version', '-')}"
        )

    st.write(
        f"Environment deployment:"
    )

    st.write(
        f"- Python: {sys.version.split()[0]}"
    )

    st.write(
        f"- scikit-learn: {sklearn.__version__}"
    )

    st.write(
        f"- pandas: {pd.__version__}"
    )

    st.write(
        f"- numpy: {np.__version__}"
    )
