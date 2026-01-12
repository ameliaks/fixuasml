import streamlit as st
import pandas as pd
import numpy as np
import joblib


def prediction_app():
    st.title("Prediksi Status Gizi Anak")
    st.write("Masukkan data anak untuk memprediksi **Nutrition_Status** menggunakan model Logistic Regression.")

    # =========================
    # 1) Load model + artifacts (HARUS sama dengan training)
    # =========================
    model = joblib.load("model_malnutrition.pkl")
    feature_names = joblib.load("model_features_malnutrition.pkl")
    le = joblib.load("label_encoder_malnutrition.pkl")
    scaler = joblib.load("scaler_malnutrition.pkl")
    cont_cols = joblib.load("continuous_columns_malnutrition.pkl")

    # =========================
    # 2) Ambil daftar kolom asli dari dataset (biar form otomatis sesuai dataset kamu)
    # =========================
    df_raw = pd.read_csv("malnutrition_children_ethiopia.csv")

    # Drop id kalau ada
    for col_id in ["id", "ID"]:
        if col_id in df_raw.columns:
            df_raw = df_raw.drop(columns=[col_id])

    target_col = "Nutrition_Status"
    if target_col not in df_raw.columns:
        st.error("Kolom target Nutrition_Status tidak ditemukan di dataset.")
        st.stop()

    feature_cols_raw = [c for c in df_raw.columns if c != target_col]

    st.write("### Input Data Anak")
    st.caption("Form di bawah otomatis mengikuti kolom dataset kamu (selain target).")

    # =========================
    # 3) Buat input user secara dinamis sesuai tipe kolom
    # =========================
    user_data = {}

    cols_ui = st.columns(3)
    col_index = 0

    for col in feature_cols_raw:
        with cols_ui[col_index]:
            if pd.api.types.is_numeric_dtype(df_raw[col]):
                default_val = float(df_raw[col].median()) if df_raw[col].notna().any() else 0.0
                min_val = float(df_raw[col].min()) if df_raw[col].notna().any() else 0.0
                max_val = float(df_raw[col].max()) if df_raw[col].notna().any() else 100.0

                if pd.api.types.is_integer_dtype(df_raw[col]):
                    user_data[col] = st.number_input(
                        col, min_value=int(min_val), max_value=int(max_val), value=int(default_val)
                    )
                else:
                    user_data[col] = st.number_input(
                        col, min_value=min_val, max_value=max_val, value=default_val
                    )
            else:
                opts = df_raw[col].dropna().astype(str).unique().tolist()
                opts = sorted(opts)
                if len(opts) == 0:
                    opts = ["Unknown"]
                user_data[col] = st.selectbox(col, opts)

        col_index = (col_index + 1) % 3

    # =========================
    # 4) Preprocess input user -> harus sama seperti training
    # =========================
    user_df = pd.DataFrame([user_data])

    user_encoded = pd.get_dummies(user_df, drop_first=True)
    user_encoded = user_encoded.reindex(columns=feature_names, fill_value=0)

    cont_exist = [c for c in cont_cols if c in user_encoded.columns]
    if len(cont_exist) > 0:
        user_encoded[cont_exist] = scaler.transform(user_encoded[cont_exist])

    # =========================
    # 5) Prediksi + Rekomendasi
    # =========================
    if st.button("Prediksi Status Gizi"):
        pred_idx = model.predict(user_encoded)[0]
        pred_label = le.inverse_transform([pred_idx])[0]

        proba = model.predict_proba(user_encoded)[0]
        proba_df = pd.DataFrame({
            "Kelas": le.classes_,
            "Probabilitas": proba
        }).sort_values("Probabilitas", ascending=False)

        top_class = proba_df.iloc[0]["Kelas"]
        top_prob = float(proba_df.iloc[0]["Probabilitas"])

        st.write("### 🔍 Hasil Prediksi")
        st.success(f"Prediksi Status Gizi: **{pred_label}**")

        st.write("### Probabilitas Tiap Kelas")
        st.dataframe(proba_df, use_container_width=True)

        st.metric("Probabilitas Prediksi Tertinggi", f"{top_prob*100:.2f}%")

        # Optional: indikator keyakinan model
        if top_prob < 0.50:
            st.warning(
                "⚠️ Model kurang yakin (probabilitas tertinggi < 50%). "
                "Sebaiknya cek kembali data input atau lakukan pemeriksaan lanjutan."
            )
        elif top_prob < 0.70:
            st.info(
                "ℹ️ Keyakinan model sedang (50–70%). "
                "Hasil bisa digunakan sebagai skrining awal, tetap perlu evaluasi tenaga kesehatan bila perlu."
            )
        else:
            st.success("✅ Keyakinan model tinggi (≥ 70%).")

        st.write("---")
        st.write("### 📌 Interpretasi Singkat")
        st.write(
            "- **Normal**: kondisi gizi sesuai standar.\n"
            "- **At_Risk**: mulai mengarah ke risiko gizi kurang.\n"
            "- **Malnourished**: gizi kurang/buruk dan perlu perhatian."
        )

        st.write("---")
        st.write("### 🧾 Rekomendasi Tindak Lanjut")

        # Rekomendasi berdasarkan kelas
        recommendations = {
            "Normal": [
                "Pertahankan pola makan seimbang (karbohidrat, protein, lemak sehat, sayur & buah).",
                "Pantau pertumbuhan rutin (berat & tinggi) sesuai jadwal posyandu/puskesmas.",
                "Pastikan imunisasi dan kebersihan lingkungan (air bersih & sanitasi) tetap terjaga."
            ],
            "At_Risk": [
                "Lakukan evaluasi asupan makan: tambah sumber protein (telur, ikan, tempe), dan energi sehat.",
                "Pantau pertumbuhan lebih sering (misalnya 2–4 minggu sekali) untuk melihat tren kenaikan.",
                "Periksa faktor pemicu (diare/infeksi berulang) dan konsultasi ke puskesmas/posyandu bila perlu."
            ],
            "Malnourished": [
                "Segera konsultasi ke tenaga kesehatan (puskesmas/RS) untuk penilaian gizi lebih lanjut.",
                "Perlu intervensi gizi terarah (misalnya PMT/terapi gizi) sesuai rekomendasi tenaga kesehatan.",
                "Cek kemungkinan penyebab: infeksi, diare kronis, anemia, kondisi lingkungan/sanitasi, dan lakukan tindak lanjut."
            ]
        }

        # Antisipasi jika label berbeda format (misal At_Risk/At Risk)
        label_key = str(pred_label).strip()
        if label_key not in recommendations:
            # fallback sederhana
            st.write("Rekomendasi umum: pantau pertumbuhan, perbaiki pola makan, dan konsultasi tenaga kesehatan bila diperlukan.")
        else:
            if label_key == "Malnourished":
                st.error("⚠️ Kategori ini butuh perhatian segera.")
            elif label_key == "At_Risk":
                st.warning("⚠️ Anak berada dalam kondisi berisiko, perlu pencegahan lebih dini.")
            else:
                st.success("✅ Anak berada pada kondisi normal, tetap perlu pemantauan rutin.")

            for i, rec in enumerate(recommendations[label_key], start=1):
                st.write(f"{i}. {rec}")

        st.write("---")
        st.caption(
            "Catatan: hasil ini bersifat *screening* berbasis model machine learning. "
            "Keputusan medis tetap mengacu pada pemeriksaan tenaga kesehatan."
        )


if __name__ == "__main__":
    prediction_app()
