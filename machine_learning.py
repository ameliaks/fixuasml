import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# === 5 algoritma ===
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# --- SMOTE: optional (biar ga error import) ---
IMBLEARN_OK = False
IMBLEARN_ERR = ""
SMOTE = None
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_OK = True
except Exception as e:
    IMBLEARN_OK = False
    IMBLEARN_ERR = str(e)
    SMOTE = None


# =========================
# Helpers biar robust
# =========================
@st.cache_data(show_spinner=False)
def _read_csv(file) -> pd.DataFrame:
    # file bisa UploadedFile atau path string
    return pd.read_csv(file)

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # strip spasi dan pastikan string
    df.columns = [str(c).strip() for c in df.columns]

    # handle kolom duplikat: A, A -> A, A__2
    seen = {}
    new_cols = []
    for c in df.columns:
        if c not in seen:
            seen[c] = 1
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}__{seen[c]}")
    df.columns = new_cols
    return df

def _suggest_target(df: pd.DataFrame) -> str:
    # cari kandidat target umum (case-insensitive)
    candidates = [
        "nutrition_status", "nutritionstatus", "status_gizi", "statusgizi",
        "target", "label", "class", "outcome", "y"
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    # fallback: kolom terakhir
    return df.columns[-1]

def _safe_stratify(y: np.ndarray):
    # stratify hanya aman kalau tiap kelas minimal 2
    uniques, counts = np.unique(y, return_counts=True)
    if len(uniques) < 2:
        return None, "Dataset hanya punya 1 kelas. Stratify dimatikan."
    if np.any(counts < 2):
        return None, "Ada kelas dengan jumlah < 2. Stratify dimatikan agar tidak error."
    return y, None

def _ensure_numeric_X(X: pd.DataFrame) -> pd.DataFrame:
    # pastikan semua fitur numeric setelah get_dummies
    X = X.copy()
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            # fallback: coba convert
            X[c] = pd.to_numeric(X[c], errors="coerce")
    # isi NaN yang mungkin muncul dari coercion
    X = X.fillna(0)
    return X


def ml_model():
    st.title("Model Prediksi (Flexible Upload CSV)")

    # =========================
    # 0) Upload data (atau pakai default)
    # =========================
    st.write("### Upload Dataset (CSV)")
    uploaded = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded is not None:
        df = _read_csv(uploaded)
        st.success("Dataset berhasil di-upload.")
    else:
        default_path = "malnutrition_children_ethiopia.csv"
        if not os.path.exists(default_path):
            st.error("Tidak ada file default 'malnutrition_children_ethiopia.csv' dan kamu belum upload dataset.")
            st.stop()
        df = _read_csv(default_path)
        st.info("Menggunakan dataset default: malnutrition_children_ethiopia.csv")

    # bersihin kolom biar stabil
    df = _clean_columns(df)

    # Drop kolom ID umum jika ada
    for col_id in ["id", "ID", "Id", "index", "Index"]:
        if col_id in df.columns:
            df = df.drop(columns=[col_id])

    if df.shape[1] < 2:
        st.error("Dataset harus punya minimal 2 kolom: 1 target + minimal 1 fitur.")
        st.stop()

    st.write("**Preview dataset**")
    st.dataframe(df.head())

    # =========================
    # 1) Tentukan target kolom
    # =========================
    st.write("### Pilih Kolom Target")
    suggested = _suggest_target(df)

    # kalau suggested kebetulan tidak ada (harusnya ada), fallback aman
    if suggested not in df.columns:
        suggested = df.columns[-1]

    target_col = st.selectbox(
        "Target (label yang diprediksi)",
        options=df.columns.tolist(),
        index=df.columns.tolist().index(suggested)
    )

    # Drop baris yang targetnya kosong
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    if df.shape[0] < 5:
        st.warning("Data sangat sedikit setelah buang target kosong. Model bisa tidak stabil.")

    # =========================
    # 2) Pisahkan numerik vs kategorik
    # =========================
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    # Pastikan target tidak masuk fitur
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    # kolom continuous (hindari biner)
    continuous_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) > 10]

    # =========================
    # 3) Outlier removal (IQR) khusus continuous
    # =========================
    st.write("### 1. Deteksi Outlier (IQR)")
    st.write(f"Jumlah data sebelum pembersihan: **{df.shape[0]} baris**")

    if len(continuous_cols) > 0:
        Q1 = df[continuous_cols].quantile(0.25)
        Q3 = df[continuous_cols].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df[~((df[continuous_cols] < lower) | (df[continuous_cols] > upper)).any(axis=1)].copy()
    else:
        df_clean = df.copy()

    st.write(f"Jumlah data setelah pembersihan outlier: **{df_clean.shape[0]} baris**")
    if df_clean.shape[0] < 10:
        st.warning("Data setelah outlier cleaning sangat sedikit. Pertimbangkan matikan/abaikan outlier cleaning.")

    # =========================
    # 4) Encoding (one-hot fitur) + encode target
    # =========================
    df_model = df_clean.copy()

    # Isi missing value fitur (simple)
    for c in numeric_cols:
        if c in df_model.columns:
            df_model[c] = df_model[c].fillna(df_model[c].median())

    for c in cat_cols:
        if c in df_model.columns:
            mode_val = df_model[c].mode(dropna=True)
            df_model[c] = df_model[c].fillna(mode_val.iloc[0] if len(mode_val) else "Unknown")
            # amankan tipe campuran
            df_model[c] = df_model[c].astype(str)

    # target juga amanin (kalau campuran)
    df_model[target_col] = df_model[target_col].astype(str)

    # one-hot hanya fitur kategorik (bukan target)
    df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

    le = LabelEncoder()
    y = le.fit_transform(df_model[target_col])

    # IMPORTANT FIX: label st.metric harus string
    class_names = list(le.classes_)
    class_labels = [str(c) for c in class_names]

    # fitur X
    if target_col not in df_model.columns:
        st.error("Target hilang setelah preprocessing. Cek nama kolom target.")
        st.stop()

    X = df_model.drop(columns=[target_col])

    if X.shape[1] == 0:
        st.error("Tidak ada fitur setelah preprocessing (X kosong). Pastikan dataset punya kolom fitur selain target.")
        st.stop()

    X = _ensure_numeric_X(X)

    # =========================
    # 5) Normalisasi MinMax (continuous)
    # =========================
    st.write("### 2. Normalisasi menggunakan MinMax Scaler")

    scaler = MinMaxScaler()
    cont_exist = [c for c in continuous_cols if c in X.columns]

    X_before = X.copy()
    if len(cont_exist) > 0:
        try:
            X[cont_exist] = scaler.fit_transform(X[cont_exist])
        except Exception as e:
            st.warning(f"Normalisasi gagal diterapkan pada kolom continuous. Detail: {e}")

        colA, colB = st.columns(2)
        with colA:
            st.write("**Sebelum Normalisasi**")
            for col in cont_exist:
                chart = (
                    alt.Chart(pd.DataFrame({col: X_before[col]}))
                    .transform_density(col, as_=[col, "density"])
                    .mark_area(opacity=0.5)
                    .encode(x=alt.X(f"{col}:Q"), y=alt.Y("density:Q"))
                    .properties(height=200, title=f"Density: {col}")
                )
                st.altair_chart(chart, use_container_width=True)

        with colB:
            st.write("**Setelah Normalisasi**")
            for col in cont_exist:
                chart = (
                    alt.Chart(pd.DataFrame({col: X[col]}))
                    .transform_density(col, as_=[col, "density"])
                    .mark_area(opacity=0.5)
                    .encode(x=alt.X(f"{col}:Q"), y=alt.Y("density:Q"))
                    .properties(height=200, title=f"Density: {col}")
                )
                st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Tidak ada kolom numerik continuous yang cocok untuk divisualisasi normalisasi.")

    # =========================
    # 6) Korelasi (continuous)
    # =========================
    st.write("### 3. Korelasi Linear antar Kolom Numerik (Continuous)")
    if len(cont_exist) >= 2:
        corr = pd.DataFrame(X_before[cont_exist]).corr().reset_index().melt("index")
        corr.columns = ["Variable1", "Variable2", "Correlation"]

        heatmap = (
            alt.Chart(corr).mark_rect()
            .encode(
                x=alt.X("Variable2:N", title=None),
                y=alt.Y("Variable1:N", title=None),
                color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["Variable1", "Variable2", alt.Tooltip("Correlation:Q", format=".2f")]
            )
            .properties(height=400, title="Correlation Heatmap")
        )
        text = (
            alt.Chart(corr).mark_text(fontSize=12, color="black")
            .encode(x="Variable2:N", y="Variable1:N", text=alt.Text("Correlation:Q", format=".2f"))
        )
        st.altair_chart(heatmap + text, use_container_width=True)
    else:
        st.info("Korelasi butuh minimal 2 kolom continuous.")

    # =========================
    # 7) Train–Test Split
    # =========================
    st.write("### 4. Train–Test Split")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

    strat, strat_msg = _safe_stratify(y)
    if strat_msg:
        st.warning(strat_msg)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(test_size), random_state=42, stratify=strat
        )
    except Exception as e:
        st.warning(f"train_test_split dengan stratify gagal ({e}). Coba tanpa stratify.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(test_size), random_state=42, stratify=None
        )

    st.write(f"X_train: **{len(X_train)}** | X_test: **{len(X_test)}**")

    # =========================
    # 8) Handling Imbalance (SMOTE optional)
    # =========================
    st.write("### 5. Handling Imbalance Class")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Distribusi kelas (sebelum balancing)**")
        unique, counts = np.unique(y_train, return_counts=True)
        before_counts = dict(zip(unique, counts))
        for k in range(len(class_labels)):
            st.metric(label=class_labels[k], value=int(before_counts.get(k, 0)))

    # cek kelayakan SMOTE: minimal 2 kelas dan kelas minoritas >= 2
    smote_feasible = False
    if IMBLEARN_OK:
        u, c = np.unique(y_train, return_counts=True)
        if len(u) >= 2 and np.min(c) >= 2:
            smote_feasible = True

    use_smote = False
    if IMBLEARN_OK and smote_feasible:
        use_smote_choice = st.checkbox("Gunakan SMOTE (jika tersedia & feasible)", value=True)
    elif IMBLEARN_OK and (not smote_feasible):
        use_smote_choice = False
        st.warning("SMOTE tidak feasible (kelas terlalu sedikit / hanya 1 kelas). Pakai class_weight='balanced' jika ada.")
    else:
        use_smote_choice = False
        st.warning("SMOTE tidak bisa dipakai (imblearn tidak terbaca di environment Streamlit). "
                   "Model akan memakai class_weight='balanced' bila tersedia.")
        st.caption(f"Detail import: {IMBLEARN_ERR}")

    if IMBLEARN_OK and smote_feasible and use_smote_choice:
        try:
            sm = SMOTE(random_state=42)
            X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
            use_smote = True
        except Exception as e:
            X_train_bal, y_train_bal = X_train, y_train
            st.warning(f"SMOTE gagal dijalankan. Pakai balancing internal (jika ada). Detail: {e}")
            use_smote = False
    else:
        X_train_bal, y_train_bal = X_train, y_train
        use_smote = False

    with col2:
        if use_smote:
            st.write("**Distribusi kelas (setelah SMOTE)**")
            unique2, counts2 = np.unique(y_train_bal, return_counts=True)
            after_counts = dict(zip(unique2, counts2))
            for k in range(len(class_labels)):
                st.metric(label=class_labels[k], value=int(after_counts.get(k, 0)))
        else:
            st.info("Tanpa SMOTE. (Beberapa model akan pakai class_weight='balanced' bila ada).")

    # =========================
    # 9) Pilih algoritma (5 model)
    # =========================
    st.write("### 6. Pemodelan (Pilih 1 dari 5 Algoritma)")
    algo = st.selectbox(
        "Algoritma",
        [
            "Logistic Regression (Multinomial)",
            "Random Forest",
            "Gradient Boosting",
            "SVM (RBF)",
            "K-Nearest Neighbors (KNN)"
        ]
    )

    colP1, colP2, colP3 = st.columns(3)
    with colP1:
        n_estimators = st.number_input("n_estimators (RF)", 50, 1000, 200, 50)
    with colP2:
        max_depth = st.number_input("max_depth (RF, 0=none)", 0, 50, 0, 1)
    with colP3:
        k_neighbors = st.number_input("k (KNN)", 1, 50, 7, 1)

    if algo == "Logistic Regression (Multinomial)":
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=3000,
            class_weight=None if use_smote else "balanced"
        )
        model_name = "logreg_multinomial"

    elif algo == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=None if int(max_depth) == 0 else int(max_depth),
            random_state=42,
            class_weight=None if use_smote else "balanced",
            n_jobs=-1
        )
        model_name = "random_forest"

    elif algo == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
        model_name = "gradient_boosting"

    elif algo == "SVM (RBF)":
        model = SVC(
            kernel="rbf",
            probability=True,
            class_weight=None if use_smote else "balanced",
            random_state=42
        )
        model_name = "svm_rbf"

    elif algo == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier(n_neighbors=int(k_neighbors))
        model_name = "knn"

    # =========================
    # 10) Training model
    # =========================
    if st.button("Train Model"):
        try:
            model.fit(X_train_bal, y_train_bal)
        except Exception as e:
            st.error(f"Training gagal: {e}")
            st.stop()

        train_acc = model.score(X_train_bal, y_train_bal)
        st.write("Akurasi Training =", f"**{round(train_acc * 100, 2)}%**")

        # Ringkasan pengaruh fitur (jika ada)
        st.write("**Ringkasan pengaruh fitur (jika tersedia)**")
        if hasattr(model, "coef_"):
            coef_df = pd.DataFrame(model.coef_, columns=X.columns)
            coef_df["Class"] = class_labels
            st.dataframe(coef_df.set_index("Class"))

            st.write("**Top 10 fitur (berdasarkan rata-rata |koefisien|)**")
            abs_mean = np.abs(model.coef_).mean(axis=0)
            top_idx = np.argsort(abs_mean)[::-1][:10]
            st.dataframe(pd.DataFrame({"Feature": X.columns[top_idx], "Avg |Coefficient|": abs_mean[top_idx]}))

        elif hasattr(model, "feature_importances_"):
            imp = np.array(model.feature_importances_)
            top_idx = np.argsort(imp)[::-1][:10]
            st.dataframe(pd.DataFrame({"Feature": X.columns[top_idx], "Importance": imp[top_idx]}))
        else:
            st.info("Model ini tidak menyediakan koefisien/feature importance secara langsung.")

        # =========================
        # 11) Evaluasi
        # =========================
        st.write("### 7. Evaluasi Model")
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec_w = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # ROC AUC (binary & multiclass)
        auc = np.nan
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                n_classes = len(class_labels)

                if n_classes == 2 and proba.shape[1] == 2:
                    auc = roc_auc_score(y_test, proba[:, 1])
                else:
                    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
                    if y_test_bin.shape[1] == proba.shape[1]:
                        auc = roc_auc_score(y_test_bin, proba, multi_class="ovr", average="weighted")
        except Exception:
            auc = np.nan

        colM1, colM2 = st.columns([2, 2])
        with colM1:
            cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_labels)))
            cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels).reset_index().melt("index")
            cm_df.columns = ["Actual", "Predicted", "Count"]

            chart = (
                alt.Chart(cm_df).mark_rect()
                .encode(
                    x=alt.X("Predicted:N"),
                    y=alt.Y("Actual:N"),
                    color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
                    tooltip=["Actual", "Predicted", "Count"]
                )
                .properties(height=350, title="Confusion Matrix")
            )
            text = alt.Chart(cm_df).mark_text(color="black").encode(
                x="Predicted:N", y="Actual:N", text="Count:Q"
            )
            st.altair_chart(chart + text, use_container_width=True)

        with colM2:
            st.metric("Accuracy", f"{acc*100:.2f}%")
            st.metric("Precision (weighted)", f"{prec_w*100:.2f}%")
            st.metric("Recall (weighted)", f"{rec_w*100:.2f}%")
            st.metric("F1 Score (weighted)", f"{f1_w*100:.2f}%")
            st.metric("ROC AUC", "-" if np.isnan(auc) else f"{auc*100:.2f}%")

        # =========================
        # 12) Save artifacts
        # =========================
        st.write("### 8. Simpan Model & Artifacts")

        model_file = f"model_{model_name}.pkl"
        feat_file = f"model_features_{model_name}.pkl"
        le_file = f"label_encoder_{model_name}.pkl"
        cont_file = f"continuous_columns_{model_name}.pkl"
        scaler_file = f"scaler_{model_name}.pkl"

        joblib.dump(model, model_file)
        joblib.dump(list(X.columns), feat_file)
        joblib.dump(le, le_file)
        joblib.dump(cont_exist, cont_file)
        joblib.dump(scaler, scaler_file)

        st.success("Model & file pendukung berhasil disimpan.")

        def _download_btn(path, label):
            with open(path, "rb") as f:
                st.download_button(label=str(label), data=f, file_name=path)

        _download_btn(model_file, f"Download {model_file}")
        _download_btn(feat_file, f"Download {feat_file}")
        _download_btn(le_file, f"Download {le_file}")
        _download_btn(cont_file, f"Download {cont_file}")
        _download_btn(scaler_file, f"Download {scaler_file}")
