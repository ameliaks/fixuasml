import streamlit as st
import pandas as pd
import altair as alt

def chart():
    df = pd.read_csv("malnutrition_children_ethiopia.csv")

    # Drop id jika ada
    for col_id in ["id", "ID"]:
        if col_id in df.columns:
            df = df.drop(columns=[col_id])

    target_col = "Nutrition_Status"
    if target_col not in df.columns:
        st.error("Kolom 'Nutrition_Status' tidak ditemukan di dataset.")
        st.stop()

    total = df.shape[0]
    status_counts_all = df[target_col].fillna("Unknown").value_counts()

    # Metric utama
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 3, 3])

    with col1:
        st.metric("Total Anak", total)

    with col2:
        st.metric("Normal", int(status_counts_all.get("Normal", 0)))

    with col3:
        st.metric("At_Risk", int(status_counts_all.get("At_Risk", 0)))

    with col4:
        st.metric("Malnourished", int(status_counts_all.get("Malnourished", 0)))

    # Filter
    st.write("### Filter")
    c1, c2, c3 = st.columns(3)

    # filter Nutrition_Status
    with c1:
        selected_status = st.multiselect(
            "Nutrition Status",
            options=sorted(df[target_col].dropna().unique().tolist()),
            default=[]
        )

    # cari kolom gender jika ada (nama bisa beda-beda)
    possible_gender_cols = ["gender", "Gender", "Sex", "sex"]
    gender_col = next((c for c in possible_gender_cols if c in df.columns), None)

    with c2:
        if gender_col:
            selected_gender = st.multiselect(
                "Gender",
                options=sorted(df[gender_col].dropna().astype(str).unique().tolist()),
                default=[]
            )
        else:
            selected_gender = []
            st.info("Kolom gender tidak ada di dataset.")

    # filter umur kalau kolom umur ada
    age_cols = ["Age (months)", "Age_months", "age_months", "age"]
    age_col = next((c for c in age_cols if c in df.columns), None)

    with c3:
        if age_col and pd.api.types.is_numeric_dtype(df[age_col]):
            min_age = float(df[age_col].min())
            max_age = float(df[age_col].max())
            age_range = st.slider("Rentang Umur", min_age, max_age, (min_age, max_age))
        else:
            age_range = None
            st.info("Kolom umur tidak ditemukan/ tidak numerik.")

    # Apply filter
    filtered_df = df.copy()

    if selected_status:
        filtered_df = filtered_df[filtered_df[target_col].isin(selected_status)]

    if gender_col and selected_gender:
        filtered_df = filtered_df[filtered_df[gender_col].astype(str).isin(selected_gender)]

    if age_col and age_range is not None:
        filtered_df = filtered_df[(filtered_df[age_col] >= age_range[0]) & (filtered_df[age_col] <= age_range[1])]

    st.write("### Data (contoh 10 baris)")
    st.dataframe(filtered_df.head(10), use_container_width=True)

    # ===== Charts =====
    st.write("### Visualisasi")

    # 1) Pie: distribusi Nutrition_Status
    status_counts = (
        filtered_df[target_col]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
    )
    status_counts.columns = [target_col, "count"]

    pie = alt.Chart(status_counts).mark_arc(innerRadius=40).encode(
        theta=alt.Theta(field="count", type="quantitative"),
        color=alt.Color(field=target_col, type="nominal"),
        tooltip=[alt.Tooltip(f"{target_col}:N"), alt.Tooltip("count:Q")]
    ).properties(height=320, title="Distribusi Nutrition Status")

    st.altair_chart(pie, use_container_width=True)

    # 2) Histogram: Age, Height, Weight (kalau ada)
    num_candidates = ["Age (months)", "Height_cm", "Weight_kg"]
    num_cols_exist = [c for c in num_candidates if c in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[c])]

    if len(num_cols_exist) > 0:
        for col in num_cols_exist:
            st.write(f"**Distribusi {col}**")
            hist = alt.Chart(filtered_df).mark_bar().encode(
                x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=30), title=col),
                y=alt.Y("count():Q", title="Frekuensi"),
                tooltip=[alt.Tooltip("count():Q", title="Frekuensi")]
            ).properties(height=280)
            st.altair_chart(hist, use_container_width=True)
    else:
        st.info("Kolom numerik utama (Age/Height/Weight) tidak ditemukan.")

    # 3) Scatter: Weight vs Height (warna = status)
    if "Height_cm" in filtered_df.columns and "Weight_kg" in filtered_df.columns:
        st.write("**Hubungan Tinggi & Berat (warna = Nutrition Status)**")
        scatter = alt.Chart(filtered_df).mark_circle(size=60).encode(
            x=alt.X("Height_cm:Q", title="Tinggi (cm)"),
            y=alt.Y("Weight_kg:Q", title="Berat (kg)"),
            color=alt.Color(f"{target_col}:N", title="Status"),
            tooltip=[target_col, "Height_cm", "Weight_kg"] + ([age_col] if age_col else [])
        ).interactive().properties(height=320)
        st.altair_chart(scatter, use_container_width=True)

    # 4) Boxplot: Age by Nutrition_Status (kalau umur ada)
    if age_col and pd.api.types.is_numeric_dtype(filtered_df[age_col]):
        st.write("**Distribusi Umur berdasarkan Nutrition Status (Box Plot)**")
        box = alt.Chart(filtered_df).mark_boxplot(extent=1.5).encode(
            x=alt.X(f"{target_col}:N", title="Nutrition Status"),
            y=alt.Y(f"{age_col}:Q", title=age_col),
            color=alt.Color(f"{target_col}:N", legend=None)
        ).properties(height=320)
        st.altair_chart(box, use_container_width=True)

    # 5) Bar: jumlah status per gender (kalau gender ada)
    if gender_col:
        st.write("**Perbandingan Nutrition Status berdasarkan Gender**")
        grp = (
            filtered_df.groupby([gender_col, target_col])
            .size()
            .reset_index(name="count")
        )
        bar = alt.Chart(grp).mark_bar().encode(
            x=alt.X(f"{gender_col}:N", title="Gender"),
            y=alt.Y("count:Q", title="Jumlah"),
            color=alt.Color(f"{target_col}:N", title="Status"),
            tooltip=[gender_col, target_col, "count"]
        ).properties(height=320)
        st.altair_chart(bar, use_container_width=True)
