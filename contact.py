import streamlit as st


def contact_app():

    st.markdown("## 📬 Get in Touch")

    st.caption(
        "Terbuka untuk diskusi, kolaborasi, maupun masukan terkait "
        "**Prediksi Status Gizi Anak menggunakan Machine Learning**."
    )

    st.markdown("---")

    col1, col2 = st.columns([1.2, 3])

    with col1:
        st.image(
            "C:/Users/lenovo/Downloads/machine learning/Desain tanpa judul (2).png",
            width=140
        )

    with col2:
        st.markdown("### 👩‍🎓 Amelia Kusuma Wardani")
        st.markdown(
            """
            **S1 Sains Data – Universitas Muhammadiyah Semarang**  
            Fokus pada *Data Analysis, Machine Learning,* dan *Data Visualization*.
            """
        )

        st.markdown("#### 📇 Kontak & Profil")
        st.markdown(
            """
            📧 **Email**  
            ameliakusuma@email.com  

            💻 **GitHub**  
            https://github.com/ameliaks  

            🔗 **LinkedIn**  
            https://www.linkedin.com/in/ameliakusuma2203
            """
        )

    st.markdown("---")

    st.success(
        "💡 **Catatan**  \n"
        "Proyek ini dikembangkan sebagai bagian dari **UAS Machine Learning Semester 5**.  \n"
        "Masukan, kritik, dan saran sangat terbuka untuk pengembangan lebih lanjut."
    )
