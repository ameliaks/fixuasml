import streamlit as st
import streamlit.components.v1 as components


def about_dataset():

    components.html(
        """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        body {
            font-family: 'Inter', sans-serif;
            background: transparent;
            color: #E5E7EB;
        }

        .section-card {
            background: #020617;
            border-radius: 22px;
            padding: 2.4rem;
            margin-bottom: 28px;
            border: 1px solid rgba(255,255,255,0.06);
            box-shadow: 0 20px 45px rgba(0,0,0,0.45);
        }

        .section-title {
            font-size: 26px;
            font-weight: 600;
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .section-text {
            color: #9CA3AF;
            font-size: 15.5px;
            line-height: 1.75;
        }

        .pill {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 999px;
            background: rgba(20,184,166,0.18);
            color: #5EEAD4;
            font-weight: 500;
            font-size: 13.5px;
            margin-right: 10px;
            margin-top: 16px;
            border: 1px solid rgba(94,234,212,0.25);
        }

        .pipeline-card {
            background: #020617;
            border-radius: 16px;
            padding: 1.4rem 1.8rem;
            margin-bottom: 14px;
            border-left: 5px solid #14B8A6;
        }

        .pipeline-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 6px;
        }

        .pipeline-desc {
            font-size: 14.5px;
            color: #9CA3AF;
            line-height: 1.65;
        }
        </style>
        </head>

        <body>

        <!-- ================= DATASET ================= -->
        <div class="section-card">
            <div class="section-title">📁 Tentang Dataset</div>

            <div class="section-text">
                Menurut <b>World Health Organization (WHO)</b>, malnutrisi anak
                merupakan masalah kesehatan global yang berdampak langsung
                pada peningkatan angka kesakitan dan kematian anak,
                khususnya di negara berkembang.
                <br><br>
                Dataset ini digunakan untuk <b>memprediksi status gizi anak</b>
                berdasarkan data antropometri dan faktor kesehatan
                guna membantu identifikasi risiko gizi secara lebih dini.
            </div>

            <div>
                <span class="pill">Normal</span>
                <span class="pill">At Risk</span>
                <span class="pill">Malnourished</span>
            </div>
        </div>

        <!-- ================= ALGORITHM ================= -->
        <div class="section-card">
            <div class="section-title">🤖 Algoritma yang Digunakan</div>

            <div class="section-text">
                Algoritma yang digunakan dalam penelitian ini adalah <b>Logistic Regression Multinomial</b>,
                yaitu metode <i>supervised learning</i> untuk klasifikasi multikelas.
                Model ini digunakan untuk memprediksi status gizi anak ke dalam kategori
                <b>Normal</b>, <b>At Risk</b>, dan <b>Malnourished</b> berdasarkan variabel antropometri
                dan karakteristik kesehatan.
                <br><br>
                Untuk meningkatkan performa model, dilakukan:
            </div>

            <ul class="section-text">
                <li><b>Normalisasi data menggunakan Min-Max Scaling</b></li>
                <li><b>Penanganan ketidakseimbangan kelas menggunakan SMOTE atau class weighting</b></li>
            </ul>
        </div>

        <!-- ================= PIPELINE ================= -->
        <div class="section-title">🧭 Alur Lengkap Regresi Logistik Multinomial</div>

        <div class="pipeline-card">
            <div class="pipeline-title">1. Menentukan Kelas Referensi</div>
            <div class="pipeline-desc">
                Satu kelas dipilih sebagai baseline, sementara kelas lain
                dibandingkan terhadap kelas referensi untuk membentuk model logit multikelas.
            </div>
        </div>

        <div class="pipeline-card">
            <div class="pipeline-title">2. Inisialisasi Parameter</div>
            <div class="pipeline-desc">
                Koefisien model (β) untuk setiap kelas non-referensi
                diinisialisasi dengan nilai awal sebagai titik awal optimasi.
            </div>
        </div>

        <div class="pipeline-card">
            <div class="pipeline-title">3. Menghitung Linear Predictor</div>
            <div class="pipeline-desc">
                Kombinasi linear antara fitur input dan parameter model
                dihitung untuk setiap kelas non-referensi.
            </div>
        </div>

        <div class="pipeline-card">
            <div class="pipeline-title">4. Menghitung Probabilitas (Softmax)</div>
            <div class="pipeline-desc">
                Nilai linear predictor dikonversi menjadi probabilitas
                setiap kelas menggunakan fungsi softmax.
            </div>
        </div>

        <div class="pipeline-card">
            <div class="pipeline-title">5. Menghitung Log-Likelihood</div>
            <div class="pipeline-desc">
                Log-likelihood digunakan untuk mengukur kesesuaian
                antara prediksi model dan data aktual.
            </div>
        </div>

        <div class="pipeline-card">
            <div class="pipeline-title">6. Menghitung Gradient</div>
            <div class="pipeline-desc">
                Gradient dihitung dari selisih antara label aktual
                dan probabilitas prediksi untuk menentukan arah perbaikan parameter.
            </div>
        </div>

        <div class="pipeline-card">
            <div class="pipeline-title">7. Update Parameter</div>
            <div class="pipeline-desc">
                Parameter model diperbarui menggunakan metode optimasi
                seperti Gradient Descent untuk memaksimalkan log-likelihood.
            </div>
        </div>

        <div class="pipeline-card">
            <div class="pipeline-title">8. Iterasi Hingga Konvergen</div>
            <div class="pipeline-desc">
                Proses perhitungan dan pembaruan parameter
                diulang hingga model mencapai kondisi konvergen.
            </div>
        </div>

        <div class="pipeline-card">
            <div class="pipeline-title">9. Prediksi Kelas</div>
            <div class="pipeline-desc">
                Kelas dengan probabilitas tertinggi
                dipilih sebagai hasil prediksi akhir model.
            </div>
        </div>

        <div class="pipeline-card">
            <div class="pipeline-title">10. Interpretasi Koefisien</div>
            <div class="pipeline-desc">
                Koefisien model menunjukkan pengaruh setiap fitur
                terhadap peluang suatu kelas dibandingkan kelas referensi.
            </div>
        </div>

        </body>
        </html>
        """,
        height=1700,
        scrolling=True
    )
