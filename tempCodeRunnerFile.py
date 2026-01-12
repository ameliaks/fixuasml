import streamlit as st
import pandas as pd
import numpy as np

st.header('Prediksi Gizi Anak Berdasarkan Data Kesehatan Menggunakan Machine Learning')
st.write('**Project UAS Machine Learning - Amelia Kusuma (B2D023001)')

tab1, tab2, tab3, tab4, tab5 = st.tabs(['About Dataset', 
                            'Dashoboards', 
                            'Machine Learning',
                            'Prediction App',
                            'Contact Me'])

with tab1:
    import about
    about.about_dataset()

with tab2:
    import visualisasi
    visualisasi.chart()

with tab3:
    import machine_learning
    machine_learning.ml_model()

with tab4:
    import prediction
    prediction.prediction_app()

