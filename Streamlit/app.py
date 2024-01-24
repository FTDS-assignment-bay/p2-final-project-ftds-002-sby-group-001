'''
Final Project Group 001 FTDS
Title : "In a very competitive food delivery market, understanding customer behavior is important for the success of marketing campaigns. 
This project aims to implement data science techniques to optimize iFood’s marketing strategies. 
We will develop a classification model to predict customer purchasing behavior in upcoming campaigns, and a clustering model to personalize the customers. 
By focusing on specific customer clusters, we can maximize iFood’s profit in future campaigns."

Role that will work in this deployment :
- Data Analyst : Ronan as EDA supervisor
- Data Science : Fariz as prediction supervisor
- Data Engineer : Syahrul as deployment engineer and streamlit dev
'''

# Import pandas and streamlit
import pandas as pd
import streamlit as st 

# Import other files in the same folder
import eda1
import eda2
import prediction

# Importing image, nanti tinggal di define aja di bawah
from PIL import Image

image_1 = Image.open('iFood_logo.png')


page = st.sidebar.selectbox('Choose Your Page: ',
                            ('Welcome Page','Data Analysis Exploration Page', 'Prediction Page', 'Model Analysis Page'))

if page == 'Welcome Page':
    st.title('Welcome to xxx')
    st.write('')
    st.write('Group     : FTDS 001')
    st.write('Batch     : SBY-FTDS-002')
    st.write('Objective :')
    st.write('')
    st.write('Please Select menu on the left bar')
    st.image(image_1,caption="iFood Logo")
    with st.expander('Background'):
        st.caption("Di tengah pasar pengiriman makanan yang sangat kompetitif, memahami perilaku pelanggan sangat penting untuk keberhasilan kampanye pemasaran. Proyek yang kami lakukan bertujuan untuk memanfaatkan teknik-teknik data science guna mengoptimalkan strategi pemasaran sebuah perusahaan bernama iFood. Kami akan mengembangkan model klasifikasi untuk memprediksi perilaku pembelian pelanggan dalam kampanye iFood berikutnya, dan model clustering untuk mempersonalisasi pengalaman pelanggan. Dengan berfokus pada kelompok-kelompok pelanggan tertentu, iFood dapat memaksimalkan keuntungan yang dapat mereka peroleh dalam kampanye-kampanye mendatang." )
    with st.expander('Problem Statement'):
        st.caption("Apakah dengan membuat model machine learning dapat memperbaiki kualitas campaign untuk campaign selanjutnya")
    with st.expander('SPONSORED BY:'):
        st.caption("HACKTIV8")
        st.image(image_1)
elif page == 'Data Analysis Exploration Page':
    eda1.run()
elif page == 'Prediction Page':
    prediction.run()
else:
    eda2.run()
    