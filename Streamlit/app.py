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
import eda
import prediction

# Importing image, nanti tinggal di define aja di bawah
from PIL import Image