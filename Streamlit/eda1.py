# Libraries Import
import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as ps
import seaborn as sns

# Non Pre-made and Pre-made library/script importing
from PIL import Image
image_1 = Image.open('iFood_logo.png')

def run():
    df = pd.read_csv('data_cleaned.csv')
    st.title("***Exploring Acceptance Rate For The Campaigns***")
    
    st.write("")
    
    
    
    
if __name__=='__main__':
    run()
    