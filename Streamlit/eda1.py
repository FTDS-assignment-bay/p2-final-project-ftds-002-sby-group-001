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
    target = df["Response"].value_counts().reset_index()
    persen = df["Response"].value_counts(normalize=True).reset_index()
    target["percentage"] = persen["proportion"]
    st.dataframe(target)
    
    
if __name__=='__main__':
    run()
    