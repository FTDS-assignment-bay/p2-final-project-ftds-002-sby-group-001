import streamlit as st 
import pickle
import json
import pandas as pd
import numpy as np
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

def run():
    #title
    st.title("***iFood*** Model Prediction")
    
    #load classification
        # Load the trained model
    with open('best_model.pkl', 'rb') as f_1:
        best_model = pickle.load(f_1)
        
    # Load the fitted StandardScaler
    with open('scaler.pkl', 'rb') as f_2:
        scaler = pickle.load(f_2)
        
    # Load the data from the text file
    with open('columns.txt', 'r') as f_3:
        data = json.load(f_3)
        
    #load clustering
    # Load the trained model
    with open('clustering_model.pkl', 'rb') as f_4:
        clustering_model = pickle.load(f_4)

    # Load the fitted StandardScaler
    with open('clustering_scaler.pkl', 'rb') as f_5:
        scaler_cluster = pickle.load(f_5)

    # Load the fitted pca
    with open('pca.pkl', 'rb') as f_6:
        pca = pickle.load(f_6)

    # Load the data from the text file
    with open('clustering_columns.txt', 'r') as f_7:
        data_clustering = json.load(f_7)

        
    # Extract the lists from the dictionary
    num_col = data['num_col']
    cat_col = data['cat_col']
    
    num_col_clust = data_clustering['num_col']
    cat_col_clust = data_clustering['cat_col']
    
    # Create formulir
    with st.form('Form_iFood_Prediction'):
        # Assuming 'id', 'year_birth', 'education', etc. are already defined
        # You can replace them with actual values or input methods
        # Input data
        id = st.number_input('ID', value=5443, step=1) # 1
        year_birth = st.number_input('Year of Birth', value=1989, step=1) # 2
        education = st.radio('Education', ['Graduation', 'PhD', 
                                        'Master', 'Basic', '2n Cycle']) # 3
        marital_status = st.radio('Marital Status', ['Single', 'Together', 
                                                    'Married', 'Divorced', 'Widow', 
                                                    'Alone','Absurd', 'YOLO']) # 4
        income = st.number_input('Income', value=25000, step=1) # 5
        kidhome = st.radio('Number of Kid at Home',(0,1,2)) # 6
        teenhome = st.radio('Number of Teens at Home',(0,1)) # 7
        # date_input = st.date_input('Customer Since', datetime.date(2024, 1, 1)) # 8
        recency = st.slider("Customer Last Purchace", 1, 100, 12) # 9
        
        # Add inputs for the remaining variables
        # Amounts
        mnt_wines = st.slider("Amount Wines Products", 0, 5000, 350) # 10
        mnt_fruits = st.slider("Amount Fruits Products", 0, 5000, 350) # 11
        mnt_meat_products = st.slider("Amount Meat Products", 0, 5000, 350) # 12
        mnt_fish_products = st.slider("Amount Fish Products", 0, 5000, 350) # 13
        mnt_sweet_products = st.slider("Amount Sweet Products", 0, 5000, 350) # 14
        mnt_gold_prods = st.slider("Amount Gold Products", 0, 5000, 350) # 15
        
        # Num Pruchases Site
        num_deals_purchases = st.slider("Number of Deals Purchases", 0, 50, 10) # 16
        num_web_purchases = st.slider("Number of Website Purchases", 0, 50, 10) # 17
        num_catalog_purchases = st.slider("Number of Catalog Purchases", 0, 50, 10) # 18
        num_store_purchases = st.slider("Number of Store Purchases", 0, 50, 10) # 19
        num_web_visits_month = st.slider("Number of Website Visits", 0, 50, 10) # 20
        
        # Campaigns
        accepted_cmp3 = st.radio('Accepted Campaign 3', (0,1)) # 21
        accepted_cmp4 = st.radio('Accepted Campaign 4', (0,1)) # 22
        accepted_cmp5 = st.radio('Accepted Campaign 5', (0,1)) # 23
        accepted_cmp1 = st.radio('Accepted Campaign 1', (0,1)) # 24
        accepted_cmp2 = st.radio('Accepted Campaign 2', (0,1)) # 25
        
        # Complain and Response
        complain = st.radio('Did Customer Complain? (0 = No, 1 = Yes )', (0,1)) # 26
        response = st.radio('Did Customer Have Any Response From Our Campaught ? (0 = No, 1 = Yes)', (0,1)) # 27
        
        submitted = st.form_submit_button("Predict")
        
    df = {'Unnamed: 0': 0,
        'id': id, # int (1)
        'year_birth': year_birth, # int (2)
        'education': education, # Radio String (3)
        'marital_status': marital_status, # Radio String (4)
        'income': income, # float (5)
        'kidhome': kidhome, # int (6)
        'teenhome': teenhome, # int (7)
        'dt_customer': '04-09-2012', # input date (8)
        'recency': recency, # int (9)
        'mnt_wines': mnt_wines, # int (10)
        'mnt_fruits': mnt_fruits, # int (11)
        'mnt_meat_products': mnt_meat_products, # int (12)
        'mnt_fish_products': mnt_fish_products, # int (13)
        'mnt_sweet_products': mnt_sweet_products, # int (14)
        'mnt_gold_prods': mnt_gold_prods, # int (15)
        'num_deals_purchases': num_deals_purchases, # int (16)
        'num_web_purchases': num_web_purchases, # int (17)
        'num_catalog_purchases': num_catalog_purchases, # int (18)
        'num_store_purchases': num_store_purchases, # int (19)
        'num_web_visits_month': num_web_visits_month, # int (20)
        'accepted_cmp3': accepted_cmp3, # int 0 or 1 (21)
        'accepted_cmp4': accepted_cmp4, # int 0 or 1 (22)
        'accepted_cmp5': accepted_cmp5, # int 0 or 1 (23)
        'accepted_cmp1': accepted_cmp1, # int 0 or 1 (24)
        'accepted_cmp2': accepted_cmp2, # int 0 or 1 (25)
        'complain': complain, # int 0 or 1 (26)
        'response': response # int 0 or 1 (27)
        }
    
    data_inf = pd.DataFrame([df])
    data_inf_2 = pd.DataFrame([df])
    
    st.dataframe(data_inf)
    
    if submitted:
        # Create new features from amount features
        data_inf["total_mnt"] = data_inf["mnt_wines"] + data_inf["mnt_fruits"] + data_inf["mnt_meat_products"] + data_inf["mnt_fish_products"] + data_inf["mnt_sweet_products"] + data_inf["mnt_gold_prods"]
        
        # Filter out rows where year_birth is less than 1928
        data_inf = data_inf[data_inf["year_birth"] >= 1928]
        
        # Define generation labels and ranges
        generations = {
            "Silent Generation": (1928, 1945),
            "Baby Boomers": (1946, 1964),
            "Generation X": (1965, 1980),
            "Millennials": (1981, 1996)
        }
        
        # Create a function to assign generation label
        def assign_generation(year):
            for gen, (start, end) in generations.items():
                if start <= year <= end:
                    return gen
                
        # Apply the function to the year_birth feature
        data_inf["generation"] = data_inf["year_birth"].apply(assign_generation)
        
        data_inf["dt_customer"] = pd.to_datetime(data_inf["dt_customer"], format="%d-%m-%Y")
        
        # Create new features from date features
        data_inf["customer_since"] = (dt.datetime(2015, 1, 1) - data_inf["dt_customer"]).dt.days
        
        # Drop unnecessary columns
        data_inf = data_inf.drop(["Unnamed: 0", "id", "dt_customer"], axis=1) ## these columns won't help the model
        
        data_num = data_inf[num_col]
        data_cat = data_inf[cat_col]
        
        def encoder(df):
            # Define the mappings for each variable
            education_mapping = {'PhD': 0, 'Basic': 1, 'Graduation': 2, 'Master': 3, '2n Cycle': 4}
            marital_status_mapping = {'Together': 0, 'Married': 1, 'Single': 2, 'Divorced': 3, 'Widow': 4, 'Alone': 5, 'YOLO': 6, 'Absurd': 7}
            accepted_cmp_mapping = {0: 0, 1: 1}
            generation_mapping = {'Silent Generation':0, 'Baby Boomers': 1, 'Millennials': 2, 'Generation X': 3}
            
            # Apply the mappings to the DataFrame
            df['education'] = df['education'].map(education_mapping)
            df['marital_status'] = df['marital_status'].map(marital_status_mapping)
            df['accepted_cmp1'] = df['accepted_cmp1'].map(accepted_cmp_mapping)
            df['accepted_cmp2'] = df['accepted_cmp2'].map(accepted_cmp_mapping)
            df['accepted_cmp3'] = df['accepted_cmp3'].map(accepted_cmp_mapping)
            df['accepted_cmp4'] = df['accepted_cmp4'].map(accepted_cmp_mapping)
            df['accepted_cmp5'] = df['accepted_cmp5'].map(accepted_cmp_mapping)
            df['generation'] = df['generation'].map(generation_mapping)
            
            return df
    
        # Apply the function to the training and test data
        data_cat_encoded = encoder(data_cat)
                
        data_num_scaled = scaler.transform(data_num)
                
        data_final = np.concatenate([data_num_scaled, data_cat_encoded], axis=1)
                
        data_final_pred = best_model.predict(data_final)
        data_final_pred
        
        # st.write('# Response: ', data_final_pred)
    
    # Start of Clustering    
    if data_final_pred == 1 :
        st.write("This customer will buy the product from the campaign")
    elif data_final_pred == 0:
        # Create new features from amount features
        data_inf_2["total_mnt"] = data_inf_2["mnt_wines"] + data_inf_2["mnt_fruits"] + data_inf_2["mnt_meat_products"] + data_inf_2["mnt_fish_products"] + data_inf_2["mnt_sweet_products"] + data_inf_2["mnt_gold_prods"]
        # Filter out rows where year_birth is less than 1928
        data_inf_2 = data_inf_2[data_inf_2["year_birth"] >= 1928]

        # Define generation labels and ranges
        generations = {
            "Silent Generation": (1928, 1945),
            "Baby Boomers": (1946, 1964),
            "Generation X": (1965, 1980),
            "Millennials": (1981, 1996)
        }

        # Create a function to assign generation label
        def assign_generation(year):
            for gen, (start, end) in generations.items():
                if start <= year <= end:
                    return gen

        # Apply the function to the year_birth feature
        data_inf_2["generation"] = data_inf_2["year_birth"].apply(assign_generation)
        data_inf_2["dt_customer"] = pd.to_datetime(data_inf_2["dt_customer"], format="%d-%m-%Y")

        # Create new features from date features
        data_inf_2["customer_since"] = (dt.datetime(2015, 1, 1) - data_inf_2["dt_customer"]).dt.days
        
        # Drop unnecessary columns
        data_inf_2 = data_inf_2.drop(["Unnamed: 0", "id", "dt_customer", "response"], axis=1) ## these columns won't help the model
        
        data_num_clustering = data_inf_2[num_col_clust]
        data_cat_clustering = data_inf_2[cat_col_clust]
        
        def encoder(df):
            # Define the mappings for each variable
            education_mapping = {'PhD': 0, 'Basic': 1, 'Graduation': 2, 'Master': 3, '2n Cycle': 4}
            marital_status_mapping = {'Together': 0, 'Married': 1, 'Single': 2, 'Divorced': 3, 'Widow': 4, 'Alone': 5, 'YOLO': 6, 'Absurd': 7}
            binary_mapping = {0: 0, 1: 1}
            generation_mapping = {'Silent Generation':0, 'Baby Boomers': 1, 'Millennials': 2, 'Generation X': 3}

            # Apply the mappings to the DataFrame
            df['education'] = df['education'].map(education_mapping)
            df['marital_status'] = df['marital_status'].map(marital_status_mapping)
            df['accepted_cmp1'] = df['accepted_cmp1'].map(binary_mapping)
            df['accepted_cmp2'] = df['accepted_cmp2'].map(binary_mapping)
            df['accepted_cmp3'] = df['accepted_cmp3'].map(binary_mapping)
            df['accepted_cmp4'] = df['accepted_cmp4'].map(binary_mapping)
            df['accepted_cmp5'] = df['accepted_cmp5'].map(binary_mapping)
            df['complain'] = df['complain'].map(binary_mapping)
            df['generation'] = df['generation'].map(generation_mapping)

            return df
        # Apply the function to the training and test data
        
        data_cat_encoded_cluster = encoder(data_cat_clustering)
        data_num_scaled_cluster = scaler_cluster.transform(data_num_clustering)
        
        data_num_scaled_df = pd.DataFrame(data_num_scaled_cluster, columns=num_col_clust)
        data_reduced = pca.transform(data_num_scaled_df)
        data_final = np.concatenate([data_reduced, data_cat_encoded_cluster], axis=1)
        
        data_final_pred_cluster = clustering_model.predict(data_final, categorical=[13, 14, 15, 16, 17, 18, 19, 20, 21])
        # st.write('# Cluster Response: ', data_final_pred_cluster)
        
        if data_final_pred_cluster == 0:
            st.write("Customer wont buy the products cause")
        elif data_final_pred_cluster == 1:
            st.write("Customer wont buy the products cause")
        elif data_final_pred_cluster == 2:
            st.write("Customer wont buy the products cause")
if __name__=='__main__':
    run()
    
    