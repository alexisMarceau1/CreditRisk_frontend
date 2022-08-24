#-----------------------
#-------run with--------
#-----------------------

# in a bash $ console type:
#  $ streamlit run dash.py

#-----------------------
#----Spécifications-----
#-----------------------

# - Permettre de visualiser le score et l’interprétation de ce score
#   pour chaque client de façon intelligible pour une personne non 
#   experte en data science.

# - Permettre de visualiser des informations descriptives relatives
#   à un client (via un système de filtre).

# - Permettre de comparer les informations descriptives relatives
#   à un client à l’ensemble des clients ou à un groupe de clients 
#   similaires.

#-----------------------
#----Import libraries---
#-----------------------

import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import shap
import json
import requests

from urllib.request import urlopen
import plotly.graph_objects as go 

#-----------------------
#---------App-----------
#-----------------------

LOGO_IMAGE = "logo.png"

#--------------
#--fonctions---
#--------------

@st.cache
def load_data():
        PATH = '../data/'

        #data avant f-engineering
        data_train = pd.read_parquet(PATH+'application_train.parquet') #train set
        data_test = pd.read_parquet(PATH+'application_test.parquet') #test set

        #data après f-engineering
        df = pd.read_parquet(PATH+'test_df.parquet') #test set

        #description des features
        description = pd.read_csv(PATH+'HomeCredit_columns_description.csv', 
                                        usecols=['Row', 'Description'], \
                                        index_col=0, encoding='unicode_escape')

        return df, data_test, data_train, description

@st.cache
def load_model():
        '''loading the trained model'''
        return pickle.load(open('./LGBMClassifier.pkl', 'rb'))
        
@st.cache
def get_client_info(data, id_client):
        client_info = data[data['SK_ID_CURR']==int(id_client)]
        return client_info

#--------------

#Chargement des données    
df, data_test, data_train, description = load_data()

ignore_features = ['Unnamed: 0','SK_ID_CURR', 'INDEX', 'TARGET']
relevant_features = [col for col in df if col not in ignore_features]

#Chargement du modèle
model = load_model()

#-------------------
#--SIDEBAR----------
#-------------------

with st.sidebar:
        st.write("## ID Client")
        id_list = df["SK_ID_CURR"].tolist()
        id_client = st.selectbox(
            "Sélectionner l'identifiant du client", id_list)

        st.write("## Actions à effectuer")
        show_credit_decision = st.checkbox("Afficher la décision de crédit")
        show_client_details = st.checkbox("Afficher les informations du client")
        show_client_comparison = st.checkbox("Comparer aux autres clients")



#-------------------
#--page principale--
#-------------------      

#title
#st.markdown("<h1 style='text-align:center'>Prêt à dépenser</h1>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3) #to center image
with col1:
    st.write(' ')
with col2:
    st.image(LOGO_IMAGE)
with col3:
    st.write(' ')
st.write("---")


#Afficher l'ID Client sélectionné
#st.write("ID Client Sélectionné :", id_client)

if (int(id_client) in id_list):

        client_info = get_client_info(data_test, id_client)

        #-------------------------------------------------------
        # Afficher la décision de crédit
        #-------------------------------------------------------

        if (show_credit_decision):
            st.header('Décision du modèle')

            #Appel de l'API :
             
            #link to the api
            API_url = f"http://127.0.0.1:8000/prediction/{id_client}"

            #open json dict
            json_url = urlopen(API_url)
            API_data = json.loads(json_url.read())

            #show prediction
            if API_data["prediction"] == 1:
                    decision = '❌ Crédit Refusé'
            else:
                    decision = '✅ Crédit Accordé'
            
            st.header(decision)

            #calcul du score...


        #-------------------------------------------------------
        # Afficher les informations du client
        #-------------------------------------------------------

        personal_info_cols = {
            'CODE_GENDER': "GENRE",
            'DAYS_BIRTH': "AGE",
            'NAME_FAMILY_STATUS': "STATUT FAMILIAL",
            'CNT_CHILDREN': "NB ENFANTS",
            'FLAG_OWN_CAR': "POSSESSION VEHICULE",
            'FLAG_OWN_REALTY': "POSSESSION BIEN IMMOBILIER",
            'NAME_EDUCATION_TYPE': "NIVEAU EDUCATION",
            'OCCUPATION_TYPE': "EMPLOI",
            'DAYS_EMPLOYED': "NB ANNEES EMPLOI",
            'AMT_INCOME_TOTAL': "REVENUS",
            'AMT_CREDIT': "MONTANT CREDIT", 
            'NAME_CONTRACT_TYPE': "TYPE DE CONTRAT",
            'AMT_ANNUITY': "MONTANT ANNUITES",
            'NAME_INCOME_TYPE': "TYPE REVENUS",
            'EXT_SOURCE_1': "EXT_SOURCE_1",
            'EXT_SOURCE_2': "EXT_SOURCE_2",
            'EXT_SOURCE_3': "EXT_SOURCE_3",
        }

        default_list=\
        ["GENRE","AGE","STATUT FAMILIAL","NB ENFANTS","REVENUS","MONTANT CREDIT"]
        numerical_features = ['DAYS_BIRTH', 'CNT_CHILDREN', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']

        rotate_label = ["NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE"]
        horizontal_layout = ["OCCUPATION_TYPE", "NAME_INCOME_TYPE"]

        if (show_client_details):
            st.header('Informations relatives au client')

            with st.spinner('Chargement des informations relatives au client...'):
                personal_info_df = client_info[list(personal_info_cols.keys())]
                #personal_info_df['SK_ID_CURR'] = client_info['SK_ID_CURR']
                personal_info_df.rename(columns=personal_info_cols, inplace=True)

                personal_info_df["AGE"] = int(round(personal_info_df["AGE"]/365*(-1)))
                personal_info_df["NB ANNEES EMPLOI"] = \
                int(round(personal_info_df["NB ANNEES EMPLOI"]/365*(-1)))


                filtered = st.multiselect("Choisir les informations à afficher", \
                                          options=list(personal_info_df.columns),\
                                          default=list(default_list))
                df_info = personal_info_df[filtered] 
                df_info['SK_ID_CURR'] = client_info['SK_ID_CURR']
                df_info = df_info.set_index('SK_ID_CURR')

                st.table(df_info.astype(str).T)
                show_all_info = st\
                .checkbox("Afficher toutes les informations")
                if (show_all_info):
                    st.dataframe(client_info)

        #-------------------------------------------------------
        # Comparaison clients
        #-------------------------------------------------------

        if (show_client_comparison):
            st.header('Comparaison clients')
            st.write('En développement...')
            

