#-----------------------
#-------run with--------
#-----------------------

# in a bash $ console type:
# $ streamlit run dash.py

#-----------------------
#----Specifications-----
#-----------------------

# - Allow to visualize the score and the interpretation of this score
# for each customer in a way that is intelligible to a person who is not
# an expert in data science.

# - Allow to visualize descriptive information relating
# to a customer (via a filter system).

# - Allow to compare descriptive information relating
# to a customer to all customers or to a group of similar customers.

#-----------------------
#----Import libraries---
#------------------------
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import shap
import json
import math
import plotly.graph_objects as go 
import warnings
warnings.filterwarnings("ignore")

st.set_option('client.showErrorDetails', False)
st.set_page_config(layout="wide")



#import joblib

from streamlit_shap import st_shap
from urllib.request import urlopen

#-----------------------
#---------App-----------
#-----------------------

LOGO_IMAGE = "logo.jpg"
SHAP_GENERAL1 = "glob1.png"
SHAP_GENERAL2 = "glob2.png"
SEUIL = "seuil.png"

#---------------
#--fonctions----
#---------------

@st.cache
def load_data():
        #PATH = 'P7_data/data/'

        #data avant f-engineering
        data_train = pd.read_parquet('application_train.parquet') #train set
        data_test = pd.read_parquet('application_test.parquet') #test set

        #data après f-engineering
        df = pd.read_parquet('test_df.parquet') #test set
        df = df.iloc[:50,:]

        #description des features
        description = pd.read_csv('HomeCredit_columns_description.csv', 
                                        usecols=['Row', 'Description'], \
                                        index_col=0, encoding='unicode_escape')

        return df, data_test, data_train, description

@st.cache
def load_model():
        '''loading the trained model'''
        return pickle.load(open('LGBMClassifier.pkl', 'rb'))

@st.cache
def load_explainer():
        '''loading the trained explainer'''
        return pickle.load(open('lgbm_explainer_local.pkl', 'rb'))
        
@st.cache
def get_client_info(data, id_client):
        client_info = data[data['SK_ID_CURR']==int(id_client)]
        return client_info

#@st.cache
def plot_distribution(applicationDF,feature, client_feature_val, title):

    if (not (math.isnan(client_feature_val))):
        fig = plt.figure(figsize = (10, 4))

        t0 = applicationDF.loc[applicationDF['TARGET'] == 0]
        t1 = applicationDF.loc[applicationDF['TARGET'] == 1]

        if (feature == "DAYS_BIRTH"):
            sns.kdeplot((t0[feature]/-365).dropna(), label = 'Remboursé', color='g')
            sns.kdeplot((t1[feature]/-365).dropna(), label = 'Défaillant', color='r')
            plt.axvline(float(client_feature_val/-365), \
                        color="blue", linestyle='--', label = 'Position Client')

        elif (feature == "DAYS_EMPLOYED"):
            sns.kdeplot((t0[feature]/365).dropna(), label = 'Remboursé', color='g')
            sns.kdeplot((t1[feature]/365).dropna(), label = 'Défaillant', color='r')    
            plt.axvline(float(client_feature_val/365), color="blue", \
                        linestyle='--', label = 'Position Client')

        else:    
            sns.kdeplot(t0[feature].dropna(), label = 'Remboursé', color='g')
            sns.kdeplot(t1[feature].dropna(), label = 'Défaillant', color='r')
            plt.axvline(float(client_feature_val), color="blue", \
                        linestyle='--', label = 'Position Client')


        plt.title(title, fontsize='20', fontweight='bold')
        #plt.ylabel('Nombre de clients')
        #plt.xlabel(fontsize='14')
        plt.legend()
        plt.show()  
        st.pyplot(fig)
    else:
        st.write("Comparaison impossible car la valeur de cette variable n'est pas renseignée (NaN)")

#@st.cache
def univariate_categorical(applicationDF,feature,client_feature_val,\
                            titre,ylog=False,label_rotation=False,
                            horizontal_layout=True):
    if (client_feature_val.iloc[0] != np.nan):

        temp = applicationDF[feature].value_counts()
        df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

        categories = applicationDF[feature].unique()
        categories = list(categories)

        # Calculate the percentage of target=1 per category value
        cat_perc = applicationDF[[feature,\
                                    'TARGET']].groupby([feature],as_index=False).mean()
        cat_perc["TARGET"] = cat_perc["TARGET"]*100
        cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

        if(horizontal_layout):
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
        else:
            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20,24))

        # 1. Subplot 1: Count plot of categorical column
        # sns.set_palette("Set2")
        s = sns.countplot(ax=ax1, 
                        x = feature, 
                        data=applicationDF,
                        hue ="TARGET",
                        order=cat_perc[feature],
                        palette=['g','r'])

        pos1 = cat_perc[feature].tolist().index(client_feature_val.iloc[0])
        #st.write(client_feature_val.iloc[0])

        # Define common styling
        ax1.set(ylabel = "Nombre de clients")
        ax1.set_title(titre, fontdict={'fontsize' : 15, 'fontweight' : 'bold'})   
        ax1.axvline(int(pos1), color="blue", linestyle='--', label = 'Position Client')
        ax1.legend(['Position Client','Remboursé','Défaillant' ])

        # If the plot is not readable, use the log scale.
        if ylog:
            ax1.set_yscale('log')
            ax1.set_ylabel("Count (log)",fontdict={'fontsize' : 15, \
                                                    'fontweight' : 'bold'})   
        if(label_rotation):
            s.set_xticklabels(s.get_xticklabels(),rotation=90)

        # 2. Subplot 2: Percentage of defaulters within the categorical column
        s = sns.barplot(ax=ax2, 
                        x = feature, 
                        y='TARGET', 
                        order=cat_perc[feature], 
                        data=cat_perc,
                        palette='Set2')

        pos2 = cat_perc[feature].tolist().index(client_feature_val.iloc[0])
        #st.write(pos2)

        if(label_rotation):
            s.set_xticklabels(s.get_xticklabels(),rotation=90)
        plt.ylabel('Pourcentage de défaillants [%]', fontsize=10)
        plt.tick_params(axis='both', which='major', labelsize=10)
        ax2.set_title(titre+" (% Défaillants)", \
                        fontdict={'fontsize' : 15, 'fontweight' : 'bold'})
        ax2.axvline(int(pos2), color="blue", linestyle='--', label = 'Position Client')
        ax2.legend()
        plt.show()
        st.pyplot(fig)
    else:
        st.write("Comparaison impossible car la valeur de cette variable n'est pas renseignée (NaN)")

#--------------

#load data  
df, data_test, data_train, description  = load_data()

ignore_features = ['Unnamed: 0','SK_ID_CURR', 'INDEX', 'TARGET']
relevant_features = [col for col in df if col not in ignore_features]

#load model
model = load_model()
explainer_local = shap.Explainer(model, df[relevant_features])

#-------------------
#--SIDEBAR----------
#-------------------

with st.sidebar:
        st.write("## ID Client")
        id_list = df["SK_ID_CURR"].tolist()
        id_client = st.selectbox(
            "Select Client ID", id_list)

        st.write("## Select an Option")
        show_credit_decision = st.checkbox("Display Credit Decision")
        show_client_details = st.checkbox("Display Client Information")
        show_client_comparison = st.checkbox("Compare with Other Clients")
        shap_general = st.checkbox("Display Global Feature Importance")
        if(st.checkbox("Feature Descriptions")):
            list_features = description.index.to_list()
            list_features = list(dict.fromkeys(list_features))
            feature = st.selectbox('Sélectionner une variable',\
                                   sorted(list_features))
            
            desc = description['Description'].loc[description.index == feature][:1]
            st.markdown('**{}**'.format(desc.iloc[0]))



#-------------------
#--main page--------
#-------------------      

#Title
st.markdown(
    """
    <h1 style='text-align: center; font-family: "Georgia", serif; font-size: 70px; color: #2C3E50; font-weight: bold;'>
    CreditRisk
    </h1>
    """,
    unsafe_allow_html=True
)

# Sub-title
st.markdown(
    """
    <h3 style='text-align: center; font-family: "Arial", sans-serif; font-size: 20px; color: #34495E; font-weight: normal;'>
     Predictive analysis and interpretation of credit risk
    </h3>
    """,
    unsafe_allow_html=True
)

# Line
st.markdown("<hr style='border: 1px solid #BDC3C7; margin: 20px 0;'>", unsafe_allow_html=True)


#st.image(LOGO_IMAGE, use_column_width=True)


#Afficher l'ID Client sélectionné
#st.write("ID Client Sélectionné :", id_client)

if (int(id_client) in id_list):

        client_info = get_client_info(data_test, id_client)

        #-------------------------------------------------------
        # Afficher la décision de crédit
        #-------------------------------------------------------

        if (show_credit_decision):
            st.markdown("<h2 style='text-align: center;'>Model decision</h2>", unsafe_allow_html=True)
            #st.image(SEUIL)
            # with st.expander("🔍 Choix du seuil"):
            #     st.write("Le modèle est entrainé pour minimiser les risques\
            #     d'accorder un prêt à un client qui ne peut pas rembourser.\
            #     Plus le seuil choisi est élevé plus le risque de perte est fort.")

            #     st.write("Un seuil de 0.2 est idéal si l'on souhaite minimiser ce risque.\
            #     Cependant, le manque à gagner peut être plus important.")

            #     st.write("Le seuil de 0.5 est le seuil par défaut.")
            
            #Appel de l'API :
             
            #link to the api
            API_url = f"http://127.0.0.1:8000/prediction/{id_client}" #local
            #API_url = f"https://api-alexis-p7.herokuapp.com/prediction/{id_client}" #web
            

            #open json dict
            json_url = urlopen(API_url)
            API_data = json.loads(json_url.read())

            #seuil_list =['Défaut (50%)', 'Minimisation risque (20%)', 'Personnalisé']
            #seuil = st.selectbox(
            #"Sélectionner le seuil", seuil_list)

            classe_predite = API_data['prediction']
            proba = 1-API_data['proba']
            client_score = round(proba*100, 2) #calcul du score
            
            fig = go.Figure(go.Indicator(
            domain = {'x': [0, 1], 'y': [0, 1]},
            value = client_score,
            mode = "gauge+number",
            title = {'text': "Risk of default (in %)"},
            gauge = {'axis': {'range': [None, 100]},
                    'steps' : [
                        {'range': [0, 20], 'color': "lightgreen"},
                        {'range': [20, 40], 'color': "yellow"},
                        {'range': [40, 60], 'color': "orange"},
                        {'range': [60, 100], 'color': "red"}],
                    'threshold' : {'line': {'color': "black", 'width': 10}, 'thickness': 0.6, 'value': client_score},
                    'bar': {'color': "black", 'thickness' : 0.2}}))
        
            st.plotly_chart(fig, use_container_width=True)
            
            if client_score < 20:
                st.markdown("<h2 style='text-align: center; color: lightgreen;'>✅ No risk</h2>", unsafe_allow_html=True)
            elif (client_score > 20) & (client_score < 40):
                st.markdown("<h2 style='text-align: center; color:  #FFD700;'>Low risk and possibility of identifying a bad payer</h2>", unsafe_allow_html=True)
            elif (client_score > 40) & (client_score < 60):
                st.markdown("<h2 style='text-align: center; color: orange;'>Medium risk and possibility of identifying a good payer</h2s>", unsafe_allow_html=True)
            elif client_score > 60:
                st.markdown("<h2 style='text-align: center; color: red;'>⚠ High risk</h2>", unsafe_allow_html=True)

            st.write("\n")
            st.write("\n")

            with st.expander("🔍 Global Risk Analysis"):
                st.image(SEUIL, use_column_width=True)  # Adapte l'image à la largeur de l'expander
                st.write("---")
                st.write("The model is trained to minimize the risk of granting a loan to a client who cannot repay.")

                st.write("The above curve shows the statistical expected gain based on the threshold. The threshold is the probability (between 0 and 1) that a client will not repay their loan.")

                st.write("Examples:")
                st.write(" - For a threshold of 0, no clients are accepted, so there is no potential gain.")
                st.write(" - For a threshold below 0.2, only clients with a default probability below 20% are accepted.")
                st.write(" - For a threshold below 0.4, only clients with a default probability below 40% are accepted. Risks are low, and potential gains are high, but it is still possible to identify defaulters.")
                st.write(" - For a threshold below 0.6, only clients with a default probability below 60% are accepted. Risks are moderate, but it is still possible to identify good payers.")
                st.write(" - For a threshold above 0.6, the risk of loss is high. Business expertise is required using local interpretability.")


                

            #adapt threshold to the client choice
            #if seuil == 'Défaut (50%)':
                #seuil_value = 50
            #elif seuil == 'Minimisation risque (20%)':
                #seuil_value = 20
            #elif seuil == 'Personnalisé':
                #seuil_value=st.number_input("Seuil:",)

            #show prediction
            #if client_score < seuil_value:
                    #decision = '✅ Crédit Accordé'
            #else:
                    #decision = '❌ Crédit Refusé'  

            
            #left_column, right_column = st.columns((1, 2))
            #left_column.markdown('(Risque de défaut: **{}%**)'.format(str(client_score)))
            #left_column.markdown('Seuil par défaut du modèle: **50%**')

            #if decision == '❌ Crédit Refusé':
                #left_column.markdown(
                    # 'Décision: <span style="color:red">**{}**</span>'.format(decision),\
                    # unsafe_allow_html=True)   
            # else:    
            #     left_column.markdown(
            #         'Décision: <span style="color:green">**{}**</span>'\
            #         .format(decision), \
            #         unsafe_allow_html=True)
            st.header("Local interpretability")
            show_local_feature_importance = st.checkbox(
                "Display the variables that contributed the most to the model's decision")
            if (show_local_feature_importance):
                
                shap.initjs()

                number = st.slider('Select the number of features to display', \
                                    2, 20, 5)

                X = df[df['SK_ID_CURR']==int(id_client)]
                X = X[relevant_features]


                
                #fig, ax = plt.subplots(figsize=(20, 20))
                #explainer_local = shap.Explainer(model, data_test)
                shap_values = explainer_local(X)
                #shap_values = joblib.load("shap_local")
                
                st_shap(shap.plots.waterfall(shap_values[0], max_display=number), height=600, width=600)  # Image ou visualisation centrée

            
                #explainer = shap.TreeExplainer(model)
                #shap_values = explainer.shap_values(X)
                #shap.summary_plot(shap_values[0], X, plot_type ="bar", \
                                    #max_display=number, color_bar=False, plot_size=(8, 8))
                
                
                #st.pyplot(fig)

        #-------------------------------------------------------
        # Show client information
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
            st.header('Client information')

            with st.spinner('Loading customer information...'):
                personal_info_df = client_info[list(personal_info_cols.keys())]
                #personal_info_df['SK_ID_CURR'] = client_info['SK_ID_CURR']
                personal_info_df.rename(columns=personal_info_cols, inplace=True)

                personal_info_df["AGE"] = int(round(personal_info_df["AGE"]/365*(-1)))
                personal_info_df["NB ANNEES EMPLOI"] = \
                int(round(personal_info_df["NB ANNEES EMPLOI"]/365*(-1)))


                filtered = st.multiselect("Choose the information to be displayed", \
                                          options=list(personal_info_df.columns),\
                                          default=list(default_list))
                df_info = personal_info_df[filtered] 
                df_info['SK_ID_CURR'] = client_info['SK_ID_CURR']
                df_info = df_info.set_index('SK_ID_CURR')

                st.table(df_info.astype(str).T)
                show_all_info = st\
                .checkbox("Show all information")
                if (show_all_info):
                    st.dataframe(client_info)

        #-------------------------------------------------------
        # Comparaison clients
        #-------------------------------------------------------

        if (show_client_comparison):
            st.header('Comparison with other clients') 

            with st.spinner('Chargement de la comparaison liée à la variable sélectionnée'):
                var = st.selectbox("Select a variable : ",\
                                   list(personal_info_cols.values()))
                feature = list(personal_info_cols.keys())\
                [list(personal_info_cols.values()).index(var)]    

                if (feature in numerical_features):                
                    plot_distribution(data_train, feature, client_info[feature], var)   
                elif (feature in rotate_label):
                    univariate_categorical(data_train, feature, \
                                           client_info[feature], var, False, True)
                elif (feature in horizontal_layout):
                    univariate_categorical(data_train, feature, \
                                           client_info[feature], var, False, True, True)
                else:
                    univariate_categorical(data_train, feature, client_info[feature], var)

        #-------------------------------------------------------
        # Global feature importance
        #-------------------------------------------------------

        if (shap_general):
            st.header('‍Global Feature Importance')
            st.write("Global interpretation of the LGBM model on the first 1000 customers for class 1 (loan not repaid): **layered violin plot**")
            st.image(SHAP_GENERAL1)
            st.write("Global interpretation of the LGBM model on the first 1000 customers for class 1 (loan not repaid): **bar plot**")
            st.image(SHAP_GENERAL2)

        
            

