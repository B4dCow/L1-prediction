import pandas as pd
import os
import re
import numpy as np
from thefuzz import process
from thefuzz import fuzz
from config import path 

os.chdir(path)

# -----------Import---------------
df_ips = pd.read_csv("data/input/fr-en-ips_lycees.csv", sep=';', decimal=',')
df_etudiant = pd.read_csv("data/work/letudiant_data.csv",sep=",")


#--------Preprocess------------
# of the IPS Dataframe:

# Filter out past years and Lycees professionnels, and drop specified columns
df_ips = df_ips[(df_ips['Rentrée scolaire'] == "2021-2022") & 
                (df_ips['Type de lycée'].isin(["LPO", "LEGT"]))]

df_ips.drop(columns=["IPS voie PRO", "IPS Ensemble GT-PRO", "Ecart-type de l'IPS voie PRO"], inplace=True)
# gets rid of information in parenthis
df_ips.loc[:,"Nom de l'établissment"] = df_ips.loc[:,"Nom de l'établissment"].str.replace(
    "(GENERAL PRIVE )|(GENERAL ET TECHNOLOGIQUE )|(PRIVE )|(TECHNOLOGIQUE )|(POLYVALENT)","",
    regex=True)

# of the L'Etudiant DataFrame
# Save the full info
df_etudiant['Lycee_name_brut'] = df_etudiant.Lycee_name

#  replace every " - " by "-" but the last one so we have as following
# ex: SAINT - JOSEPH - CAEN -> SAINT-JOSEPH - CAEN  
while sum(df_etudiant.Lycee_name.str.count(" - ") > 1)>0:
    df_etudiant.Lycee_name = np.where(
        df_etudiant.Lycee_name.str.count(" - ") > 1 ,
        df_etudiant.Lycee_name.str.replace(" - ","-",n=1),
        df_etudiant.Lycee_name)

# split to get the city 
df_etudiant[["Lycee_name",'Commune']] = df_etudiant.Lycee_name.str.split(" - ",expand=True,n=1)
df_etudiant.Commune = df_etudiant.Commune.str.upper()

# retrieve the lycee's type from the information in parenthis
df_etudiant['type_lycee'] = np.where(
    df_etudiant.Lycee_name.str.contains("(General Et Techno.)",regex=False),
    "LPO",
    "LEGT")

df_etudiant.Lycee_name = df_etudiant.Lycee_name.str.replace("(General Et Techno.)","",regex=False).str.strip().str.upper()

# uniformize the lycee's names and city so it is the same in the IPS and letudiant data
df_etudiant['Commune'] = df_etudiant['Commune'].str.replace("((^| )STE )"," SAINTE ",regex=True).str.replace("((^| )ST )"," SAINT ",regex=True).str.replace("( +)"," ",regex=True).str.strip()
df_etudiant.Lycee_name = df_etudiant.Lycee_name.str.replace("((^| )STE )"," SAINTE ",regex=True).str.replace("((^| )ST )"," SAINT ",regex=True).str.strip()

# ---------------Matching--------------

# get the best matches between communes
def extractOne_departage(commune,commune_list):
    # if there is an equality between the at least two names chooses the one with the closest number of letters
    result = process.extract(commune,commune_list,scorer=fuzz.partial_ratio)
    if len(result)>1:
        # if at least two potential matches have the same similarity score
        if result[0][1] == result[1][1]:
            # calculate the absolute differences of the character length
            len_diffs = [abs(len(commune)-len(result[i][0])) for i in range(len(result)) if result[i][1] == result[0][1]]
            # outputs the one minimizing character length
            return result[np.argmin(len_diffs)][0]
        else:
            return result[0][0]
    else:
        return result[0][0]
            
commune_corrigee = {x : extractOne_departage(x,set(df_ips["Nom de la commune"])) for x in set(df_etudiant.loc[:,"Commune"])}

 # necessary hard code modification
commune_corrigee["SAINT OUEN"] = "SAINT OUEN SUR SEINE"

# get the best match between lycees in the same commune
list_match = []
list_name_etud = []
list_commune_etud = []
list_commune_ips = []

# for each commune
for commune in set(df_etudiant["Commune"]):
    # gets the name of the lycees in this commune in both datasets
    name_etud = df_etudiant.loc[df_etudiant["Commune"] == commune,"Lycee_name"]
    name_ips = df_ips.loc[df_ips["Nom de la commune"] == commune,"Nom de l'établissment"]
    
    # if the commune names are not the same
    if len(name_ips)==0:        
        # use the matched communes
        name_ips = df_ips.loc[df_ips["Nom de la commune"] == commune_corrigee[commune],"Nom de l'établissment"]
    else:
        # use the commune name from 
        commune_corrigee[commune] = commune
    list_match.extend((name_etud.apply(
    lambda x: process.extractOne(x,name_ips, scorer=fuzz.partial_ratio)[0]
    )).tolist())
    list_commune_etud.extend([commune]*len(name_etud))
    list_commune_ips.extend([commune_corrigee[commune]]*len(name_etud))
    list_name_etud.extend(name_etud.tolist())

# compute the similarity score
similarity_score = [fuzz.partial_ratio(list_name_etud[i],list_match[i]) for i in range(len(list_match))]

# aggregate the result
df_match = pd.DataFrame.from_dict({
    "etud_name":list_name_etud,
    "match_name":list_match,
    "similarity_score":similarity_score,
    "commune_etud":list_commune_etud,
    "commune_ips":list_commune_ips
    })

# selects only the best ones
df_match_select = df_match[df_match.similarity_score > 70]


df_match_select2 = df_match_select.merge(
    df_ips,
    left_on=["match_name","commune_ips"],
    right_on=["Nom de l'établissment","Nom de la commune"],
    how='left').merge(
    df_etudiant,
    left_on=["etud_name","commune_etud"],
    right_on=["Lycee_name","Commune"],
    how='left'
)

# get rid of the duplicates (keep the most similar match)
df_match_select2 = df_match_select2.sort_values('similarity_score')
df_match_select2 = df_match_select2.drop_duplicates('UAI',keep='last')

# export the full dataset
df_match_select2.to_csv('data/work/df_lycee_matched.csv',index=False)

# only keeps the relevant columns
col_keep = ["UAI",'Type de lycée', 'IPS voie GT', "Ecart-type de l'IPS voie GT",
        "Note de l'Etudiant", 'taux de reussite au bac',
       'capac a faire progresser eleves', 'capac a garder les eleves',
       'taux de mention au bac', 'capac a faire briller les eleves',"Secteur"]

df_match_select2 = df_match_select2.loc[:,df_match_select2.columns.isin(col_keep)]

# renames it to more standard names
col_rename = {"Secteur":"lycee_secteur",'Type de lycée': "lycee_type", 'IPS voie GT': 'lycee_ips_mean', 
              "Ecart-type de l'IPS voie GT":"lycee_ips_std", "Note de l'Etudiant": "lycee_etud_score", 
              'taux de reussite au bac': "lycee_txbac",'capac a faire progresser eleves': "lycee_etud_progress", 'capac a garder les eleves': "lycee_etud_attract",
       'taux de mention au bac': "lycee_txbac_mention", 'capac a faire briller les eleves': "lycee_etud_glow"}

df_match_select2.rename(columns=col_rename, inplace = True)

# ------ Export --------
df_match_select2.to_csv('data/work/df_lycee_ml.csv',index=False)

