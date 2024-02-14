import pandas as pd
from bs4 import BeautifulSoup
import requests
import re

# Function to get the link to each lycee's score in the ranking
def get_lycee_links(page:int):
    # sets the url corresponding to the page
    url = f"https://www.letudiant.fr/classements/classement-lycees.html?filters[715]=Générale&page={page}"
    text = requests.get(url).text
    soup = BeautifulSoup(text, 'lxml')
    urls_lycee = []
    # filters the right links
    for x in soup.findAll("a",class_="tw-cursor-pointer"):
        try:
            if x['href'][0:63]=="https://www.letudiant.fr/classements/classement-lycees/laureat/":
                urls_lycee.append(x['href'])
        except:
            pass
    return urls_lycee

# Function to get the scores of the lycee from the url
def get_lycee_scores(url:str):

    text = requests.get(url).text
    soup = BeautifulSoup(text, 'lxml')
    info = soup.findAll('div',class_='tw-w-full')
    pattern = re.compile(r'(\n)|( {2})')
    clean_info = [re.sub(pattern,"",x.text) for x in info]

    dic_col_id = {
        "Lycee_name":3,
        "Note de l'Etudiant":11,
        "taux de reussite au bac":13,
        "capac a faire progresser eleves":16,
        "capac a garder les eleves":19,
        "taux de mention au bac":22,
        "capac a faire briller les eleves":25
    }
    result = {x: [clean_info[dic_col_id[x]]] for x in dic_col_id.keys()}
    return result

# Function to loop through the list of lycees's url to get their information 
def get_lycee_df():
    urls = [url_lycee for page in range(1,109) for url_lycee in get_lycee_links(page)]
    result_df=[]
    for url in urls:
        if len(result_df)==0:
            result_df = pd.DataFrame.from_dict(get_lycee_scores(url=url))
        else:        
            result_df = pd.concat(
                [pd.DataFrame.from_dict(get_lycee_scores(url=url)),result_df],
                ignore_index=True)
    return result_df