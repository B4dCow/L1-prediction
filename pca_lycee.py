
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipe
import numpy as np
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import os 
import pandas as pd
from config import path 

os.chdir(path)

# -----------------Import----------------
df_lycee = pd.read_csv('data/work/df_lycee_ml.csv')

# -----------------Preprocess---------------------
# encoding dummy variables
enc = OneHotEncoder(
      categories = [['LEGT','LPO'],['privé sous contrat', 'public']],
      handle_unknown='ignore',
      drop = [['LPO'],['privé sous contrat']])

df_lycee[["LEGT",'public']] = enc.fit_transform(df_lycee[['lycee_type','lycee_secteur']].to_numpy()).toarray()
col_num = df_lycee.select_dtypes(include=np.number).columns
df_lycee_num = df_lycee.loc[:,col_num]

# scale the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_lycee_num)

# performs a PCA
pca = PCA(random_state=17)
pca.fit(df_scaled)
coord = pd.DataFrame(columns=[f"PC{i+1}" for i in range(df_scaled.shape[1])],data = pca.transform(df_scaled))

# --------------Graphic representation of the results------------
# plots the scree plot
plt.plot(np.arange(1,df_scaled.shape[1]+1), pca.explained_variance_ratio_*100, 'o-')
plt.xticks(np.arange(1,df_scaled.shape[1]+1, 5))
plt.xlabel('Dimension')
plt.ylabel('Explained Variance Ratio (%)')
plt.title('PCA Explained Variance Ratio')
plt.show()

display(pca.explained_variance_ratio_*100)
display(pd.DataFrame(
    columns=col_num,
    data=pca.components_))

# df_stud = pd.read_csv("data/work/df_stud.csv")
# df_stud = df_stud.loc[:,['UAI','target']]
# print(f'{sum(~df_stud.UAI.isin(set(df_lycee.UAI)))} students which lycee is not present in the dataset')
# print(df_stud.loc[~df_stud.UAI.isin(set(df_lycee.UAI)),'target'].value_counts())
# df_stud = df_stud.loc[df_stud.UAI.isin(set(df_lycee.UAI))]
 
# correlation between each variable and the first and second PC
corvar = pd.DataFrame(data=[[np.corrcoef(df_lycee_num[c],coord[f"PC{n+1}"])[1,0] 
               for n in range(pca.n_components_)] for c in col_num],
               columns = [f"PC{n+1}"for n in range(pca.n_components_)] ,
               index = col_num)
# plot it 
fig, axes = plt.subplots(figsize=(8,8)) 
axes.set_xlim(-1,1) 
axes.set_ylim(-1,1) 
# Labels (variables' names)
for j in range(df_lycee_num.shape[1]): 
    plt.annotate(
        col_num[j],
        (corvar.loc[col_num[j],'PC1'],corvar.loc[col_num[j],'PC2'])
        ) 
# Adding axes
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1) 
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
# Adding circle
circle = plt.Circle((0,0),1,color='blue',fill=False) 
axes.add_artist(circle) 
axes.set_xlabel(f'variable correlation with PC1 ({pca.explained_variance_ratio_[0]:.0%})')
axes.set_ylabel(f'variable correlation with PC2 ({pca.explained_variance_ratio_[1]:.0%})')
axes.set_title('Correlation Circle')
plt.show()

# df_coord_stud = pd.DataFrame()
# df_coord_stud[['PC1','PC2']] = coord.loc[df_lycee.UAI.isin(set(df_stud['UAI'])),['PC1','PC2']]
# df_coord_stud = pd.concat([df_coord_stud.reset_index(drop=True),df_lycee.loc[df_lycee.UAI.isin(set(df_stud['UAI'])),'UAI'].reset_index(drop=True)],axis=1)

# df_stud = df_coord_stud.merge(df_stud,how='right',on='UAI')


# # Plot the students on the the PCA discriminate the one who
# import seaborn as sns
# sns.scatterplot(df_stud,x='PC1',y='PC2',hue='target')
# plt.show()

# df_coord = pd.DataFrame()
# df_coord[["PC1","PC2"]] = coord.loc[:,['PC1','PC2']]
# df_coord['L1mido'] = df_lycee.UAI.isin(set(df_stud['UAI']))
# sns.scatterplot(df_coord,x='PC1',y='PC2',hue='L1mido')
# plt.show()