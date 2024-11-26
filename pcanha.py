import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
docu = pd.DataFrame({'Classe': ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'],
                   '2020': [163896, 336243, 742493, 102341, 69043, 77002, 203054, 700546, 2798972, 2395407, 355793, 284762],
                   '2021': [167644, 172726, 363343, 60821, 51720, 57785, 196135, 739318, 1112632, 1055524, 254636, 90158],
                    '2022':[424960, 375200, 94983, 50622, 65709, 125721, 275758, 677541, 2983989, 1589985, 941088, 283994],
                     '2023':[115260, 341790, 123108, 213785, 309392, 88099, 229812, 572167, 1582618, 2280561, 1725758, 1078987]})
docu.drop('Classe',axis=1,inplace=True)
X=docu.iloc[:,0:2]
y=docu.iloc[:,2]
x_trei,x_teste,y_trei,y_teste = train_test_split(X,y,test_size=0.2,random_state=10)
padroniza = StandardScaler()
x_trei=padroniza.fit_transform(x_trei)
x_trei=pd.DataFrame(x_trei,columns=['2021', '2023'])
x_trei.head(12)
vari=PCA()
X_pca=vari.fit_transform(x_trei)
vari.explained_variance_ratio_
vari2=PCA(0.95)
X_pca2=vari2.fit_transform(x_trei)
X_pca2.shape
vari21=PCA(n_components=2)
X_pca2c=vari21.fit_transform(x_trei)
colormap=plt.cm.get_cmap('autumn')
plt.figure()
scatter=plt.scatter(X_pca2c[:,0],X_pca2c[:,1],c=y_trei,cmap=colormap)
plt.xlabel("Variância por ano")
plt.ylabel("Variância por mês")
plt.colorbar(scatter,label="Hectares queimados")
plt.show()