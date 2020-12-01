import pandas as pd
import math as m
import numpy as numpy
from scipy import stats
#Importation et extraction des données relatifs à la banque star
#2018
data2018=pd.read_csv("data2018.csv", header = 0, sep=",")
starData = data2018.loc[data2018['LIB_VAL']=="STAR",]
starDataFiltered = starData[['SEANCE', 'LIB_VAL', 'CLOTURE', 'NB_TRAN']]
#2017
data2017=pd.read_csv("data2017.csv", header=0, sep=",")
starData2 = data2017.loc[data2017['VALEUR']=="STAR",]
starDataFiltered2 = starData2[['SEANCE', 'VALEUR', 'CLOTURE', 'NB_TRANSACTION']]
#2016
data2016=pd.read_csv("data2016.csv", header=0, sep=",")
starData3 = data2016.loc[data2016['VALEUR']=="STAR",]
starDataFiltered3 = starData3[['SEANCE', 'VALEUR', 'CLOTURE', 'NB_TRANSACTION']]
starDataFiltered3.to_csv("data.txt", index=False,index_label=False,mode="w")
starDataFiltered2.to_csv("data.txt", index=False,index_label=False,header=False,mode="a")
starDataFiltered.to_csv("data.txt", index=False,index_label=False,header=False,mode="a")

# Elimination des lignes qui contient NB_TRANSACTION = 0
newDF = pd.read_csv("data.txt", sep=",", header=0)
newDF = pd.DataFrame(newDF)
newDF = newDF.drop(newDF[newDF.NB_TRANSACTION.astype('int64') == 0].index)
newDF.to_csv("data.txt", index=False, index_label=False, header = True, mode="w")
newDF = pd.read_csv("data.txt", sep=",", header=0)
# add rendement
newDF.insert(4,'Rendement',0.0)
#insertion dividendes
newDF.insert(5, 'Dividende', 0.0)
newDF.loc[newDF.SEANCE == "1/6/2016", "Dividende"] = 9.200
newDF.loc[newDF.SEANCE == "1/6/2017", "Dividende"] = 2.700
newDF.loc[newDF.SEANCE == "30/05/18", "Dividende"] = 1.220
total_rows = newDF.count()[0]
for i in range(0,total_rows-1):
  if(str(newDF.SEANCE.loc[i+1])[-2:]!=str(newDF.SEANCE.loc[i])[-2:]):
    newDF.Rendement.loc[i+1] = 0
  else:
    newDF.Rendement.loc[i+1] = m.log((newDF.CLOTURE.loc[i+1]+newDF.Dividende.loc[i+1])/newDF.CLOTURE.loc[i])

# test de normalité
print(stats.shapiro(newDF.Rendement))
# resultat (0.9482401609420776, 4.966767082792596e-14)
print(stats.jarque_bera(newDF.Rendement))
# resultat (74.18243217351991, 1.1102230246251565e-16)
# moyenne
moyRendement = numpy.mean(newDF.Rendement)
print(moyRendement)
# somme 
somme = numpy.sum((newDF.Rendement-moyRendement)**2)
# sigma²
delta = 1/250
sigSqr = (1/delta)*(1/(newDF.count()[0]-1))*somme
sigma = m.sqrt(sigSqr)
print(sigma)
# mu
mu = (0.5*sigSqr)+(moyRendement*250)
print(mu)


#Methode 1 (Black & Scholes)
# rss : rendement sans risque
rss = 0.07
S_zero = newDF['CLOTURE'][newDF.count()[0]-1]
print(S_zero)
k=S_zero+10

d1 = (numpy.log(S_zero/k)+(rss+sigSqr/2))/sigma
d2 = d1 - sigma
prix1 = S_zero*stats.norm.cdf(d1)-k*numpy.exp(-rss)*stats.norm.cdf(d2)
print(prix1)


#2eme methode
# simulation de la loi uniforme
uniform=numpy.random.normal(size=1000000)
uniDF=pd.DataFrame(uniform)
# simulation de la loi normale
#uniDF = pd.DataFrame(uniform)
#moy = uniform.mean()
#sd = uniDF.values.std(ddof=1)

#uniDF.insert(1,'normale',(uniDF-moy)/(sd))
# S

uniDF.insert(1, 'SS', uniDF*sigma)
test = (rss-(sigSqr/2))+uniDF.SS
uniDF.insert(2, 'S', S_zero*(m.e ** test))
xx = []
for i in uniDF.S:
  xx.append(max(i-k, 0))

prix2 = numpy.mean(xx)*numpy.exp(-rss)
print(prix2)