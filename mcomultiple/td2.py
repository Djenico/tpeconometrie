import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
import sympy

# importation du fichier exemple2.xlsx
df=pd.read_excel("exemple2.xlsx",index_col=0)
print(df)


# specification du model
X=df[['pib', 'avoir']]
y=df['importation']
#ajout d'une constante
X=sm.add_constant(X)
# specification du model
model =sm.OLS(y, X)

# Estimation des parametres
results=model.fit()
# Affichage des résultats
print(results.summary())

# validation des hypotheses


# independance des erreurs (voir Durbin watson proche de 2)

# test homoscedascite
# Calculer le test de Breusch-Pagan
bp_test = het_breuschpagan(results.resid, results.model.exog)
# Imprimer les résultats
labels = ['Statistique de test de Lagrange multiplier', 'p-valeur de LM','Statistique de test à base de F', 'p-valeur de F']
print(dict(zip(labels, bp_test)))
# absence de multicolinearite
# Calcul du VIF
VIF = pd.DataFrame()
VIF["Variable"] = X.columns
VIF["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(VIF)

# Prédiction avec le modèle
df['pred'] = results.predict(X)
# Comparaison des valeurs prédites avec les valeurs réelles
plt.scatter(df['pred'], y, color='red', label='Valeurs prédites')
plt.scatter(df['pred'],df['importation'], color='blue', label='Valeurs réelles')
plt.xlabel('Valeurs prédites')
plt.ylabel('Valeurs réelles')
plt.title('Valeurs prédites vs Valeurs réelles')
plt.legend()
plt.show()
