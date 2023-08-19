import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# importation du fichier exemple1.xlsx
df=pd.read_excel("exemple1.xlsx",index_col=0)
print(df)

# specification du model
X=df['pib']
y=df['recette']
#ajout d'une constante
X=sm.add_constant(X)
# specification du model
model=sm.OLS(y,X)
# Estimation des parametres
results=model.fit()
# Affichage
print(results.summary())

# Validation des hypotheses
# linearite
predictions =results.predict(X) # calcule des prédictions
residuals = results.resid # calcul des résidus
plt.scatter(predictions, residuals)
plt.axhline(0, color='red')
plt.xlabel('Valeurs prédites')
plt.ylabel('Résidus')
plt.title('Résidus vs Valeurs prédites')
plt.show()