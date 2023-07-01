import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
np.random.seed(0)
x=np.linspace(0,100,100)
y=3*x+np.random.normal(0,10,100)
data=pd.DataFrame(data={'x':x,'y':y})
print(data.head())
# specification du model
X=data['x']
y=data['y']
#ajout d'une constante
X=sm.add_constant(X)
# specification du model
model=sm.OLS(y,X)
# Estimation des parametres
results=model.fit()
# Affichage
print(results.summary())
predictions = results.predict(X) # calcule des prédictions
residuals = results
.resid # calcul des résidus
plt.scatter(predictions, residuals)
plt.axhline(0, 