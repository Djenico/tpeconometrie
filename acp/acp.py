import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns








# chargement des donnees
donnee = pd.read_excel('analyse.xlsx',sheet_name=0,header=0,index_col=0)

 #Statistiques sur les variables (moyenne=mean, ecart-type, stat descriptive)
print(donnee.describe(include = "all").round(2))

 # Dimension des donnees actives(dimension des matice)
n = donnee.shape[0] # nombre de lignes
p = donnee.shape[1] # nombre de colonnes
print(f'n = {n} et p = {p}')

# Distance entre les individus
def distance(x,y):
    return np.sum((x-y)**2)
# Calcul de la distance
rowdist = pd.DataFrame(np.zeros(shape=(n,n),dtype=float),index = donnee.index,columns = donnee.index)
for i in range(n):
    for j in range(i+1,n):
        rowdist.values[i,j] = distance(donnee.values[i,:],donnee.values[j,:])
# Affichage
print(rowdist.round(2))

# Visualisation - heatmap
plt.figure(figsize = (10,8))
sns.heatmap(rowdist,xticklabels=rowdist.columns,yticklabels = rowdist.columns,cmap = sns.color_palette("Blues",12),linewidths=0.5)
plt.title('Heatmap des distances entre les individus')
plt.show()

# Matrice de correlation lineaire entre paires de variables
corr = donnee.corr(method = "pearson")
# Affichage
print(corr.round(2))
# heatmap de la matrice des correlation
plt.figure(figsize = (10,8))
sns.heatmap(corr,xticklabels=corr.columns,yticklabels = corr.columns,vmin = -1, vmax = +1, center = 0,cmap = 'RdBu',linewidths=0.5)
plt.title('Heatmap des correlations croisees entre variables')
plt.show()

# Transformation de donnees
# Centrage des donnees
def centrage(x):
    # x est un vecteur
    return (x-x.mean())
# Application
Y = donnee.transform(centrage)
# Affichage
print(Y.round(2))

# Centrage et reduction des donnees(Centree et reduite)
def StandardScaler(x):
    # x est vecteur
    return (x - x.mean())/x.std(ddof=0)
# Application
Z = donnee.transform(StandardScaler)
#Affichage
print(Z.round(2))

# Calcul de la distance euclidienne ponderee entre les individus (donnee centre reduite)
dist = pd.DataFrame(np.zeros(shape=(n,n),dtype=float),index = donnee.index,columns = donnee.index)
for i in range(n):
    for j in range(i+1,n):
        dist.values[i,j] = distance(Z.values[i,:],Z.values[j,:])
# Visualisation - heatmap
plt.figure(figsize = (10,8))
sns.heatmap(dist,xticklabels=dist.columns,yticklabels = dist.columns,cmap = sns.color_palette("Blues",12),linewidths=0.5)
plt.title('Heatmap des distances entre les individus')
plt.show()

# Information sur les individus
rowdisto = Z.apply(lambda x : np.sum((x)**2),axis=1)
rowweight = np.ones(n)/n
rowinertie = rowweight*rowdisto.values
rowinfos = pd.DataFrame(np.transpose([rowdisto,rowweight, rowinertie]),columns = ["Disto2", "Poids", "Inertie"],index = donnee.index)
print(rowinfos.round(3))

# INERTIE
# Inertie : moyenne des carres des distances
InK = (1/n**2)*rowdist.sum().sum()
print('Inertie totale : %.2f'%(InK))
# Inertie sur donnees centrees et reduites
print('Inertie totale (donnee centree reduite) : %.1f'%(Z.var(ddof=0).sum()))

#Valeur propre
# Diagonalisation de la matrice des corr´elations
eigenvalue, eigenvector = np.linalg.eig(corr)
# eigenvalue dataframe
percent = np.array([100*x/sum(eigenvalue) for x in eigenvalue])
cumpercent = np.cumsum(percent)
columns = ['valeur propre','pourcentage d\'inertie','pourcentage d\'inertie cumulee']
index = ['Dim.{}'.format(x+1) for x in range(p)]
Eigen = pd.DataFrame(np.transpose([eigenvalue,percent,cumpercent]),
index=index,columns = columns)
Eigen.index.name = 'Dimension'
# Affichage
print(Eigen.round(2))

# Visualisation des valeurs propres
def screeplot(data,choice=None,figsize=None):
    #
    p = data.shape[0]
    fig,axes = plt.subplots(figsize = figsize); axes.grid()
    axes.set_xlabel('Dimensions',fontsize=14)
    axes.set_title('Scree plot',fontsize=14)
    axes.set_xticks([x for x in range(1,p+1)])
    if choice is None or choice=='scree plot':
        eigen = data.iloc[:,0].round(2)
        ylim = np.max(eigen)+1
        axes.set_ylim(0,ylim)
        axes.bar(np.arange(1,p+1),eigen.values,width=0.9)
        axes.plot(np.arange(1,p+1),eigen.values,c="black")
        axes.set_ylabel('Eigenvalue',fontsize=13)
        ## Add text
        for i in range(p):
            axes.scatter(i+1,eigen.values[i],color='black',alpha=1)
            axes.text(i+.75,0.10+eigen.values[i],str(eigen.values[i]),color = 'black')
    elif choice == "percentage":
        percent = data.iloc[:,1].round()
        axes.set_ylim(0,100)
        axes.bar(np.arange(1,p+1),percent.values,width=0.9)
        axes.plot(np.arange(1,p+1),percent.values,c="black")
        axes.set_ylabel('Percentage of variance',fontsize=13)
        ## Add text
        for i in range(p):
            axes.scatter(i+1,percent.values[i],color='black',alpha=1)
            axes.text(i+.6,0.10+percent.values[i],f'{percent.values[i]}%',color = 'black',fontweight='bold',fontsize=12)
    elif choice == "cumulative":
        cumul = data.iloc[:,2].round()
        axes.set_ylim(0,105)
        axes.bar(np.arange(1,p+1),cumul.values,width=0.9)
        axes.set_ylabel('Cumulative percentage of variance',fontsize=13)
    plt.show()
# Affichage
screeplot(data=Eigen,figsize=(8,6))

# Representation des individus

# coordonnees factorielles des individus
rowcoord = pd.DataFrame(np.dot(Z,eigenvector),index = donnee.index,columns = index)
# Affichage
print(rowcoord.loc[:,['Dim.1','Dim.2']].T.round(2))

# Contribution absolues des individus
rowcontrib = rowcoord.apply(lambda x: 100*x**2/(n*eigenvalue),axis=1)
# Affichage
print(rowcontrib.iloc[:,[0,1]].round(2))

# Cosinus carre des individus
rowcos2 = rowcoord.apply(lambda x : x**2/rowdisto, axis=0)
#Affichage
print(rowcos2.iloc[:,[0,1]].T.round(2))

#Graphique des individus

# Affichage graphique des contributions et cosinus 2
def plot_graph(data,axis,xlabel,title,figsize=None):
    p = data.shape[1]
    try:
        if axis<0 or axis>p:
             raise ValueError(f'axis doit ^etre compris entre {0} et {p-1}.')
        else:
            sort = data.sort_values(by=f'Dim.{1+axis}', ascending=True)
            sort.iloc[:,axis].plot.barh(figsize=figsize)
            plt.xlabel(xlabel)
            plt.title(f'{title} in axis {1+axis}')
            plt.show()
    except ValueError as f:
        print(f)
# Contribution axe 1
plot_graph(data=rowcontrib,axis=0,xlabel = 'Contribution (%)',title = 'Rows contributions',figsize=(10,8))
# Contribution axe 2
plot_graph(data=rowcontrib,axis=1,xlabel = 'Contribution (%)',title = 'Rows contributions',figsize=(10,8))

# Cosinus carre axe 1
plot_graph(data=rowcos2,axis=0,xlabel = 'Cosinus 2',title = 'Rows cosinus 2',figsize=(10,8))
# Cosinus carre axe 2
plot_graph(data=rowcos2,axis=1,xlabel = 'Cosinus 2',title = 'Rows cosinus 2',figsize=(10,8))


# Fonction de visualisation en 2D
def pca_row_plot(data,eigen,axei,axej,figsize=None):
    try:
        if axei==axej:
            raise ValueError('Erreur: axei doit ^etre diff´erent de axej.')
        elif axei>axej:
            raise ValueError('Erreur: axei doit ^etre inferieur a axej.')
        elif axei<0 or axej<0:
            msg = 'Erreur: les valeurs des axes doivent etre positives ou nulles.'
            raise ValueError(msg)
        else:
            # set limite
            n = data.shape[0]
            # Valeurs propres
            percent = np.array([100*x/sum(eigen) for x in eigen])
            dimi = round(percent[axei],2); dimj = round(percent[axej],2)
            # Graphique
            fig, axes = plt.subplots(figsize = figsize); axes.grid()
            axes.axis([-8,8,-6,6])
            axes.set_title("Projection des individus")
            axes.set_xlabel(f"Dim.{1+axei} ({dimi}%)")
            axes.set_ylabel(f"Dim.{1+axej} ({dimj}%)")
            for i in range(n):
                plt.scatter(data.iloc[i,axei], data.iloc[i,axej],c = "black", alpha = 1)
                axes.text(data.iloc[i,axei],data.iloc[i,axej],data.index[i],color = "black", fontsize = 11)
            plt.axhline(0, color='blue',linestyle="--", linewidth=0.5)
            plt.axvline(0, color='blue',linestyle="--", linewidth=0.5)
            plt.show()
    # if false then raise the value error
    except ValueError as e:
            print(e)  
# Nuage des individus sur les axes 1 et 2
pca_row_plot(data=rowcoord,eigen=eigenvalue,axei=0,axej=1,figsize=(12,8))










