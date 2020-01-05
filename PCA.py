import seaborn as sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import random
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import preprocessing


data = pd.read_csv("clean_data.csv")
print(data.shape)
print(data.head())

    #Features
x=data[['Extras','Trip Total', 'Fare','Tips', 'Trip Miles', 'Trip Seconds', 'Dropoff Community Area', 'Pickup Community Area', 'Tolls']]     #Features
standardized_X = preprocessing.scale(x)
    #PCA
pca = PCA()
principalComponents = pca.fit_transform(standardized_X)
print("explained variance ratio")
print(pca.explained_variance_ratio_)
var=(pca.explained_variance_)
print("explained variance ")
print(var)

objects = ('PC1', 'PC2','PC 3','PC 4','PC 5','PC 6', 'PC 7', 'PC 8', 'PC9')
y_pos = np.arange(len(objects))
plt.bar(y_pos, var, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Information')
plt.title('Explained variance')
plt.show()




pca = PCA(n_components=3)
principalComponents = pca.fit_transform(standardized_X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
finalDf = pd.concat([principalDf,data[['Company']]], axis = 1)#Company
plt.scatter(finalDf['principal component 1'], finalDf['principal component 2'], color='black')
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("PC1-PC2")
print("correlation between PC1 and PC2", finalDf['principal component 1'].corr(finalDf['principal component 2']) )
plt.show()

plt.scatter(finalDf['principal component 1'], finalDf['principal component 3'], color='black')
plt.xlabel("PC 1")
plt.ylabel("PC 3")
plt.title("PC1-PC3")
print("correlation between PC1 and PC3", finalDf['principal component 1'].corr(finalDf['principal component 3']) )
plt.show()
plt.scatter(finalDf['principal component 2'], finalDf['principal component 3'], color='black')
plt.xlabel("PC 2")
plt.ylabel("PC 3")
plt.title("PC2-PC3")
print("correlation between PC3 and PC2", finalDf['principal component 3'].corr(finalDf['principal component 2']) )
plt.show()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)


#k means clustering
ks = range(1, 11)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(principalDf.iloc[:, :2])

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

kmeans = KMeans(n_clusters=3)
kmeans.fit(principalComponents)
y_pred = kmeans.predict(principalComponents)
# plot the cluster assignments and cluster centers
plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=y_pred, cmap="plasma")
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            marker='^',
            c=[0, 1, 2],
            s=100,
            linewidth=2,
            cmap="plasma")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()
plt.scatter(principalComponents[:, 0], principalComponents[:, 2], c=y_pred, cmap="plasma")
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            marker='^',
            c=[0, 1, 2],
            s=100,
            linewidth=2,
            cmap="plasma")
plt.xlabel("PC 1")
plt.ylabel("PC 3")
plt.show()
plt.scatter(principalComponents[:, 1], principalComponents[:, 2], c=y_pred, cmap="plasma")
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            marker='^',
            c=[0, 1, 2],
            s=100,
            linewidth=2,
            cmap="plasma")
plt.xlabel("PC 2")
plt.ylabel("PC 3")
plt.show()