"""
Created on Tue Feb 28, 2023

@author: Manav.Shah
"""

import pandas as pd
import numpy as np
import missingno
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# Data Extraction
data = pd.read_csv('etfs.csv',index_col="Date")
data.index = pd.to_datetime(data.index)

# Check for missing values
missingno.matrix(data)
# print('Data Shape before cleaning =', data.shape)
missing_percentage = data.isnull().mean().sort_values(ascending=False)
missing_percentage.head(10)
dropped_list = sorted(list(missing_percentage[missing_percentage > 0.2].index))
data.drop(labels=dropped_list, axis=1, inplace=True)
# print('Data Shape after cleaning =', data.shape)
data = data.fillna(method='ffill')

# Create DF for mean returns and volatility
returns = data.pct_change().mean()*266
returns = pd.DataFrame(returns) #Calculate returns and create a data frame
returns.columns = ['returns']
#Calculate the volatility
returns['volatility'] = data.pct_change().std()*np.sqrt(266)
data = returns


#Prepare the scaler
# scale = StandardScaler().fit(data)
scale = MinMaxScaler(feature_range=(-1,1))
#Fit the scaler
scaled_data = pd.DataFrame(scale.fit_transform(data),columns = data.columns, index = data.index)
X = scaled_data

# K-Means Clustering
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

# Find optimal number of clusters using elbow method
K = range(1,15)
distortions = []

#Fit the method
for k in K:
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)
    # inertia[k] = kmeanModel.inertia_
    
#Plot the results
fig = plt.figure(figsize= (15,5))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('Elbow Method')
plt.grid(True)
# plt.show()

from kneed import KneeLocator
kl = KneeLocator(K, distortions, curve="convex", direction="decreasing")
print(kl.elbow)

# Find optimal number of clusters using silhouette method
from sklearn.metrics import silhouette_score

K = range(2,15)
silhouettes = []

#Fit the method
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, init='random')
    kmeans.fit(X)
    silhouettes.append(silhouette_score(X, kmeans.labels_))

#Plot the results
fig = plt.figure(figsize= (15,5))
plt.plot(K, silhouettes, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette score')
plt.title('Silhouette Method')
plt.grid(True)
# plt.show()

kl = KneeLocator(K, silhouettes, curve="convex", direction="decreasing")
print('Suggested number of clusters: ', kl.elbow)

# Finally using kl.elbow number of clusters
c = kl.elbow
#Fit the model
k_means = KMeans(n_clusters=c)
k_means.fit(X)
prediction = k_means.predict(X)

#Plot the results
centroids = k_means.cluster_centers_
fig = plt.figure(figsize = (18,10))
ax = fig.add_subplot(111)
scatter = ax.scatter(X.iloc[:,0],X.iloc[:,1], c=k_means.labels_, cmap="rainbow", label = X.index)
ax.set_title('k-Means Cluster Analysis Results')
ax.set_xlabel('Mean Return')
ax.set_ylabel('Volatility')
plt.colorbar(scatter)
plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=10)
# plt.show()

# We see that with K-means 4 clusters are optimal and we get ETFs distributed in clusters as shown above



# Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc

#Fit the model
clusters = 3
hc = AgglomerativeClustering(n_clusters= clusters, affinity='euclidean', linkage='ward')
hc_labels_ = hc.fit_predict(X)

#Plot the results
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
scatter = ax.scatter(X.iloc[:,0], X.iloc[:,1], c=hc_labels_, cmap='rainbow')
ax.set_title('Hierarchical Clustering Results')
ax.set_xlabel('Mean Return')
ax.set_ylabel('Volatility')
plt.colorbar(scatter)
# plt.show()

# We see that with the Hierarchical method, 3 clusters are optimal and we get ETFs distributed in clusters as shown above


# Affinity Propagation Clustering
from sklearn.cluster import AffinityPropagation

#Fit the model
ap = AffinityPropagation()
ap.fit(X)
labels1 = ap.predict(X)

#Plot the results
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
scatter = ax.scatter(X.iloc[:,0], X.iloc[:,1], c=labels1, cmap='rainbow')
ax.set_title('Affinity Propagation Clustering Results')
ax.set_xlabel('Mean Return')
ax.set_ylabel('Volatility')
plt.colorbar(scatter)
# plt.show()

from itertools import cycle

cci = ap.cluster_centers_indices_
labels2 = ap.labels_

clusters = len(cci)
print('The number of clusters is:',clusters)

#Plot the results
X_ap = np.asarray(X)
plt.close('all')
plt.figure(1)
plt.clf
fig=plt.figure(figsize=(15,10))
colors = cycle('cmykrgbcmykrgbcmykrgbcmykrgb')
for k, col in zip(range(clusters),colors):
    cluster_members = labels2 == k
    cluster_center = X_ap[cci[k]]
    plt.plot(X_ap[cluster_members, 0], X_ap[cluster_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=12)
    for x in X_ap[cluster_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

# plt.show()

# We see that with the Affinity Propagation method, 11 clusters are optimal and we get ETFs distributed in clusters as shown above

# Compare Silhouette Scores of 3 Clustering Methods
print("k-Means Clustering", metrics.silhouette_score(X, k_means.labels_, metric='euclidean'))
print("Hierarchical Clustering", metrics.silhouette_score(X, hc.labels_, metric='euclidean'))
print("Affinity Propagation Clustering", metrics.silhouette_score(X, ap.labels_, metric='euclidean'))

# k-Means Clustering 0.5966602344782646
# Hierarchical Clustering 0.42893806971845977
# Affinity Propagation Clustering 0.47953238822731825


# We create a class to create opportunistic pairs from clustering results of all three algos

from itertools import combinations

class MakePairs():

    def __init__(self,labels,X):
        self.labels = labels
        self.X = X
        self.X['cluster'] = self.labels
        # self.data = data
        self._make_pairs()

    def _find_useful_clusters(self):

        self.useful_clusters = list(self.X.groupby(by='cluster', ).count().sort_values('returns',ascending=False).head(3).index)

    def _remove_highly_correlated_pairs(self):

        remove_cluster = []        

        for x in self.pairs:

            remove_cluster_list = []

            # for y in x:
            #     if abs(self.X.loc[y[0],'returns']-self.X.loc[y[1],'returns'])<0.01 and abs(self.X.loc[y[0],'volatility']-self.X.loc[y[1],'volatility'])<0.001:
            #         remove_cluster_list.append(y)

            for y in x:
                if (self.X.loc[y[0],'volatility'])/self.X.loc[y[0],'returns']<0.5 or (self.X.loc[y[1],'volatility'])/self.X.loc[y[1],'returns']<0.5:
                    remove_cluster_list.append(y)
            
            remove_cluster.append(remove_cluster_list)

        for i in range(2):
            for x in remove_cluster[i]:
                self.pairs[i].remove(x)

    def _make_pairs(self):

        self._find_useful_clusters()
        
        self.X_useful = self.X[self.X['cluster'].isin(self.useful_clusters)]

        self.clustered_etfs = []

        for x in self.X_useful['cluster'].unique():

            self.clustered_etfs.append(list(self.X_useful[self.X_useful['cluster'] == x].index))

        self.pairs = []
        self.pairs += [list(combinations(x,2)) for x in self.clustered_etfs]

        self._remove_highly_correlated_pairs()

# We get list of ETF pairs from k-means cluster labels
k_means_pairs = MakePairs(k_means.labels_,X)
# print(k_means_pairs.pairs_

# We get list of ETF pairs from hierarchical cluster labels
hc_pairs = MakePairs(hc.labels_,X)
# print(hc_pairs.pairs)

# We get list of ETF pairs from affinity propagation cluster labels
ap_pairs = MakePairs(ap.labels_,X)
# print(ap_pairs.pairs)

# we write the list of pairs for each method to a json file
import json
# Data to be written
cluster_models = {'k-means':k_means_pairs.pairs,'hierarchical':hc_pairs.pairs,'affinity-propagation':ap_pairs.pairs}
# Serializing json
json_object = json.dumps(cluster_models, indent=4)
# Writing to sample.json
with open("cluster_models.json", "w") as outfile:
    outfile.write(json_object)


