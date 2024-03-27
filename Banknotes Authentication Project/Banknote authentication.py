"""University of London Coursera Course"""
"""Banknote Authentication Project"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import KMeans

"""Part 1"""
#Step 1: Load the given Banknote authentication dataset.  
data = pd.read_csv('Banknote-authentication-dataset-.csv')
V1 = np.array(data['V1'])
V2 = np.array(data['V2'])

#Step 2: Calculate statistical measures, e.g. mean and standard deviation. 
V1_mean = np.mean(V1)
V2_mean = np.mean(V2)
V1_std = np.std(V1)
V2_std = np.std(V2)

#Step 3: Normalisation

V1_min = np.min(V1)
V2_min = np.min(V2)
V1_max = np.max(V1)
V2_max = np.max(V2)
normed_V1 = (V1 - V1_min) / (V1_max - V1_min)
normed_V2 = (V2 - V2_min) / (V2_max - V2_min)

#Step 4: Outliers
"""
ellipse = patches.Ellipse([V1_mean, V2_mean], V1_std*3, V2_std*3, alpha=0.25)
fig, graph = plt.subplots()
"""

#Step 5: Visualise your data as you consider fit. (testing)
"""
plt.xlabel('V1')
plt.ylabel('V2')
plt.scatter(V1, V2)
graph.add_patch(ellipse)
"""

#Evaluate if the given dataset is suitable for the K-Means clustering task.
#Write a short description of the dataset and your evaluation of its suitability for the K-Means clustering task.

"""
It is suitable for the K-Means clustering.

The dataset contains continuous numeric data, the correct datatype for K-Means clustering.
It is large in size (1372 instances) so there is enough training data.

There are only two features:
V1 (variance of wavelet transformed image) 
V2 (skewness of wavelet transformed image)

For V1:
mean = 0.43373525728862977
standard deviation = 2.841726405206097
For V2:
mean = 1.9223531209912539
standard deviation = 5.866907488271993

Outliers are the points outside the light blue ellipse.
I used +-3 standard deviation.
"""

"""Part 2"""
#The goal of this assignment is to use the Banknote authentication dataset 
#to train a model that can predict if a banknote is genuine or not.  

#Step 1: Run K-means on the given dataset

normed_V1_V2 = np.column_stack((normed_V1, normed_V2))
V1_V2 = np.column_stack((V1, V2))
#km_res = KMeans(n_clusters=4).fit(V1_V2)
#clusters = km_res.cluster_centers_

distortions = []
K = range(1,10)
for i in K:
    k_mean_model = KMeans(n_clusters=i).fit(normed_V1_V2)
    distortions.append(k_mean_model.inertia_)
  
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method to find the optimal k')
plt.show()


"""
From this graph, we have to select the value of k at the
“elbow” ie the point after which the distortion/inertia 
starts decreasing in a linear fashion, so the optimal number of cluster is 5 this time.
"""


for i in range(10):
    kmeans = KMeans(n_clusters=5).fit(V1_V2)
    clusters = kmeans.cluster_centers_

plt.scatter(V1, V2)
plt.scatter(clusters[:,0], clusters[:,1], s=100, alpha=0.5, c='red')
plt.show()

#For different clusters

for i in range(10):
    kmeans = KMeans(n_clusters=5)
    y_kmeans = kmeans.fit_predict(V1_V2)

plt.scatter(V1, V2, c=y_kmeans)
plt.show()

#Step 4: compare the results: is the K-means algorithm stable?
#Step 5:  describe your results.

"""
This is the graph after 10 runs of the K-Means algorithm where K = 4.
The centroids (orange dots) becomes stable after 7 to 8 runs.
The light blue ellipse is the accepting values within +-3 standard deviation form the mean of the data.
"""