import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import sys
import scipy
import matplotlib.pyplot as plt

kValue = int(sys.argv[1])

# Importing the train dataset
trainingData = pd.read_csv('nba_treino.csv')
testingData = pd.read_csv('nba_teste.csv')
 
np.random.seed(42)

class KMeans():
    def __init__(self, trainingData, testingData, k) -> None:
        self.k = k
        self.trainingData = trainingData
        self.testingData = testingData
        self.clusters = dict()
        self.clusterPerInstance = np.empty(shape=(self.trainingData.shape[0]), dtype=int)
        self.converged = False
        
    def distance(self, element, cluster):
        return np.linalg.norm(element - cluster)

    def assign_centroids(self) -> dict:
        for idx in range(self.k):
            randomIndex = np.random.randint(0, self.trainingData.shape[0])
            cluster = {
                'center' : self.trainingData.iloc[randomIndex, 0:-1],
                'elements' : []
            }
            self.clusters[idx] = cluster
        
    def assign_clusters(self):
        updatedClusterPerInstance = np.empty(shape=(self.trainingData.shape[0]), dtype=int)
        for idx in range(self.trainingData.shape[0]): # For each player
            dist = []
            for i in range(self.k): # For each cluster
                dis = self.distance(self.trainingData.iloc[idx, 0:-1], self.clusters[i]['center'])
                dist.append(dis)
            curr_cluster = np.argmin(dist) # Get the index of the cluster with the minimum distance
            self.clusters[curr_cluster]['elements'].append(self.trainingData.iloc[idx, 0:-1]) # Add the player to the cluster
            updatedClusterPerInstance[idx] = curr_cluster
        if np.array_equal(self.clusterPerInstance, updatedClusterPerInstance):
            self.converged = True
        else:
            self.clusterPerInstance = updatedClusterPerInstance

    def update_clusters(self):
        for i in range(self.k):
            elements = np.array(self.clusters[i]['elements'])
            if elements.shape[0] > 0:
                new_center = elements.mean(axis = 0)
                self.clusters[i]['center'] = new_center
                self.clusters[i]['elements'] = []
                
    def pred_cluster(self):
        pred = []
        for i in range(self.testingData.shape[0]): # For each player
            dist = []
            for j in range(self.k): # For each cluster
                dist.append(self.distance(self.testingData.iloc[i, 0:-1], self.clusters[j]['center']))
            pred.append((np.argmin(dist), self.testingData.iloc[i, 0:-1]))
        return pred
    
    def fit(self):
        self.assign_centroids()
        self.assign_clusters()
        while not self.converged:
            self.update_clusters()
            self.assign_clusters()
        pred = self.pred_cluster()
        return self.clusters, pred
    
KMeans = KMeans(trainingData, testingData, kValue)
clusters, pred = KMeans.fit()
for i in range(kValue):
    zero = 0
    one = 0
    for j in range(len(clusters[i]['elements'])):
        indx = int(clusters[i]['elements'][j].name)
        if trainingData.iloc[indx, -1] == 0:
            zero += 1
        else:
            one += 1
    print(f'Cluster {i}: {zero} zeros and {one} ones')
        
    

                
    
