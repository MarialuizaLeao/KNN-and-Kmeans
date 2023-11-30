import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import sys

class Player():
    def __init__(self, data, classification = None) -> None:
        self.data = np.array(data)
        self.classification = classification
        
    def distance(self, other):
        return np.linalg.norm(self.data - other.data)
    
kValue = int(sys.argv[1])

# Importing the train dataset
trainingData = pd.read_csv('nba_treino.csv')
lines = trainingData.shape[0]
playersTrain = []     
for i in range(lines):
    player = Player(trainingData.iloc[i, 0:-1], classification = trainingData.iloc[i, -1])
    playersTrain.append(player)
    
# Importing the test dataset
testingData = pd.read_csv('nba_teste.csv')
lines = testingData.shape[0]
playersTest = []     
for i in range(lines):
    player = Player(testingData.iloc[i, 0:-1], classification = testingData.iloc[i, -1])
    playersTest.append(player)
real_classifications = testingData.iloc[:, -1].to_numpy()

class KNN():
    def __init__(self, trainingData, testingData, k) -> None:
        self.k = k
        self.playersTrain = trainingData
        self.playersTest = testingData
        
    def predict(self, playersTest) -> list:
        knn_classifications = []
        for player in playersTest:
            distances = np.empty(shape=(len(self.playersTrain)), dtype=tuple)
            for j in range(len(self.playersTrain)):
                distances[j] = (self.playersTrain[j].distance(player),self.playersTrain[j].classification)
            distances = sorted(distances, key=lambda x: x[0])
            sum = 0
            for i in range(self.k):
                sum += distances[i][1]
            if(sum > self.k/2): knn_classifications.append(1)
            elif(sum < self.k/2): knn_classifications.append(0)
            else: knn_classifications.append(np.random.choice([0,1]))
        return knn_classifications
    
KNN = KNN(playersTrain, playersTest, kValue)
knn_classifications = KNN.predict(playersTest)
accuracy = metrics.accuracy_score(real_classifications, knn_classifications)
matrix = metrics.confusion_matrix(real_classifications, knn_classifications)
recall = metrics.recall_score(real_classifications, knn_classifications)
precision = metrics.precision_score(real_classifications, knn_classifications)
print(f"-------------------------------------\nK value = {kValue}\n-------------------------------------")
print(f"Metrics:\nAccuracy: {accuracy}\nRecall: {recall}\nPrecision: {precision}\nConfusion Matrix:\n{matrix}")
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels = [False, True])

cm_display.plot()
plt.savefig(f"KNN_{kValue}.png")