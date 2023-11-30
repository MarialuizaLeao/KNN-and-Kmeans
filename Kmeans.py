import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import sys

kValue = int(sys.argv[1])

# Importing the train dataset
trainingData = pd.read_csv('nba_treino.csv')