import os
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from keras.models import load_model
from C_trainClassifiers import LABELS
from scipy.spatial import distance
from collections import Counter
from pathlib import Path

if __name__ == '__main__':
    #Get Model's Path
    modelsNeuralNet = []
    for model in os.listdir(Path('Models-NeuralNets/')):
        acc = float(model.split('_')[0].split('-')[1])
        if acc>=0.9:
            modelsNeuralNet.append(Path(f'Models-NeuralNets/{model}'))
    #Get Target Signature
    trainData = pd.read_csv('trainData.csv', index_col=0)
    targetData = pd.read_csv('targetHand.csv', index_col=0)
    x_train = trainData.drop(['id'], axis=1)
    for col in x_train.columns:
        targetData[col] = (targetData[col] - x_train[col].mean()) / x_train[col].std() #mean=0, std=1
    shapeSignature = targetData.values
    
    #Predict class - Neural Network
    majorityVoting_NN = []

    for m in modelsNeuralNet:
        model = load_model(str(m))
        pred = model.predict(shapeSignature)
        targetName = list(LABELS.keys())[np.argmax(pred[0])].split('-')[0]
        majorityVoting_NN.append(targetName)

    print(f'N.of voters: {len(modelsNeuralNet)}')
    print(f'Neural Network Majority Voting: {Counter(majorityVoting_NN)}')
    
    print('Fim!')