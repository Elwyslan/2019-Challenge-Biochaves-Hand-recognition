import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

trainData = pd.read_csv('trainData.csv', index_col=0)
targetData = pd.read_csv('targetHand.csv', index_col=0)

def compareSpectrums(ids=['thi01', 'thi02']):
    f, ax = plt.subplots(sharey=True, sharex=True)
    spectrums = []
    for i in range(trainData.shape[0]):
        if trainData.iloc[i]['id'] in ids:
            logCoefs = np.log10(trainData.iloc[i].values[1:].tolist())
            #logCoefs = trainData.iloc[i].values[1:].tolist()
            spectrums.append((logCoefs, str(trainData.iloc[i]['id'])))
    
    ax.bar([i+0.05 for i in range(len(spectrums[0][0]))], spectrums[0][0], width=0.1, label=f'{spectrums[0][1]}.txt')
    ax.bar([i-0.05 for i in range(len(spectrums[1][0]))], spectrums[1][0], width=0.1, label=f'{spectrums[1][1]}.txt')

    targetCoefs = np.log10(targetData.iloc[0].values.tolist())
    ax.bar([i+0.15 for i in range(len(targetCoefs))], targetCoefs, width=0.1, label='marcaMao.jpg')

    ax.legend()
    f.tight_layout()
    plt.show()


if __name__ == '__main__':
    ids = [['thi01', 'thi02'],
           ['jan01', 'jan02'],
           ['gab01', 'gab02'],
           ['isr01', 'isr02'],
           ['jon01', 'jon02'],
           ['jug01', 'jug02'],
           ['ita01', 'ita02'],
           ['vit01', 'vit02'],
           ['mic01', 'mic02']
    ]

    for id_ in ids:
        compareSpectrums(id_)
        
    print('Fim!')
