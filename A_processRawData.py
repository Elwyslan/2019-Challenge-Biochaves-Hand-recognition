import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance
from pathlib import Path

if __name__ == '__main__':
    #Read hand data and write into dataframe
    columns = ['id']
    for i in range(14):
        columns.append(f'x{i}')
        columns.append(f'y{i}')
    rawData = pd.DataFrame(columns=columns)

    for file in os.listdir(Path('MaosBioChavesPontos/')):
        with open(Path(f'MaosBioChavesPontos/{file}')) as f:
            lines = f.readlines()
            rowDict = {}
            rowDict['id'] = file.split('.')[0]
            for i in range(len(lines)):
                x, y = lines[i].split(' ')
                rowDict[f'x{i}'] = float(x)
                rowDict[f'y{i}'] = float(y)
            rawData = rawData.append(rowDict, ignore_index=True)

    #Create shape signature train data (fourier descriptors)
    trainData = []
    for row in rawData.iterrows():
        #Read points
        idName = list(row)[1].values[0]
        points = list(row)[1].values[1:]
        x = []
        y = []
        for i in range(0,len(points),2):
            x.append(points[i])
            y.append(points[i+1])
        
        #Compute centroid of those points
        centroid = (sum(x)/(len(points)/2), sum(y)/(len(points)/2))
        #print(idName)
        #plt.scatter(x,y,c='k')
        #plt.scatter(centroid[0],centroid[1],c='g')
        #plt.show()
        
        #Compute Centroid Distance Function
        shapeSignature = []
        for i in range(len(points)//2):
            shapeSignature.append(distance.euclidean(centroid, (x[i], y[i])))
        #plt.plot(shapeSignature)
        #plt.show()
        
        #Compute fourier coeficientes (fourier descriptors) from Centroid Distance Function
        shapeSignature = [idName] + list(map(abs, np.fft.fft(shapeSignature)))
        #logCoefs = np.log10(shapeSignature[1:])
        #plt.bar([i for i in range(len(logCoefs))], logCoefs, width=0.1)
        #plt.show()
        
        #Store train data
        trainData.append(shapeSignature)

    #Save train data in .csv format
    columns = ['id']
    for i in range(len(trainData[0]) - 1):
        columns.append(f'coef{i}')
    trainData = pd.DataFrame(trainData, columns=columns)
    trainData.to_csv('trainData.csv')

    print('Fim!')