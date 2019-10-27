import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance
from pathlib import Path

#Hand Points collected manually from "targetHand/marcaMao.jpg"
POINTS = [(235, 780),
          (295, 480),
          (225, 170),
          (375, 415),
          (440, 50),
          (490, 400),
          (640, 60),
          (580, 410),
          (810, 150),
          (660, 460),
          (660, 636),
          (925, 575),
          (725, 760),
          (555, 915)]

if __name__ == '__main__':
    #Read e clean the target image
    targetHand = cv2.imread(str(Path('targetHand/marcaMao.jpg')))
    targetHand = cv2.cvtColor(targetHand, cv2.COLOR_BGR2GRAY)
    targetHand = targetHand[550:1550,550:1550]
    ret,thresh1 = cv2.threshold(targetHand, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    thresh1 = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2RGB)

    #Draw Points key-points
    for point in POINTS:
        cv2.circle(thresh1, point, 10, (255,0,0), -1)
    plt.imshow(thresh1)
    plt.show()

    #Create shape signature from key-points points (fourier descriptors)
    x = []
    y = []
    for point in POINTS:
        x.append(point[0])
        y.append(point[1])
    centroid = (sum(x)/(len(POINTS)), sum(y)/(len(POINTS)))
    #Compute Centroid Distance Function
    shapeSignature = []
    for i in range(len(POINTS)):
        shapeSignature.append(distance.euclidean(centroid, (x[i], y[i])))
    #plt.plot(shapeSignature)
    #plt.show()
    shapeSignature = list(map(abs, np.fft.fft(shapeSignature)))
    #logCoefs = np.log10(shapeSignature)
    #plt.bar([i for i in range(len(logCoefs))], logCoefs, width=0.1)
    #plt.show()
    
    #Save target data
    rowDict = {}
    for i in range(len(shapeSignature)):
        rowDict[f'coef{i}'] = shapeSignature[i]
    targetDf = pd.DataFrame(columns=list(rowDict.keys()))
    targetDf = targetDf.append(rowDict, ignore_index=True)
    targetDf.to_csv('targetHand.csv')

    print('Fim!')