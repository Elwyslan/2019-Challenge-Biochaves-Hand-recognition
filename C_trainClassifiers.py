import os
import numpy as np
import pandas as pd
from keras.layers import Dropout, Dense, Activation
from keras import optimizers
from keras.models import Sequential
from pathlib import Path

#One-hot encoded labels
LABELS = {'thi-xx':[1,0,0,0,0,0,0,0,0],
          'jan-xx':[0,1,0,0,0,0,0,0,0],
          'gab-xx':[0,0,1,0,0,0,0,0,0],
          'isr-xx':[0,0,0,1,0,0,0,0,0],
          'jon-xx':[0,0,0,0,1,0,0,0,0],
          'jug-xx':[0,0,0,0,0,1,0,0,0],
          'ita-xx':[0,0,0,0,0,0,1,0,0],
          'vit-xx':[0,0,0,0,0,0,0,1,0],
          'mic-xx':[0,0,0,0,0,0,0,0,1]
}


if __name__ == '__main__':
    trainData = pd.read_csv('trainData.csv', index_col=0)
    
    #Pre-process data
    y_train = list(trainData['id'].values)
    for i in range(len(y_train)):
        y_train[i] = LABELS[y_train[i][0:3] + '-xx']
    y_train = np.array(y_train)
    x_train = trainData.drop(['id'], axis=1)
    for col in x_train.columns:
        x_train[col] = (x_train[col] - x_train[col].mean()) / x_train[col].std() #mean=0, std=1
    x_train = x_train.values
    
    #Train models (neural nets)
    for learningRate in [0.001, 0.0001]:
        for activationFunction in ['tanh', 'relu', 'sigmoid']:
            for nH in [2,3,4]:
                for nNeu in [16, 32, 64, 128]:
                    model = Sequential()
                    for ly in range(nH):
                        if ly==0:
                            model.add(Dense(nNeu, input_shape=(x_train.shape[1],)))
                        else:
                            model.add(Dense(nNeu))
                        model.add(Activation(activationFunction))
                        model.add(Dropout(0.3))
                    model.add(Dense(len(LABELS), activation='softmax'))
                    opt = optimizers.Adam(lr=learningRate, decay=0.0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
                    model.compile(loss='categorical_crossentropy',
                                  optimizer=opt,
                                  metrics=['accuracy'])
                    print(model.summary())
                    ret = model.fit(x_train, y_train, epochs=1500,shuffle=True)
                    acc = np.round(ret.history['acc'][-1], decimals=4)
                    modelName = Path(f'Models-NeuralNets/ACC-{acc:.4f}_HL-{nH}_NEU-{nNeu}_Acti-{activationFunction}_LR-{learningRate}.h5')
                    model.save(str(modelName))

    print('Fim!')