#!/usr/bin/python3
#
# Title: Fibre Reinforced Concrete Compressive Strength prediction.
# File : main.py
#

import pandas            as pd
import matplotlib.pyplot as plt
import numpy             as np
from sklearn.preprocessing       import MinMaxScaler
from sklearn.model_selection     import train_test_split
from tensorflow.keras.optimizers import Adam

import CNN_Model
import SVM_Model
import LSTM_Model

dataset = "dataset/data.csv"

dataset_labels = ["Temperature", "Heating rate", "Maintenance", "Specimen type", 
                  "Cement","Water", "Sand", "Aggregate", "Silica fume",
                  "Fly ash", "Slag","Metakaolin", "Superplasticizer", "Steel fiber", 
                  "SF length","SF aspect ratio", "Polypropylene fiber", 
                  "PPF length","PPF aspect ratio", "Compressive strength"]


def main():
    # importing dataset
    labeled_data = pd.read_csv(dataset, names=dataset_labels)
    data = pd.read_csv(dataset)
    compression = data.iloc[ :, -1:]

    # dataset overview
    index  = 1
    row    = 3
    column = 7
    fig = plt.figure(1)
    fig.canvas.manager.set_window_title('Dataset Overview')

    for label in dataset_labels:
        ax = fig.add_subplot(row, column , index)
        ax.title.set_text(label)
        ax.hist(labeled_data[label])
        plt.subplots_adjust(hspace = 1, wspace= 0.5)
        index += 1

    plt.show()

    #  Data preprocessing
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    #  split training and testing data
    x_train, x_test, y_train, y_test = train_test_split(scaled_data ,compression ,test_size=0.2)

    #  Models
    cnn = CNN_Model.CNN(4, 19)
    cnn_model = cnn.build()
    cnn_model.compile(optimizer='adam', loss='mean_absolute_error')

    svm = SVM_Model.SVM(19)
    svm_model = svm.build()
    svm_model.compile(optimizer='adam', loss='mean_absolute_error')

    lstm = LSTM_Model.LSTM_(19)
    lstm_model = lstm.build()
    lstm_model.compile(optimizer='adam', loss='mean_absolute_error')


    #  Training and testing
    print("CNN Model")
    cnn_history = cnn_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=20)
    print("SVM Model")
    svm_history = svm_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=20)
    print("LSTM Model")
    lstm_history = lstm_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=20)
    
    
    #  Model evaluation
    plt.figure(2) 
    plt.plot(cnn_history.history['loss'], color='blue', label='CNN')
    plt.plot(svm_history.history['loss'], color='red', label='SVM')
    plt.plot(lstm_history.history['loss'], color='yellow', label='LSTM')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend(['CNN','SVM','LSTM'], loc='upper left')
    plt.show()
    

# call main function
if __name__ == "__main__":
    main()