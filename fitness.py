import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from test_train_data import x_train, y_train, x_test, y_test

#PRSET WEIGHTS

w1 = 0.6
w2 = 0.2
w3 = 0.2

def getFitness(model, epochs=2 , verbose=0):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'Precision'])
    model.fit(x_train, y_train, epochs=epochs, verbose=verbose,validation_data=(x_test, y_test))
    
    # find the accuracy, precision and recall
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5)
    test_acc = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)

    return w1*test_acc + w2*test_prec + w3*test_recall


def getMFLOP(model, epochs=10, verbose=0):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, verbose=verbose,validation_data=(x_test, y_test))
    # get the number of parameters
    num_params = model.count_params()
    # get the number of flops
    num_flops = model.count_params() * x_train.shape[0] * epochs
    # get the number of MFLOPS
    num_mflops = num_flops / 1e6
    return num_mflops