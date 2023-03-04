import tensorflow as tf
from tensorflow import keras


from test_train_data import x_train, y_train, x_test, y_test

def getFitness(model, epochs=2 , verbose=0):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, verbose=verbose,validation_data=(x_test, y_test))
    _, test_acc = model.evaluate(x_test, y_test , verbose=verbose)
    return test_acc