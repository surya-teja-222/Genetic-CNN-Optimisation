import tensorflow as tf
from tensorflow import keras


def getFitness(model, x_train, y_train, x_test, y_test,epochs=2 , verbose=0):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=256, verbose=verbose)
    _, test_acc = model.evaluate(x_test, y_test , verbose=verbose)
    return test_acc