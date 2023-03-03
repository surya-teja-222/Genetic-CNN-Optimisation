import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import copy

def genSubModel(to_remove_as_list, model):
    to_remove = {}
    for i, layer in enumerate(model.layers):
        if(i != 0 and i != len(model.layers) - 1):
            if(isinstance(layer, Conv2D)):
                ls = to_remove_as_list[:int(layer.filters)]
                to_remove_as_list = to_remove_as_list[int(layer.filters):]
                zero_idx = [i for i, x in enumerate(ls) if x == 0]
                to_remove[i] = zero_idx
            elif (isinstance(layer, Dense)):
                ls = to_remove_as_list[:int(layer.units)]
                to_remove_as_list = to_remove_as_list[int(layer.units):]
                zero_idx = [i for i, x in enumerate(ls) if x == 0]
                to_remove[i] = zero_idx
    new_model = Sequential()
    for i, layer in enumerate(model.layers):
        # print(i, layer.get_weights()[0].shape if len(layer.get_weights()) > 0 else "No weights")
        if(i == 0 or isinstance(layer, MaxPooling2D) or isinstance(layer, Flatten)):
            if(isinstance(layer, MaxPooling2D)):
                new_model.add(MaxPooling2D())
            elif(isinstance(layer, Flatten)):
                new_model.add(Flatten())
            else:
                new_model.add(layer)
        if i in to_remove and isinstance(layer, Conv2D):
            if (i == 2): # for 1st poss
                weights = copy.deepcopy(layer.get_weights()[0])
                bias = copy.deepcopy(layer.get_weights()[1])
                for node_index in sorted(to_remove[i], reverse=True):
                    weights = np.delete(weights, node_index, axis=3)
                    bias = np.delete(bias, node_index, axis=0)
                new_model.add(Conv2D(weights.shape[3], 3, padding='same', activation='relu', weights=[weights, bias]))
            else:
                weights = copy.deepcopy(layer.get_weights()[0])
                bias = copy.deepcopy(layer.get_weights()[1])
                for node_index in sorted(to_remove[2], reverse=True):
                    weights = np.delete(weights, node_index, axis=2)
                for node_index in sorted(to_remove[i], reverse=True):
                    weights = np.delete(weights, node_index, axis=3)
                    bias = np.delete(bias, node_index, axis=0)
                new_model.add(Conv2D(weights.shape[3], 3, padding='same', activation='relu', weights=[weights, bias]))
        if isinstance(layer, Dense):
            if(isinstance(new_model.layers[i-1], Flatten)):
                weights = copy.deepcopy(layer.get_weights()[0])
                bias = copy.deepcopy(layer.get_weights()[1])
                prev_conv2d_index = i-2
                while not isinstance(new_model.layers[prev_conv2d_index], Conv2D):
                    prev_conv2d_index -= 1
                _,_,sz, _ =new_model.layers[prev_conv2d_index].get_weights()[0].shape
                new_weights = []
                for i in range(len(weights),0, -int(weights.shape[1]**0.5)):
                    k = weights[i-int(weights.shape[1]**0.5):i]
                    for node_index in sorted(to_remove[prev_conv2d_index], reverse=True):
                        k = np.delete(k, node_index, axis=0)
                    new_weights.extend(k)
                weights = np.array(new_weights)
                print(weights.shape)
                for node_index in sorted(to_remove[i-1], reverse=True):
                    weights = np.delete(weights, node_index, axis=1)
                    bias = np.delete(bias, node_index, axis=0)
                new_model.add(Dense(weights.shape[1], activation='relu', weights=[weights, bias]))
            else:
                weights = copy.deepcopy(layer.get_weights()[0])
                bias = copy.deepcopy(layer.get_weights()[1])
                for node_index in sorted(to_remove[i-1], reverse=True):
                    weights = np.delete(weights, node_index, axis=0)
                new_model.add(Dense(weights.shape[1], activation='sigmoid', weights=[weights, bias]))

    return new_model