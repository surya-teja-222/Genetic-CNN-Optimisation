import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import copy
import numpy as np

def genSubModel(to_remove_as_list, model):
    to_remove = {}
    to_remove_as_list = copy.deepcopy(to_remove_as_list)
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
        if(i == 0 or isinstance(layer, MaxPooling2D) or isinstance(layer, Flatten)):
            if(isinstance(layer, MaxPooling2D)):
                new_model.add(MaxPooling2D(name='maxpooling_cp_'+str(i)))
            elif(isinstance(layer, Flatten)):
                new_model.add(Flatten(name='flatten_cp_'+str(i)))
            else:
                new_model.add(layer)
        if i in to_remove and isinstance(layer, Conv2D):
            if (i == 2): # for 1st poss
                weights = copy.deepcopy(layer.get_weights()[0])
                bias = copy.deepcopy(layer.get_weights()[1])
                for node_index in sorted(to_remove[i], reverse=True):
                    weights = np.delete(weights, node_index, axis=3)
                    bias = np.delete(bias, node_index, axis=0)
                new_model.add(Conv2D(weights.shape[3], 3, padding='same', activation='relu', weights=[weights, bias], name='conv2d_cp_'+str(i)))
            else:
                weights = copy.deepcopy(layer.get_weights()[0])
                bias = copy.deepcopy(layer.get_weights()[1])
                for node_index in sorted(to_remove[i-2], reverse=True):
                    weights = np.delete(weights, node_index, axis=2)
                for node_index in sorted(to_remove[i], reverse=True):
                    weights = np.delete(weights, node_index, axis=3)
                    bias = np.delete(bias, node_index, axis=0)
                new_model.add(Conv2D(weights.shape[3], 3, padding='same', activation='relu', weights=[weights, bias], name='conv2d_cp_'+str(i)))
        if isinstance(layer, Dense):
            if(isinstance(new_model.layers[i-1], Flatten)):
                weights = copy.deepcopy(layer.get_weights()[0])
                sz = int(model.layers[i-3].get_config()['filters'])

                weights = copy.deepcopy(layer.get_weights()[0])
                bias = copy.deepcopy(layer.get_weights()[1])
                prev_conv2d_index = i-2
                while not isinstance(new_model.layers[prev_conv2d_index], Conv2D):
                    prev_conv2d_index -= 1
                new_weights = []
                for ii in range(len(weights),0, -sz):
                    k = weights[ii-sz:ii]
                    for node_index in sorted(to_remove[prev_conv2d_index], reverse=True):
                        k = np.delete(k, node_index, axis=0)
                    new_weights.extend(k)
                weights = np.array(new_weights)

                for node_index in sorted(to_remove[i], reverse=True):
                    weights = np.delete(weights, node_index, axis=1)
                    bias = np.delete(bias, node_index, axis=0)
                new_model.add(Dense(weights.shape[1], activation='relu', weights=[weights, bias], name='dense_cp_'+str(i)))
            else:
                weights = copy.deepcopy(layer.get_weights()[0])
                bias = copy.deepcopy(layer.get_weights()[1])
                for node_index in sorted(to_remove[i-1], reverse=True):
                    weights = np.delete(weights, node_index, axis=0)
                new_model.add(Dense(weights.shape[1], activation='sigmoid', weights=[weights, bias], name='dense_cp_'+str(i)))
    return new_model