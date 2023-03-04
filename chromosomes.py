import numpy as np
import json
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import *
from tensorflow.keras.models import *

# generate 100 chromosomes
def gen_chromosomes(model, count):
    chromosomes = []
    for _ in range(count):
        chromosome = []
        for i,layer in enumerate(model.layers):
            if isinstance(layer, keras.layers.Conv2D) and i != 0:
                chromosome.extend([int(np.random.choice([0,1], p=[0.3,0.7])) for _ in range(layer.filters)])
            elif isinstance(layer, keras.layers.Dense) and i != len(model.layers)-1:
                chromosome.extend([int(np.random.choice([0,1], p=[0.3,0.7])) for _ in range(layer.units)])
        chromosomes.append(chromosome)
    # save chromosomes to json file
    with open('chromosomes/initial_chromosomes.json', 'w') as f:
        json.dump(chromosomes, f)
    return chromosomes

