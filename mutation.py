import sys


sys.stdout = open('logs/results/mutation.log', 'w')
sys.stderr = open('logs/results/error/mutation-err.log', 'w')
print("INFO: STDOUT/STDERR redirected to file")

import numpy as np
import random
import json
import tensorflow as tf
from tensorflow import keras


from tensorflow.keras.layers import *
from tensorflow.keras.models import *

import fitness
print("INFO: IMPORTING DATA")

import generate_sub_model as gsm

print("INFO: BASE MODEL")
model = keras.models.load_model('models/base_model.h5')


print("INFO: ORIGINAL FITNESS " , fitness.getFitness(model, epochs=10))


def mutate(chromosomes, new_fitness):
    new_chromosomes = []
    new_chromosomes_acc = []
    for i, chromosome in enumerate(chromosomes):
        print("Mutating for {} th time".format(i+1))
        # randomly select a mutation point
        mutation_point = random.randint(0, len(chromosome)-1)
        # mutate the chromosome
        chromosome[mutation_point] = 1 - chromosome[mutation_point]
        # calculate the fitness of old and new chromosomes
        model2 = gsm.genSubModel(chromosome, model)
        fitness1 = new_fitness[i]
        fitness2 = fitness.getFitness(model2, epochs=2)
        if(fitness2 < 0.7):
            print("Breaking because of less fitness")
            continue
        # select the best chromosome
        fitnesses = [fitness1, fitness2]
        best_chromosomes = [chromosomes[i], chromosome ]
        best_chromosomes , best_fitnesses = zip(*sorted(zip(best_chromosomes, fitnesses), reverse=True))
        new_chromosomes.append(best_chromosomes[0])
        new_chromosomes_acc.append(best_fitnesses[0])
        print("Newly aded {}".format(best_fitnesses[0]))
    return new_chromosomes, new_chromosomes_acc


chromosomes = None
with open('chromosomes/newly_bred.json') as f:
    chromosomes = json.load(f)

new_chromosomes = []
acc = []

for i in chromosomes.keys():
    new_chromosomes.append(chromosomes[i][0])
    acc.append(chromosomes[i][1])


print("INFO: MUTATING CHROMOSOMES")
new_chromosomes_mutated, new_chromosomes_acc_mutated = mutate(new_chromosomes , acc)

new_chromosomes_mutated = {i: [new_chromosomes_mutated[i], new_chromosomes_acc_mutated[i]] for i in range(len(new_chromosomes_mutated))}

with open('chromosomes/new_cromosomes_mutated.json', 'w') as f:
    json.dump(new_chromosomes_mutated, f)

print("Obtained FITNESS of {}  on chromosome {}".format(max(new_chromosomes_acc_mutated) , new_chromosomes_mutated[new_chromosomes_acc_mutated.index(max(new_chromosomes_acc_mutated))]))
