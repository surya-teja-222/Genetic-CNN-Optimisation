import sys

sys.stdout = open('logs/results/app.log', 'w')
sys.stderr = open('logs/results/error/app-err.log', 'w')
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

print("INFO: CREATING CHROMOSOMES")
import chromosomes as ch

chromosomes = ch.gen_chromosomes(model, 100)

print("INFO: BREEDING CHROMOSOMES")
def breed(chromosomes):
    new_chromosomes = []
    new_fitness = []
    for i in range(40):
        print("Breeding for {}th time".format(i+1))
        # randomly select 2 chromosomes
        c1 = random.choice(chromosomes)
        c2 = random.choice(chromosomes)
        # randomly select a crossover point
        crossover_point = random.randint(0, len(c1))
        # create a new chromosome by combining the 2 chromosomes
        new_chromosome_1 = c1[:crossover_point] + c2[crossover_point:]
        new_chromosome_2 = c2[:crossover_point] + c1[crossover_point:]
        # calculate the fitness of old and new chromosomes
        model1 = gsm.genSubModel(c1, model)
        model2 = gsm.genSubModel(c2, model)
        model3 = gsm.genSubModel(new_chromosome_1, model)
        model4 = gsm.genSubModel(new_chromosome_2, model)
        fitness1 = fitness.getFitness(model1)
        if(fitness1 < 0.8):
            print("Breaking because of less Fitness")
            continue
        fitness2 = fitness.getFitness(model2)
        if(fitness2 < 0.8):
            print("Breaking because of less Fitness")
            continue
        fitness3 = fitness.getFitness(model3)
        if(fitness3 < 0.8):
            print("Breaking because of less Fitness")
            continue
        fitness4 = fitness.getFitness(model4)
        if(fitness4 < 0.8):
            print("Breaking because of less Fitness")
            continue
        # select the best 2 chromosomes
        fitnesses = [fitness1, fitness2, fitness3, fitness4]
        best_chromosomes = [c1, c2, new_chromosome_1, new_chromosome_2]
        best_chromosomes = [x for _, x in sorted(zip(fitnesses, best_chromosomes), reverse=True)]
        new_chromosomes.append(best_chromosomes[0])
        new_chromosomes.append(best_chromosomes[1])
        new_fitness.append(fitnesses[0])
        new_fitness.append(fitnesses[1])
        print("Newly added {} and {}".format(fitnesses[0], fitnesses[1]))

    return new_chromosomes, new_fitness

new_chromosomes, new_fitness = breed(chromosomes)

newly_bred_chromosomes_dict = {i:(new_chromosomes[i], new_fitness[i]) for i in range(len(new_chromosomes))}

# sort the chromosomes based on fitness
newly_bred_chromosomes_dict = {k: v for k, v in sorted(newly_bred_chromosomes_dict.items(), key=lambda item: item[1][1], reverse=True)}

# save to file

with open('chromosomes/newly_bred.json', 'w') as f:
    json.dump(newly_bred_chromosomes_dict, f)




# def mutate(chromosomes, new_fitness):
#     new_chromosomes = []
#     new_chromosomes_acc = []
#     for i, chromosome in enumerate(chromosomes):
#         print("Mutating for {} th time".format(i+1))
#         # randomly select a mutation point
#         mutation_point = random.randint(0, len(chromosome))
#         # mutate the chromosome
#         chromosome[mutation_point] = 1 - chromosome[mutation_point]
#         # calculate the fitness of old and new chromosomes
#         model2 = gsm.genSubModel(chromosome, model)
#         fitness1 = new_fitness[i]
#         fitness2 = fitness.getFitness(model2, epochs=2)
#         if(fitness2 < 0.7):
#             print("Breaking because of less fitness")
#             continue
#         # select the best chromosome
#         fitnesses = [fitness1, fitness2]
#         best_chromosomes = [chromosomes[i], chromosome ]
#         best_chromosomes , best_fitnesses = zip(*sorted(zip(best_chromosomes, fitnesses), reverse=True))
#         new_chromosomes.append(best_chromosomes[0])
#         new_chromosomes_acc.append(best_fitnesses[0])
#         print("Newly aded {}".format(best_fitnesses[0]))
#     return new_chromosomes, new_chromosomes_acc

# print("INFO: MUTATING CHROMOSOMES")
# new_chromosomes_mutated, new_chromosomes_acc_mutated = mutate(new_chromosomes, new_fitness)

# new_chromosomes_mutated = {i: [new_chromosomes_mutated[i], new_chromosomes_acc_mutated[i]] for i in range(len(new_chromosomes_mutated))}

# with open('chromosomes/new_cromosomes_mutated.json', 'w') as f:
#     json.dump(new_chromosomes_mutated, f)

# print("Obtained Accuracy {} obtained on chromosome {}".format(max(new_chromosomes_acc_mutated) , new_chromosomes_mutated[new_chromosomes_acc_mutated.index(max(new_chromosomes_acc_mutated))]))
