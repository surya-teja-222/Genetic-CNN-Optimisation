# A Genetic Minimisation Algorithm to predict covid-19 cases based on X-ray images on IoT devices

## Dataset

The dataset used for this project is the [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database).

## Files

-   `baseModel.py` - The base neural network model used for initial training
-   `chromosome.py` - Creates a list of chromosomes for the genetic algorithm natural selection
-   `fitness.py` - Calculates the fitness of model made from each chromosome
-   `test_train_data.py` - provides the test and train data for model training
-   `gen_sub_model.py` - Creates a new sub model on basis of the chromosome
-   `app.py` - to generate initial set of chromosomes, and breed them to create new set of chromosomes
-   `mutation.py` - to mutate the chromosomes
