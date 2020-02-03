from timeit import default_timer as timer
import random
import numpy as np
from matplotlib import pyplot as plt
import time

N = 300
mutation_pct = 0.1
iteration = 3000
fitness_avg_data = []
global generations

class Individual(object):
    def __init__(self):
        #self.state= np.array(random.sample(range(8), 8))
        self.state = np.random.choice(8, 8, replace=True)
        self.fitness = None
        self.selection_prob = None

    def __repr__(self):
        return '{}: {} fitness: {} Selection prob: {}%'.format(self.__class__.__name__, self.state, self.fitness, self.selection_prob*100)

    def __cmp__(self, other):
        if hasattr(other, 'fitness'):
            return self.fitness.__cmp__(other.fitness)

    def set_state(self, state):
        self.state = state

    def set_fitness(self, fitness):
        self.fitness = fitness

    def set_selection_prob(self, prob):
        self.selection_prob = prob

class Population():
    def __init__(self, N):
        self.population = [Individual() for _ in range(N)]
        fitness_sum = 0
        for i in range(N):
            f = evaluate_fitness(self.population[i].state)
            fitness_sum += f
            self.population[i].set_fitness(f)
        for i in range(N):
            self.population[i].set_selection_prob(self.population[i].fitness / fitness_sum)
        self.sort_population()

    def new_population(self, new_population):
        self.population = new_population
        fitness_sum = 0
        for i in range(N):
            f = evaluate_fitness(self.population[i].state)
            fitness_sum += f
            self.population[i].set_fitness(f)
        for i in range(N):
            self.population[i].set_selection_prob(self.population[i].fitness / fitness_sum)
        self.sort_population()

    def sort_population(self):
        self.population = sorted(self.population, key= self.getKey, reverse= True)

    def print_population(self):
        for i in range(len(self.population)):
            print('{}'.format(self.population[i]))

    def set_fitness_sum(self):
        fitness_sum = 0
        for i in range(len(self.population)):
            fitness_sum += self.population[i].fitness
        fitness_avg_data.append(fitness_sum / N)

    def getKey(self, individual):
        return individual.fitness

def evaluate_fitness(individual_state):
    # Find # of non attacking pairs
    dagnal_attacks = 0
    attacks_horizontal = 0
    # Check horizontal
    # Same row means queens are attacking in horizontal direction
    attacks_horizontal += abs(len(individual_state) - len(np.unique(individual_state)))
    # For each queen in coloumn i
    for i in range(len(individual_state)):
        # Check against every other queen
        for j in range(len(individual_state)):
            if (i != j):
                delta_row = abs(individual_state[i] - individual_state[j])
                delta_coloumn = abs(i - j)
                if (delta_row == delta_coloumn):
                    dagnal_attacks += 1

    total_attacks = (dagnal_attacks/2) + attacks_horizontal
    return 28 - total_attacks

def random_selection(population):
    choice = []
    arr_idx = np.arange(N)
    # Array of selection probabilities from current generation
    select_probs = [x.selection_prob for x in population]
    while True:
        a = np.random.choice(arr_idx, 2, p=select_probs)
        if a[0] == a[1]:
            continue
        else:
            choice.append(a[0])
            choice.append(a[1])
            break

    parent1 = population[a[0]]
    parent2 = population[a[1]]

    return parent1, parent2

def reproduce(parent1, parent2):
    cutoff = random.randint(1,7)
    offspring1 = Individual()
    offspring2 = Individual()
    offspring1.set_state(np.concatenate([parent1.state[:cutoff], parent2.state[cutoff:]]))
    offspring2.set_state(np.concatenate([parent2.state[:cutoff], parent1.state[cutoff:]]))

    # Mutate
    if random.random() < mutation_pct:
        # Random index
        i = random.randint(0,7)
        # Random value
        offspring1.state[i] = random.randint(0,7)
    if random.random() < mutation_pct:
        # Random index
        i = random.randint(0, 7)
        # Random value
        offspring2.state[i] = random.randint(0, 7)

    offspring1.set_fitness(evaluate_fitness(offspring1.state))
    offspring2.set_fitness(evaluate_fitness(offspring2.state))

    return offspring1, offspring2

def genetic_algorithm(iteration):
    stop = False
    solution = None
    generations = 0
    population = Population(N)
    print('starting population')
    population.print_population()

    for i in range(iteration):
        print('---------------------Generation {}-----------------------'.format(i+1))
        new_population = []

        for _ in range(int(N/2)):
            parent1, parent2 = random_selection(population.population)
            offspring1, offspring2 = reproduce(parent1, parent2)
            new_population.append(offspring1)
            new_population.append(offspring2)

        population.new_population(new_population)
        population.set_fitness_sum()
        population.print_population()

        # Check if there is a solution
        for individual in population.population:
            if individual.fitness == 28:
                solution = individual
                print('solution found!')
                generations = i + 1
                stop = True
                break
            else: generations = iteration
        if stop == True:
            break

    return solution, generations

if __name__ == "__main__":
    start = timer()
    solution, generations = genetic_algorithm(iteration)

    if solution is None:
        print('No solution found')
    else:
        print(solution)
        print('In generation {}'.format(generations))

    end = timer()
    print("Time taken: {} seconds".format(end-start))
    
    plt.figure(figsize=(10, 10))
    generation_data = range(generations)
    plt.title('Fitness average vs Generation')
    plt.plot(generation_data, fitness_avg_data)
    plt.xlabel("Generation")
    plt.ylabel("Fitness average")
    plt.savefig('plot.png', bbox_inches='tight')
    plt.show()