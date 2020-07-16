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


class Individual():
    """
    Each individual object has a state, which represents the queen placement on the board, aka a solution.
    Within an individual, there are "genes", which are parameters that join together to create a solution.
    Each individual has fitness, indicating how "good" its solution is (When fitness is 28, we have a solution)

    # State is a solution encoded as a Python list of integers from 0 - 7. Index of the list represents the column,
    # and the value for that index represents a Queen's placement on that column.
    """

    def __init__(self):
        # Generate randomly placed queens
        self.state = np.random.choice(8, 8, replace=True)
        self.fitness = None
        self.selection_prob = None

    def __repr__(self):
        return '{}: {} fitness: {} Selection prob: {}%'.format(self.__class__.__name__, self.state, self.fitness,
                                                               self.selection_prob * 100)

    def __cmp__(self, other):
        # For sorting individuals by their fitness in descending order
        if hasattr(other, 'fitness'):
            return self.fitness.__cmp__(other.fitness)

    def set_state(self, state):
        self.state = state

    def set_fitness(self, fitness):
        self.fitness = fitness

    def set_selection_prob(self, prob):
        self.selection_prob = prob


class Population():
    """
    Population is a set of individuals
    """

    def __init__(self, N):
        # Generate an initial population consisting of N Individuals with random "genes"
        self.population = [Individual() for _ in range(N)]

        # Iterate through Population, calculate each Individual's fitness, and the fitness sum of the entire population
        fitness_sum = 0
        for i in range(N):
            f = self.evaluate_fitness(self.population[i].state)
            self.population[i].set_fitness(f)
            fitness_sum += f

        # Set selection probability for each Individual.
        # Selection probability = Individual's fitness / fitness sum of entire population.
        for i in range(N):
            self.population[i].set_selection_prob(self.population[i].fitness / fitness_sum)

        # Sort by fitness, descending order
        self.sort_population()

    def new_population(self, new_population):
        """
        Creates a new population with offsprings after performing crossover

        """
        self.population = new_population
        fitness_sum = 0
        for i in range(N):
            f = self.evaluate_fitness(self.population[i].state)
            fitness_sum += f
            self.population[i].set_fitness(f)
        for i in range(N):
            self.population[i].set_selection_prob(self.population[i].fitness / fitness_sum)
        self.sort_population()

    def evaluate_fitness(self, individual_state):
        """
        Each Individual's solution's fitness is evaluated by calculating how many distinct queens are attacking another queen

        :param individual_state: Individual's state representing the board state aka a "solution"
        :return: Individual's fitness
        """
        # Find # of non attacking pairs
        dagnal_attacks = 0
        attacks_horizontal = 0
        # Check horizontal
        # Same row means queens are attacking in horizontal direction
        attacks_horizontal += abs(len(individual_state) - len(np.unique(individual_state)))
        # Check diagonal
        for i in range(len(individual_state)):
            for j in range(len(individual_state)):
                if (i != j):
                    delta_row = abs(individual_state[i] - individual_state[j])
                    delta_coloumn = abs(i - j)
                    if (delta_row == delta_coloumn):
                        dagnal_attacks += 1

        total_attacks = (dagnal_attacks / 2) + attacks_horizontal
        return 28 - total_attacks

    @staticmethod
    def random_selection(population):
        """
        Randomly select two Individuals for crossover. Individuals with higher fitness are more likely to be selected.

        :param population: Current population containing N Individuals
        :return: Two individuals, or "parents", selected to "reproduce"
        """
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

    def reproduce(self, parent1, parent2):
        """
        Decide where the cutoff will be in the genes.

        For example, if we have cutoff = 3, we take the first 3 bits from gene1,
        and take last 5 bits from gene2 to create offspring1 and vice versa for for offspring 2

        eg. gene1 = [0,1,2,3,4,5,6,7] and gene2 = [7,6,5,4,3,2,1,0]
        After crossover, Offspring1 = [0,1,2,4,3,2,1,0]
                         Offspring2 = [7,6,5,3,4,5,6,7]
        """

        cutoff = random.randint(1, 7)
        offspring1 = Individual()
        offspring2 = Individual()
        offspring1.set_state(np.concatenate([parent1.state[:cutoff], parent2.state[cutoff:]]))
        offspring2.set_state(np.concatenate([parent2.state[:cutoff], parent1.state[cutoff:]]))

        # Mutate the newly created offsprings (Randomly select one bit, and set to a random int)
        # This mechanism is for avoiding premature convergence
        if random.random() < mutation_pct:
            # Random index
            i = random.randint(0, 7)
            # Random value
            offspring1.state[i] = random.randint(0, 7)
        if random.random() < mutation_pct:
            # Random index
            i = random.randint(0, 7)
            # Random value
            offspring2.state[i] = random.randint(0, 7)

        offspring1.set_fitness(self.evaluate_fitness(offspring1.state))
        offspring2.set_fitness(self.evaluate_fitness(offspring2.state))

        return offspring1, offspring2

    def sort_population(self):
        self.population = sorted(self.population, key=self.getKey, reverse=True)

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


def genetic_algorithm(iteration):
    """
    Main function for performing the Genetic Algorithm

    :param iteration: How many iterations before stoppping the algorithm
    :return: solution: Board representation, or state, of a solution
             generations: If a solution is found, how many iterations it took
    """
    stop = False
    solution = None
    generations = 0
    pop = Population(N)
    print('starting pop')
    pop.print_population()

    for i in range(iteration):
        print('---------------------Generation {}-----------------------'.format(i + 1))
        new_population = []

        for _ in range(int(N / 2)):
            parent1, parent2 = pop.random_selection(pop.population)
            offspring1, offspring2 = pop.reproduce(parent1, parent2)
            new_population.append(offspring1)
            new_population.append(offspring2)

        pop.new_population(new_population)
        pop.set_fitness_sum()
        pop.print_population()

        # Check if there is a solution
        for individual in pop.population:
            if individual.fitness == 28:
                solution = individual
                print('solution found!')
                generations = i + 1
                stop = True
                break
            else:
                generations = iteration
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
    print("Time taken: {} seconds".format(end - start))

    plt.figure(figsize=(10, 10))
    generation_data = range(generations)
    plt.title('Fitness average vs Generation')
    plt.plot(generation_data, fitness_avg_data)
    plt.xlabel("Generation")
    plt.ylabel("Fitness average")
    plt.savefig('plot.png', bbox_inches='tight')
    plt.show()