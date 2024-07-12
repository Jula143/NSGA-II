# Importing required modules
import math
import random
import matplotlib.pyplot as plt
import numpy as np


def zdt1_f1(x):
    return x[0]

def zdt1_f2(x):
    g_x = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    h_x = 1 - math.sqrt(x[0] / g_x)
    f2 = g_x * h_x
    return f2


def zdt2_f1(x):
    return x[0]

def zdt2_f2(x):
    g_x = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    h_x = 1 - math.pow(x[0] / g_x, 2)
    f2 = g_x * h_x
    return f2

# First function to optimize
def zdt3_f1(x):
    return x[0]

def zdt3_f2(x):
    g_x = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    h_x = 1 - math.sqrt(x[0] / g_x) - (x[0]/g_x) * math.sin(10*math.pi*x[0])
    f2 = g_x * h_x
    return f2


def zdt4_f1(x):
    return x[0]

def zdt4_f2(x):
    g_x = 1 + 10 * (len(x) - 1) + sum(math.pow(xi, 2) - 10 * math.cos(4 * math.pi * xi) for xi in x[1:])
    h_x = 1 - math.sqrt(x[0] / g_x)
    f2 = g_x * h_x
    return f2


def zdt6_f1(x):
    return 1 - np.exp(-4 * x[0]) * np.sin(6 * np.pi * x[0]) ** 6

def zdt6_f2(x):
    f1 = 1 - np.exp(-4 * x[0]) * np.sin(6 * np.pi * x[0]) ** 6
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1) ** 0.25
    f2 = g * (1 - (f1 / g) ** 2)
    return f2


# Function to find index of list
def index_of(a, lst):
    for i, value in enumerate(lst):
        if value == a:
            return i
    return -1

# Function to sort by values
def sort_by_values(lst, values):
    sorted_list = []
    while len(sorted_list) != len(lst):
        if index_of(min(values), values) in lst:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list

# Function to carry out NSGA-II's fast non-dominated sort
def fast_non_dominated_sort(values1, values2):
    S = [[] for _ in range(len(values1))] #set of solutions dominated by each solution
    front = [[]] 
    n = [0 for _ in range(len(values1))] #the domination count for each solution
    rank = [0 for _ in range(len(values1))] #stores the rank of each solution

    for p in range(len(values1)):
        for q in range(len(values1)):
            if ((values1[p] <= values1[q]) and (values2[p] < values2[q])) or ((values1[p] < values1[q]) and (values2[p] <= values2[q])):
                if q not in S[p]:
                    S[p].append(q)
            elif ((values1[q] <= values1[p]) and (values2[q] < values2[p])) or ((values1[q] < values1[p]) and (values2[q] <= values2[p])):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while front[i] != []:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)
        
    del front[len(front) - 1]
    return front, rank


# Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for _ in range(len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])

    distance[0] = float('inf')
    distance[len(front) - 1] = float('inf')
    
    for k in range(1, len(front) - 1):
        if (max(values1) - min(values1))==0:
            continue
        distance[k] = distance[k] + (values1[sorted1[k + 1]] - values2[sorted1[k - 1]]) / (max(values1) - min(values1))

    for k in range(1, len(front) - 1):
        if (max(values2) - min(values2))==0:
            continue
        distance[k] = distance[k] + (values1[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (max(values2) - min(values2))

    return distance

def simulated_binary_crossover(parent1, parent2, eta=2.0, crossover_prob=0.9):
    child1, child2 = np.copy(parent1), np.copy(parent2)

    if random.random() <= crossover_prob:
        u = random.random()
        beta = 0.0

        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))

        child1 = 0.5 * ((1.0 + beta) * parent1 + (1.0 - beta) * parent2)
        child2 = 0.5 * ((1.0 - beta) * parent1 + (1.0 + beta) * parent2)
        
    child1 = np.clip(child1, 0, 1)
    child2 = np.clip(child2, 0, 1)

    return child1, child2



def polynomial_mutation(individual, mutation_prob=0.1, eta=20.0):
    mutated_individual = np.copy(individual)

    for i in range(len(mutated_individual)):
        if random.random() <= mutation_prob:
            u = random.random()
            delta1 = (u ** (1.0 / (eta + 1.0))) - 1.0
            delta2 = 1.0 - (u ** (1.0 / (eta + 1.0)))

            mutated_individual[i] += delta1 if random.random() <= 0.5 else delta2
    mutated_individual = np.clip(mutated_individual, 0, 1)
    return mutated_individual




# Main program starts here
pop_size = 100
max_gen = 501

dimensions = 30

# Initialization
min_x = 0
max_x = 1

pareto_fronts=[]

population = [[min_x + (max_x - min_x) * random.random() for _ in range(dimensions)] for _ in range(pop_size)]

for gen in range(max_gen):
    zdt1_f1_values = [zdt1_f1(population[i]) for i in range(0, pop_size)]
    zdt1_f2_values = [zdt1_f2(population[i]) for i in range(0, pop_size)]

    parents = []
    
    non_dominated_sorted_population, ranks = fast_non_dominated_sort(zdt1_f1_values[:], zdt1_f2_values[:])
    distances = crowding_distance(zdt1_f1_values[:], zdt1_f2_values[:], list(range(0, len(population))))
    distances_sorted = sort_by_values(list(range(0, len(population))), distances)

    # create parents population
    while len(parents) != int(pop_size/2):
        a1 = random.randint(0, pop_size - 1)
        b1 = random.randint(0, pop_size - 1)
        #compare
        if ranks[a1]<ranks[b1]:
            parents.append(population[a1])
        elif ranks[b1]<ranks[a1]:
            parents.append(population[b1])
        else:
            if distances_sorted[a1]>distances_sorted[b1]:
                parents.append(population[a1])
            else:
                parents.append(population[b1])
                
    population2=population[:]
    # Generating offsprings and apply mutation
    while len(population2) != 2 * pop_size:
        a1 = random.randint(0, len(parents) - 1)
        b1 = random.randint(0, len(parents) - 1)
        
        c1, c2 = simulated_binary_crossover(np.array(parents[a1]), np.array(parents[b1]))
        c1 = polynomial_mutation(c1)
        c2 = polynomial_mutation(c2)
        population2.append(c1)
        population2.append(c2)
    
           
    zdt1_f1_values2 = [zdt1_f1(population2[i]) for i in range(0, 2 * pop_size)]
    zdt1_f2_values2 = [zdt1_f2(population2[i]) for i in range(0, 2 * pop_size)]
    non_dominated_sorted_population2, ranks2 = fast_non_dominated_sort(zdt1_f1_values2[:], zdt1_f2_values2[:])

    new_population = []
    j = 0
    while len(new_population) + len(non_dominated_sorted_population2[j]) < pop_size: 
        new_population.extend(non_dominated_sorted_population2[j])   
        j+=1 
        
    crowding_values = crowding_distance(zdt1_f1_values2[:], zdt1_f2_values2[:], non_dominated_sorted_population2[j][:])
            
    non_dominated_sorted_population2_1 = list(range(0, len(non_dominated_sorted_population2[j])))
    front22 = sort_by_values(non_dominated_sorted_population2_1[:], crowding_values)
    front = [non_dominated_sorted_population2[j][front22[i]] for i in range(0, len(non_dominated_sorted_population2[j]))]
    front.reverse()
   
    for value in front:
        if len(new_population)==pop_size:
            break
        new_population.append(value)    
        
    if gen in [20, 50, 100, 500]:
        pareto_fronts.append((zdt1_f1_values, zdt1_f2_values))
            
    population = [population2[i] for i in new_population]
    
# Plot Pareto fronts at specified generations
gens = [20, 50, 100, 500]
plt.figure(figsize=(10, 6))
for i, (f1_values, f2_values) in enumerate(pareto_fronts):
    plt.scatter(f1_values, f2_values, label=f'Generation {gens[i]}')


plt.xlabel('Function 1')
plt.ylabel('Function 2')
plt.legend()
plt.show()
