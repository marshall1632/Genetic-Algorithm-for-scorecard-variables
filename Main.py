
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Load dataset
data = pd.read_csv('data/bank-additional-full.csv', sep=';')

target = 'y'
X = data.drop(columns=[target])
y = data[target].apply(lambda x: 1 if x == 'yes' else 0)

X_encoded = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
feature_names = X_encoded.columns.tolist()

# Genetic Algorithm Parameters
population_size = 100
num_generations = 20
mutation_rate = 0.1
num_features = X_scaled.shape[1]  # Number of features

print("the process is starting now")


def initialize_population():
    population = []
    for _ in range(population_size):
        individual = np.random.choice([0, 1], size=num_features)
        population.append(individual)
    return np.array(population)


def fitness(individual):
    selected_features = [i for i in range(num_features) if individual[i] == 1]
    if len(selected_features) == 0:
        return 0
    X_selected = X_scaled[:, selected_features]
    model = LogisticRegression(max_iter=1000)
    scores = cross_val_score(model, X_selected, y, cv=5, scoring='roc_auc')
    return np.mean(scores)


def selection(population, fitness_scores):
    sorted_indices = np.argsort(fitness_scores)[::-1]
    selected = population[sorted_indices[:population_size // 2]]  #
    return selected


def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, num_features - 1)
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2


def mutate(individual):
    for i in range(num_features):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual


population = initialize_population()
best_scores = []  # Track best AUC per generation

for generation in range(num_generations):
    fitness_scores = np.array([fitness(ind) for ind in population])
    best_score = np.max(fitness_scores)
    best_scores.append(best_score)  # Store the best score for visualization
    print(f"Generation {generation + 1} - Best AUC: {best_score:.4f}")

    selected_population = selection(population, fitness_scores)

    new_population = []
    while len(new_population) < population_size:
        parents = np.random.choice(len(selected_population), size=2, replace=False)
        child1, child2 = crossover(selected_population[parents[0]], selected_population[parents[1]])
        new_population.append(mutate(child1))
        new_population.append(mutate(child2))

    population = np.array(new_population[:population_size])

final_fitness_scores = np.array([fitness(ind) for ind in population])
best_individual = population[np.argmax(final_fitness_scores)]
best_features = [feature_names[i] for i in range(num_features) if best_individual[i] == 1 and feature_names[i] in data.columns]

print("\nBest feature subset:", best_features)


# Visualization of Best Fitness Score over Generations
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_generations + 1), best_scores, marker='o', color='b', label='Best AUC per Generation')
plt.xlabel('Generation')
plt.ylabel('Best AUC Score')
plt.title('Best Fitness Score (AUC) Across Generations')
plt.legend()
plt.grid()
plt.show()

# Visualization of Selected Features and their Frequency
feature_counts = np.array(best_individual)  # Binary array of the best features
selected_features = [feature_names[i] for i in range(num_features) if feature_counts[i] == 1]

plt.figure(figsize=(10, 8))
plt.barh(selected_features, feature_counts[feature_counts == 1], color='teal')
plt.xlabel('Frequency (Selected Features)')
plt.title('Feature Importance in the Best Feature Subset')
plt.show()
