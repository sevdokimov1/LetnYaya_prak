import numpy as np
import matplotlib.pyplot as plt
import random


class GeneticAlgorithm:
    def __init__(self, population_size=10, generations=50,
                 crossover_prob=0.7, mutation_prob=0.2):
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.best_solution = None
        self.best_fitness = float('-inf')

    def fitness(self, x):
        return x ** 2 + 20 * x - 34

    def create_initial_population(self, method='random'):
        if method == 'random':
            population = [random.uniform(8, 12) for _ in range(self.population_size)]
        else:
            population = [8 + i * (12 - 8) / (self.population_size - 1)
                          for i in range(self.population_size)]
        return population

    def selection(self, population, fitnesses):
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return random.choice(population)

        probabilities = [f / total_fitness for f in fitnesses]
        cumulative_probs = [sum(probabilities[:i + 1]) for i in range(len(probabilities))]

        r = random.random()
        for i, prob in enumerate(cumulative_probs):
            if r <= prob:
                return population[i]
        return population[-1]

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_prob:
            alpha = random.random()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2
            return child1, child2
        return parent1, parent2

    def mutation(self, individual):
        if random.random() < self.mutation_prob:
            mutation = random.uniform(-0.5, 0.5)
            individual += mutation
            individual = max(8, min(12, individual))
        return individual

    def run(self, method='random', visualize=True):
        population = self.create_initial_population(method)
        history = []
        all_intermediate = []

        for generation in range(self.generations):
            fitnesses = [self.fitness(ind) for ind in population]

            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_solution = population[best_idx]

            history.append({
                'generation': generation,
                'best_fitness': self.best_fitness,
                'best_solution': self.best_solution,
                'population': population.copy(),
                'fitnesses': fitnesses.copy()
            })

            all_intermediate.extend([(ind, fit) for ind, fit in zip(population, fitnesses)])

            new_population = []

            new_population.append(self.best_solution)

            while len(new_population) < self.population_size:
                parent1 = self.selection(population, fitnesses)
                parent2 = self.selection(population, fitnesses)

                child1, child2 = self.crossover(parent1, parent2)

                child1 = self.mutation(child1)
                child2 = self.mutation(child2)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

        if visualize:
            self.visualize(history, all_intermediate)

        return self.best_solution, self.best_fitness, history

    def visualize(self, history, all_intermediate):
        x = np.linspace(8, 12, 400)
        y = x ** 2 + 20 * x - 34

        plt.figure(figsize=(12, 8))
        plt.plot(x, y, 'g-', linewidth=2, label='f(x) = x² + 20x - 34')

        intermediate_x = [ind for ind, _ in all_intermediate]
        intermediate_y = [self.fitness(ind) for ind in intermediate_x]
        plt.scatter(intermediate_x, intermediate_y, c='blue', s=20,
                    alpha=0.5, label='Промежуточные решения', zorder=2)

        plt.scatter([self.best_solution], [self.best_fitness],
                    c='red', s=100, marker='*', label=f'Оптимальное решение\nx={self.best_solution:.4f}',
                    zorder=3)

        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.title('Генетический алгоритм: Поиск максимума функции', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axvline(x=8, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=12, color='gray', linestyle='--', alpha=0.5)
        plt.show()

        generations = [h['generation'] for h in history]
        best_fitnesses = [h['best_fitness'] for h in history]

        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitnesses, 'b-', linewidth=2)
        plt.xlabel('Поколение', fontsize=12)
        plt.ylabel('Лучшая приспособленность', fontsize=12)
        plt.title('Сходимость генетического алгоритма', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("ГЕНЕТИЧЕСКИЙ АЛГОРИТМ")
    print("Поиск максимума функции: f(x) = x² + 20x - 34")
    print("Интервал: [8, 12]")
    print("=" * 60)

    ga = GeneticAlgorithm(
        population_size=10,
        generations=50,
        crossover_prob=0.7,
        mutation_prob=0.2
    )

    best_x, best_f, history = ga.run(method='random', visualize=True)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 60)
    print(f"Лучшее решение: x = {best_x:.6f}")
    print(f"Значение функции: f(x) = {best_f:.6f}")
    print(f"Теоретический максимум на [8,12]: f(12) = {12 ** 2 + 20 * 12 - 34:.6f}")
    print("=" * 60)
