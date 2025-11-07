import numpy as np

# ---------------------------
# Step 1: Define the Problem
# ---------------------------
def objective_function(x):
    # Function: f(x) = x^2 - 4x + 4
    return x**2 - 4*x + 4

# ---------------------------
# Step 2: Initialize Parameters
# ---------------------------
GRID_SIZE = (10, 10)      # 10x10 grid
NUM_ITERATIONS = 100      # Number of iterations
SEARCH_SPACE = (-10, 10)  # Range for x
NEIGHBORHOOD_RADIUS = 1   # 3x3 neighborhood

# ---------------------------
# Step 3: Initialize Population
# ---------------------------
def initialize_grid(grid_size, bounds):
    grid = np.random.uniform(bounds[0], bounds[1], size=grid_size)
    return grid

# ---------------------------
# Step 4: Evaluate Fitness
# ---------------------------
def evaluate_fitness(grid):
    return objective_function(grid)

# ---------------------------
# Step 5: Update Cell States
# ---------------------------
def update_grid(grid, fitness):
    new_grid = np.copy(grid)
    rows, cols = grid.shape

    for i in range(rows):
        for j in range(cols):
            # Get neighborhood indices
            neighborhood = []
            for dx in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
                for dy in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
                    ni = (i + dx) % rows  # Wrap around (toroidal)
                    nj = (j + dy) % cols
                    neighborhood.append(grid[ni, nj])
            
            # Update rule: average of neighbors
            new_value = np.mean(neighborhood)
            
            # Optional: add small random mutation
            mutation = np.random.uniform(-0.1, 0.1)
            new_value += mutation
            
            # Clip within search space
            new_value = np.clip(new_value, SEARCH_SPACE[0], SEARCH_SPACE[1])
            new_grid[i, j] = new_value

    return new_grid

# ---------------------------
# Step 6: Main Algorithm Loop
# ---------------------------
def parallel_cellular_algorithm():
    grid = initialize_grid(GRID_SIZE, SEARCH_SPACE)
    best_solution = None
    best_fitness = float('inf')

    for iteration in range(NUM_ITERATIONS):
        fitness = evaluate_fitness(grid)
        
        # Find current best
        min_idx = np.unravel_index(np.argmin(fitness), fitness.shape)
        current_best = grid[min_idx]
        current_fitness = fitness[min_idx]

        if current_fitness < best_fitness:
            best_solution = current_best
            best_fitness = current_fitness

        print(f"Iteration {iteration+1}/{NUM_ITERATIONS} | Best Fitness: {best_fitness:.6f}")

        # Update all cells
        grid = update_grid(grid, fitness)

    return best_solution, best_fitness

# ---------------------------
# Step 7: Run and Display Result
# ---------------------------
if __name__ == "__main__":
    best_x, best_fit = parallel_cellular_algorithm()
    print("\n✅ Best Solution Found: x =", best_x)
    print("✅ Best Fitness Value: f(x) =", best_fit)

#########OUTPUT############
Iteration 1/100 | Best Fitness: 0.042669
Iteration 2/100 | Best Fitness: 0.001046
Iteration 3/100 | Best Fitness: 0.000409
Iteration 4/100 | Best Fitness: 0.000409
Iteration 5/100 | Best Fitness: 0.000409
Iteration 6/100 | Best Fitness: 0.000409
Iteration 7/100 | Best Fitness: 0.000409
Iteration 8/100 | Best Fitness: 0.000409
Iteration 9/100 | Best Fitness: 0.000409
Iteration 10/100 | Best Fitness: 0.000409
Iteration 11/100 | Best Fitness: 0.000409
Iteration 12/100 | Best Fitness: 0.000409
Iteration 13/100 | Best Fitness: 0.000409
Iteration 14/100 | Best Fitness: 0.000409
Iteration 15/100 | Best Fitness: 0.000409
Iteration 16/100 | Best Fitness: 0.000409
Iteration 17/100 | Best Fitness: 0.000409
Iteration 18/100 | Best Fitness: 0.000409
Iteration 19/100 | Best Fitness: 0.000409
Iteration 20/100 | Best Fitness: 0.000409
Iteration 21/100 | Best Fitness: 0.000409
Iteration 22/100 | Best Fitness: 0.000409
Iteration 23/100 | Best Fitness: 0.000409
Iteration 24/100 | Best Fitness: 0.000409
Iteration 25/100 | Best Fitness: 0.000409
Iteration 26/100 | Best Fitness: 0.000409
Iteration 27/100 | Best Fitness: 0.000409
Iteration 28/100 | Best Fitness: 0.000409
Iteration 29/100 | Best Fitness: 0.000409
Iteration 30/100 | Best Fitness: 0.000409
Iteration 31/100 | Best Fitness: 0.000409
Iteration 32/100 | Best Fitness: 0.000409
Iteration 33/100 | Best Fitness: 0.000409
Iteration 34/100 | Best Fitness: 0.000409
Iteration 35/100 | Best Fitness: 0.000409
Iteration 36/100 | Best Fitness: 0.000409
Iteration 37/100 | Best Fitness: 0.000409
Iteration 38/100 | Best Fitness: 0.000409
Iteration 39/100 | Best Fitness: 0.000409
Iteration 40/100 | Best Fitness: 0.000409
Iteration 41/100 | Best Fitness: 0.000409
Iteration 42/100 | Best Fitness: 0.000409
Iteration 43/100 | Best Fitness: 0.000409
Iteration 44/100 | Best Fitness: 0.000409
Iteration 45/100 | Best Fitness: 0.000409
Iteration 46/100 | Best Fitness: 0.000409
Iteration 47/100 | Best Fitness: 0.000409
Iteration 48/100 | Best Fitness: 0.000409
Iteration 49/100 | Best Fitness: 0.000409
Iteration 50/100 | Best Fitness: 0.000409
Iteration 51/100 | Best Fitness: 0.000409
Iteration 52/100 | Best Fitness: 0.000409
Iteration 53/100 | Best Fitness: 0.000409
Iteration 54/100 | Best Fitness: 0.000409
Iteration 55/100 | Best Fitness: 0.000409
Iteration 56/100 | Best Fitness: 0.000409
Iteration 57/100 | Best Fitness: 0.000409
Iteration 58/100 | Best Fitness: 0.000409
Iteration 59/100 | Best Fitness: 0.000409
Iteration 60/100 | Best Fitness: 0.000409
Iteration 61/100 | Best Fitness: 0.000409
Iteration 62/100 | Best Fitness: 0.000409
Iteration 63/100 | Best Fitness: 0.000409
Iteration 64/100 | Best Fitness: 0.000409
Iteration 65/100 | Best Fitness: 0.000409
Iteration 66/100 | Best Fitness: 0.000409
Iteration 67/100 | Best Fitness: 0.000409
Iteration 68/100 | Best Fitness: 0.000409
Iteration 69/100 | Best Fitness: 0.000409
Iteration 70/100 | Best Fitness: 0.000409
Iteration 71/100 | Best Fitness: 0.000409
Iteration 72/100 | Best Fitness: 0.000409
Iteration 73/100 | Best Fitness: 0.000409
Iteration 74/100 | Best Fitness: 0.000409
Iteration 75/100 | Best Fitness: 0.000409
Iteration 76/100 | Best Fitness: 0.000409
Iteration 77/100 | Best Fitness: 0.000409
Iteration 78/100 | Best Fitness: 0.000409
Iteration 79/100 | Best Fitness: 0.000409
Iteration 80/100 | Best Fitness: 0.000409
Iteration 81/100 | Best Fitness: 0.000409
Iteration 82/100 | Best Fitness: 0.000409
Iteration 83/100 | Best Fitness: 0.000409
Iteration 84/100 | Best Fitness: 0.000409
Iteration 85/100 | Best Fitness: 0.000409
Iteration 86/100 | Best Fitness: 0.000409
Iteration 87/100 | Best Fitness: 0.000409
Iteration 88/100 | Best Fitness: 0.000409
Iteration 89/100 | Best Fitness: 0.000409
Iteration 90/100 | Best Fitness: 0.000409
Iteration 91/100 | Best Fitness: 0.000409
Iteration 92/100 | Best Fitness: 0.000409
Iteration 93/100 | Best Fitness: 0.000409
Iteration 94/100 | Best Fitness: 0.000409
Iteration 95/100 | Best Fitness: 0.000409
Iteration 96/100 | Best Fitness: 0.000409
Iteration 97/100 | Best Fitness: 0.000409
Iteration 98/100 | Best Fitness: 0.000409
Iteration 99/100 | Best Fitness: 0.000409
Iteration 100/100 | Best Fitness: 0.000409

✅ Best Solution Found: x = 2.020232001049803
✅ Best Fitness Value: f(x) = 0.00040933386647878933
