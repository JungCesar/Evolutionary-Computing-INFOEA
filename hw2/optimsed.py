import random
import math
import time
import os
import numpy as np
import pandas as pd

# Global constants
NUM_VERTICES = 500
FM_PASSES = 10000
POPULATION_SIZE = 50
EXPERIMENT_RUNS = 10
RUNTIME_RUNS = 25


class Node:
    __slots__ = ("gain", "neighbors", "prev", "next", "partition", "moved")

    def __init__(self, gain=0, neighbors=None, prev=-1, next=-1, partition=0):
        self.gain = gain
        self.neighbors = neighbors if neighbors is not None else []
        self.prev = prev
        self.next = next
        self.partition = partition
        self.moved = False


def calculate_gain(node_idx, solution, graph):
    """
    Calculates the gain for moving a node to the opposite partition.
    This version assumes vertices are 0-indexed.
    """
    gain = 0
    for neighbor in graph[node_idx]:
        if solution[neighbor] == solution[node_idx]:
            gain -= 1
        else:
            gain += 1
    return gain


def insert_node(nodes, gain_lists, max_gain, node_index, max_degree):
    """
    Inserts a node into the doubly-linked list corresponding to its gain.
    """
    gain = nodes[node_index].gain
    part = nodes[node_index].partition
    gain_index = gain + max_degree
    nodes[node_index].next = gain_lists[part][gain_index]
    if gain_lists[part][gain_index] != -1:
        nodes[gain_lists[part][gain_index]].prev = node_index
    gain_lists[part][gain_index] = node_index
    nodes[node_index].prev = -1
    if gain > max_gain[part]:
        max_gain[part] = gain


def remove_node(nodes, gain_lists, node_index, max_degree):
    """
    Removes a node from the doubly-linked list.
    """
    gain = nodes[node_index].gain
    part = nodes[node_index].partition
    gain_index = gain + max_degree
    if nodes[node_index].prev != -1:
        nodes[nodes[node_index].prev].next = nodes[node_index].next
    else:
        gain_lists[part][gain_index] = nodes[node_index].next
    if nodes[node_index].next != -1:
        nodes[nodes[node_index].next].prev = nodes[node_index].prev
    nodes[node_index].prev = -1
    nodes[node_index].next = -1


def update_max_gain(gain_lists, max_gain, max_degree):
    """
    Updates the maximum gain pointers after node movements.
    """
    for part in range(2):
        while (
            max_gain[part] >= -max_degree
            and gain_lists[part][max_gain[part] + max_degree] == -1
        ):
            max_gain[part] -= 1
        if max_gain[part] < -max_degree:  # Correctly handle empty gain lists
            max_gain[part] = -max_degree
            for g in range(-max_degree, max_degree + 1):
                if gain_lists[part][g + max_degree] != -1:
                    max_gain[part] = g
                    break


def fm_heuristic(solution, graph, max_passes=FM_PASSES):
    """
    Fiduccia-Mattheyses heuristic for graph bipartitioning using a doubly-linked list.
    Ensures balanced partitions.

    Args:
        solution (list): Initial partition assignment (0 or 1) for each vertex.
        graph (list): List of adjacency lists for each vertex (0-indexed).
        max_passes (int): Maximum number of passes to perform.

    Returns:
        tuple: (optimized_solution, passes_performed)
    """
    n = len(solution)
    max_degree = max(len(neighbors) for neighbors in graph)

    # Initialize nodes and gain lists
    nodes = [Node() for _ in range(n)]
    gain_lists = [[-1] * (2 * max_degree + 1) for _ in range(2)]
    max_gain = [-max_degree] * 2

    for i in range(n):
        nodes[i].gain = calculate_gain(i, solution, graph)
        nodes[i].neighbors = graph[i]
        nodes[i].partition = solution[i]
        nodes[i].moved = False

        insert_node(nodes, gain_lists, max_gain, i, max_degree)

    best_solution = solution.copy()
    best_cut_size = fitness_function(solution, graph)

    improved = True
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1

        # Reset moved flags and rebuild gain lists
        for i in range(n):
            nodes[i].moved = False
        gain_lists = [[-1] * (2 * max_degree + 1) for _ in range(2)]
        max_gain = [-max_degree] * 2
        indices = list(range(n))
        random.shuffle(indices)  # shuffling is key!
        for i in indices:
            nodes[i].gain = calculate_gain(i, solution, graph)
            insert_node(nodes, gain_lists, max_gain, i, max_degree)
        moves = []
        cumulative_gain = 0
        best_cumulative_gain = 0
        best_move_index = -1
        num_moved_part0 = 0  # Track moved nodes from partition 0
        num_moved_part1 = 0  # Track moved nodes from partition 1
        initial_part0_count = solution.count(0)
        initial_part1_count = n - initial_part0_count

        # Process all vertices in one pass
        for _ in range(n):
            best_node = -1
            best_gain_val = float("-inf")
            best_part = -1  # Store the best partition

            for part in range(2):
                if (
                    max_gain[part] > best_gain_val
                    and gain_lists[part][max_gain[part] + max_degree] != -1
                ):
                    potential_node = gain_lists[part][max_gain[part] + max_degree]

                    # --- Balance Check ---
                    if part == 0:
                        # Check the number of nodes is not under the limit if we removed one.
                        if (initial_part0_count - (num_moved_part0 + 1)) >= (
                            n // 2
                        ) - 1:
                            best_gain_val = max_gain[part]
                            best_node = potential_node
                            best_part = part  # Important
                    elif part == 1:
                        # Check the number of nodes is not under the limit if we removed one.

                        if (initial_part1_count - (num_moved_part1 + 1)) >= (
                            n // 2
                        ) - 1:
                            best_gain_val = max_gain[part]
                            best_node = potential_node
                            best_part = part

            if best_node == -1:
                break  # No more nodes to move

            # Move the selected node
            nodes[best_node].moved = True
            remove_node(nodes, gain_lists, best_node, max_degree)
            old_partition = solution[best_node]
            solution[best_node] = 1 - solution[best_node]
            nodes[best_node].partition = solution[best_node]

            # --- Update Moved Counts ---
            if old_partition == 0:
                num_moved_part0 += 1
            else:
                num_moved_part1 += 1

            cumulative_gain += best_gain_val
            moves.append(best_node)

            if cumulative_gain > best_cumulative_gain:
                best_cumulative_gain = cumulative_gain
                best_move_index = len(moves) - 1

            # Recompute gains for all neighbors
            for neighbor in nodes[best_node].neighbors:
                if not nodes[neighbor].moved:
                    remove_node(nodes, gain_lists, neighbor, max_degree)
                    new_gain = calculate_gain(neighbor, solution, graph)
                    nodes[neighbor].gain = new_gain
                    insert_node(nodes, gain_lists, max_gain, neighbor, max_degree)

            update_max_gain(gain_lists, max_gain, max_degree)

        # Rollback moves beyond the best point in this pass
        # Ensure that after rollback, the partitions remain balanced.
        if best_move_index >= 0:
            for i in range(len(moves) - 1, best_move_index, -1):
                node_idx = moves[i]
                solution[node_idx] = 1 - solution[node_idx]  # Correct Rollback

            current_cut_size = fitness_function(solution, graph)
            if current_cut_size < best_cut_size:
                best_cut_size = current_cut_size
                best_solution = solution.copy()
                improved = True
            else:  # Important: revert to best solution.
                solution = best_solution.copy()

        else:  # Important: roll back all changes
            solution = best_solution.copy()

    return best_solution, passes


def fitness_function(solution, graph):
    """
    Computes the fitness as the number of cut-edges between partitions.
    """
    total = 0
    for i, neighbors in enumerate(graph):
        for nb in neighbors:
            if solution[i] != solution[nb]:
                total += 1
    return total // 2


def read_graph_data(filename):
    """Reads the graph and returns adjacency lists (using sets)."""
    graph = [set() for _ in range(NUM_VERTICES)]  # Use sets for neighbors
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            vertex_index = int(parts[0]) - 1  # 0-based indexing
            if (
                vertex_index < 0 or vertex_index >= NUM_VERTICES
            ):  # Correct the out of bounds issue.
                print(f"Vertex index out of bounds: {vertex_index + 1}")
                continue
            connected_vertices = [
                int(n) - 1 for n in parts[3:]
            ]  # 0-based, and start from index 3
            graph[vertex_index].update(
                connected_vertices
            )  # Use update to correctly add the neighbours.
            for neighbor in connected_vertices:  # Add the other way around too.
                graph[neighbor].add(vertex_index)
    return graph


def generate_random_solution():
    """Generates a balanced random solution."""
    half = NUM_VERTICES // 2
    solution = [0] * half + [1] * half
    random.shuffle(solution)
    return solution


def mutate(solution, mutation_size):
    """Perturbs by swapping bits."""
    mutated = solution.copy()
    zeros = [i for i, bit in enumerate(mutated) if bit == 0]
    ones = [i for i, bit in enumerate(mutated) if bit == 1]
    random.shuffle(zeros)
    random.shuffle(ones)
    num_mutations = min(mutation_size, len(zeros), len(ones))
    for i in range(num_mutations):
        mutated[zeros[i]] = 1
        mutated[ones[i]] = 0
    return mutated


def MLS(graph, num_starts, time_limit=None):
    """
    Multi-start Local Search.
    Applies FM local search to multiple random starting solutions.
    Includes an optional time limit.
    """
    best_solution = None
    best_fit = float("inf")
    start_time = time.perf_counter()  # Record start time

    for _ in range(num_starts):
        if time_limit is not None and time.perf_counter() - start_time >= time_limit:
            break  # Exit if time limit reached

        sol = generate_random_solution()
        local_opt, _ = fm_heuristic(sol, graph)
        fit = fitness_function(local_opt, graph)
        if fit < best_fit:
            best_fit = fit
            best_solution = local_opt.copy()  # Use .copy() for safety

    return best_solution


def ILS(graph, initial_solution, mutation_size, time_limit=None):
    """
    Simple Iterated Local Search (accepts only improvements).
    Includes an optional time limit.
    """
    best_solution = initial_solution.copy()
    best_fit = fitness_function(best_solution, graph)
    no_improvement = 0
    max_no_improvement = 10
    unchanged_count = 0
    start_time = time.perf_counter()  # Record start time

    while no_improvement < max_no_improvement:
        if time_limit is not None and time.perf_counter() - start_time >= time_limit:
            break  # Exit if time limit reached

        mutated = mutate(best_solution, mutation_size)
        local_opt, _ = fm_heuristic(mutated, graph)
        fit = fitness_function(local_opt, graph)
        if fit == best_fit:
            unchanged_count += 1
        if fit < best_fit:
            best_solution = local_opt.copy()
            best_fit = fit
            no_improvement = 0
        else:
            no_improvement += 1
    return best_solution, unchanged_count


def ILS_annealing(graph, initial_solution, mutation_size, time_limit=None):
    """
    Iterated Local Search with annealing.
    Includes an optional time limit.
    """
    best_solution = initial_solution.copy()
    best_fit = fitness_function(best_solution, graph)
    current_solution = initial_solution.copy()
    current_fit = best_fit
    no_improvement = 0
    max_no_improvement = 10
    unchanged_count = 0
    temperature = 1000.0
    cooling_rate = 0.95
    start_time = time.perf_counter()

    while no_improvement < max_no_improvement:
        if time_limit is not None and time.perf_counter() - start_time >= time_limit:
            break

        mutated = mutate(current_solution, mutation_size)
        local_opt, _ = fm_heuristic(mutated, graph)
        fit = fitness_function(local_opt, graph)
        if fit == current_fit:
            unchanged_count += 1
        if fit < current_fit:
            current_solution = local_opt.copy()
            current_fit = fit
            if fit < best_fit:
                best_solution = local_opt.copy()
                best_fit = fit
                no_improvement = 0
        else:
            # Annealing acceptance
            if math.exp((current_fit - fit) / temperature) > random.random():
                current_solution = local_opt.copy()
                current_fit = fit
            no_improvement += 1
        temperature *= cooling_rate
    return best_solution, unchanged_count


def ILS_adaptive(graph, initial_solution, time_limit=None):
    """Adaptive ILS.  Includes a time limit."""
    initial_mutation_mean = 8
    mutation_std_dev = 2
    decay_factor = 0.98
    stagnation_threshold = 50
    max_iterations = 2000  # Keep a max iteration count, even with time limit
    min_mutation_size = 1
    max_mutation_size = 50
    restart_reset = True
    stagnation_window = 15
    stagnation_tolerance = 0.0001

    best_solution = initial_solution.copy()
    best_fit = fitness_function(best_solution, graph)
    current_solution = initial_solution.copy()
    current_fit = best_fit
    current_mutation_mean = initial_mutation_mean
    mutation_history = []
    stagnation_count = 0
    iteration = 0
    best_fitness_history = [best_fit]
    fitness_history = []
    start_time = time.perf_counter()

    while iteration < max_iterations and best_fit != 0:
        if time_limit is not None and time.perf_counter() - start_time >= time_limit:
            break  # Exit if time limit reached
        iteration += 1

        mutation_size = int(np.random.normal(current_mutation_mean, mutation_std_dev))
        mutation_size = max(min_mutation_size, min(mutation_size, max_mutation_size))

        mutated_solution = mutate(current_solution, mutation_size)
        local_opt_solution, passes = fm_heuristic(mutated_solution, graph)
        local_opt_fit = fitness_function(local_opt_solution, graph)
        fitness_history.append(local_opt_fit)

        if local_opt_fit < current_fit:
            current_solution = local_opt_solution.copy()
            current_fit = local_opt_fit
            if current_fit < best_fit:
                best_solution = current_solution.copy()
                best_fit = current_fit
            current_mutation_mean = max(
                min_mutation_size, current_mutation_mean * decay_factor
            )
            mutation_std_dev = max(0.5, mutation_std_dev * decay_factor)
            stagnation_count = 0
        else:
            stagnation_count += 1

        mutation_history.append((iteration, mutation_size, current_fit))
        best_fitness_history.append(best_fit)

        if stagnation_count >= stagnation_threshold:
            if len(fitness_history) >= stagnation_window:
                avg_change = np.mean(np.diff(fitness_history[-stagnation_window:]))
                if abs(avg_change) < stagnation_tolerance:
                    if restart_reset:
                        current_mutation_mean = min(
                            max_mutation_size, current_mutation_mean + 1
                        )
                        mutation_std_dev = min(
                            mutation_std_dev * 1.2, current_mutation_mean / 4
                        )
                        print(
                            f"Iteration {iteration}: Stagnation detected. Increasing mutation mean to {current_mutation_mean:.2f}, std dev to {mutation_std_dev:.2f}"
                        )
                    stagnation_count = 0
                else:
                    current_mutation_mean = max(
                        min_mutation_size, current_mutation_mean * decay_factor
                    )
            else:
                if restart_reset:
                    current_mutation_mean = min(
                        max_mutation_size, current_mutation_mean + 1
                    )
                    mutation_std_dev = min(
                        mutation_std_dev * 1.2, current_mutation_mean / 4
                    )
                    print(
                        f"Iteration {iteration}: Stagnation detected. Increasing mutation mean to {current_mutation_mean:.2f}, std dev to {mutation_std_dev:.2f}"
                    )
                stagnation_count = 0
        if iteration % 100 == 0:
            print(
                f"Iteration {iteration}: Best fitness = {best_fit}, Current mutation mean = {current_mutation_mean:.2f}, std_dev = {mutation_std_dev:.2f}"
            )

    print(f"Terminated after {iteration} iterations. Best fitness: {best_fit}")
    return (
        best_solution,
        best_fit,
        mutation_history,
        iteration,
        stagnation_count,
        int(current_mutation_mean),
    )


def GLS(graph, population_size, stopping_crit=None, time_limit=None):
    """
    Genetic Local Search.  Includes a time limit.
    """
    num_starts = 5
    population = [MLS(graph, num_starts) for _ in range(population_size)]
    fitness_values = [fitness_function(sol, graph) for sol in population]
    best_fit = min(fitness_values)
    best_solution = population[fitness_values.index(best_fit)]
    generation_without_improvement = 0
    start_time = time.perf_counter()  # Record start time

    for _ in range(10000):  # Large iteration limit
        if time_limit is not None and time.perf_counter() - start_time >= time_limit:
            break  # Exit if time limit reached

        if generation_without_improvement >= stopping_crit:
            break
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        child = crossover(parent1, parent2)
        child_opt, _ = fm_heuristic(child, graph)
        fit_child = fitness_function(child_opt, graph)
        worst_fit = max(fitness_values)
        worst_index = fitness_values.index(worst_fit)
        if fit_child <= worst_fit:
            population[worst_index] = child_opt.copy()
            fitness_values[worst_index] = fit_child
            if fit_child < best_fit:
                best_fit = fit_child
                best_solution = child_opt.copy()
                generation_without_improvement = 0
            else:
                generation_without_improvement += 1
        else:
            generation_without_improvement += 1
    return best_solution


def get_hamming_distance(parent1, parent2):
    """Compute the Hamming distance between two solutions."""
    return sum(1 for a, b in zip(parent1, parent2) if a != b)


def balance_child(child):
    """
    Adjusts a child solution to restore balance.
    Ensures that the total number of ones is equal to NUM_VERTICES/2.
    """
    target = len(child) // 2
    current_ones = sum(child)
    indices = list(range(len(child)))
    if current_ones > target:
        ones_indices = [i for i in indices if child[i] == 1]
        to_flip = random.sample(ones_indices, current_ones - target)
        for i in to_flip:
            child[i] = 0
    elif current_ones < target:
        zeros_indices = [i for i in indices if child[i] == 0]
        to_flip = random.sample(zeros_indices, target - current_ones)
        for i in to_flip:
            child[i] = 1
    return child


def crossover(parent1, parent2):
    """
    Performs uniform crossover that respects the balance constraint.
    Steps:
      1. Compute Hamming distance; if > len(parent)/2, invert parent1.
      2. For each position, if parents agree, copy that bit.
      3. For disagreement, choose randomly.
      4. Finally, adjust the child to restore balance.
    """
    child = []
    hd = get_hamming_distance(parent1, parent2)
    if hd > len(parent1) / 2:
        parent1 = [1 - bit for bit in parent1]
    for i in range(len(parent1)):
        if parent1[i] == parent2[i]:
            child.append(parent1[i])
        else:
            child.append(random.choice([parent1[i], parent2[i]]))
    child = balance_child(child)
    return child


# --- Experiment Runners ---


def run_MLS_experiment(graph):
    """Runs MLS (fixed FM passes) EXPERIMENT_RUNS times."""
    num_starts = 5
    results = []
    for i in range(EXPERIMENT_RUNS):
        start_time = time.perf_counter()
        best_sol = MLS(graph, num_starts)  # Calls MLS, which now uses DLL
        end_time = time.perf_counter()
        fit = fitness_function(best_sol, graph)
        comp_time = end_time - start_time
        solution_str = "".join(str(bit) for bit in best_sol)
        results.append(
            (i, solution_str, fit, comp_time, "", "", "")
        )  # Consistent format
        print(f"MLS run {i}: Fitness={fit}")
    return results


def run_simple_ILS_experiment(graph):
    """Runs simple ILS (accepting only improvements)."""
    mutation_sizes = [1, 2, 3, 4, 5, 10, 25, 40, 50, 60, 75, 100, 200]
    results = []
    for mutation_size in mutation_sizes:
        for i in range(EXPERIMENT_RUNS):
            start_time = time.perf_counter()
            best_sol, unchanged_count = ILS(
                graph, generate_random_solution(), mutation_size
            )
            end_time = time.perf_counter()
            fit = fitness_function(best_sol, graph)
            comp_time = end_time - start_time
            solution_str = "".join(str(bit) for bit in best_sol)
            results.append(
                (i, solution_str, fit, comp_time, unchanged_count, mutation_size, "")
            )
            print(f"Simple_ILS run {i}: Fitness={fit}, Unchanged={unchanged_count}")
    return results


def run_GLS_experiment(graph):
    """Runs GLS."""
    stopping_crits = [1, 2]  # Different stopping criteria
    results = []
    for stopping_crit in stopping_crits:
        for i in range(EXPERIMENT_RUNS):
            start_time = time.perf_counter()
            best_sol = GLS(graph, POPULATION_SIZE, stopping_crit)
            end_time = time.perf_counter()
            fit = fitness_function(best_sol, graph)
            comp_time = end_time - start_time
            solution_str = "".join(str(bit) for bit in best_sol)
            results.append((i, solution_str, fit, comp_time, "", "", stopping_crit))
            print(f"GLS run {i} (stop={stopping_crit}): Fitness={fit}")
    return results


def run_ILS_adaptive_experiment(graph):
    """Runs the ILS_adaptive experiment with parameters INSIDE ILS_adaptive."""
    results = []
    for i in range(EXPERIMENT_RUNS):
        start_time = time.perf_counter()
        (
            best_sol,
            best_fit,
            mutation_history,
            iterations,
            final_stagnation_count,
            final_mutation_size,
        ) = ILS_adaptive(
            graph,
            generate_random_solution(),  # Only pass graph and initial solution
        )
        end_time = time.perf_counter()
        comp_time = end_time - start_time
        fit = fitness_function(best_sol, graph)
        solution_str = "".join(str(bit) for bit in best_sol)
        results.append(
            (
                i,
                solution_str,
                fit,
                comp_time,
                final_stagnation_count,
                final_mutation_size,
                mutation_history,
            )
        )
        print(
            f"ILS_Adaptive run {i}: Fitness={fit}, FinalMutSize={final_mutation_size}, Unchanged={final_stagnation_count}, Time={comp_time:.4f}"
        )
    return results


def run_ILS_annealing_experiment(graph):
    """Runs ILS with annealing for various mutation sizes."""
    mutation_sizes = [1, 2, 3, 4, 5, 10, 25, 40, 50, 60, 75, 100, 200]
    results = []
    for mutation_size in mutation_sizes:
        for i in range(EXPERIMENT_RUNS):
            start_time = time.perf_counter()
            best_sol, unchanged_count = ILS_annealing(
                graph, generate_random_solution(), mutation_size
            )
            end_time = time.perf_counter()
            fit = fitness_function(best_sol, graph)
            comp_time = end_time - start_time
            solution_str = "".join(str(bit) for bit in best_sol)
            results.append(
                (i, solution_str, fit, comp_time, unchanged_count, mutation_size, "")
            )  # Consistent format
            print(
                f"ILS_Annealing run {i} (mut={mutation_size}): Fitness={fit}, Unchanged={unchanged_count}"
            )
    return results


# --- Runtime experiments (equal clock time based) ---


def run_MLS_runtime_experiment(graph, time_limit):
    results = []
    i = 0
    global_start_time = time.perf_counter()  # Global start time

    best_sol = MLS(graph, num_starts=100000, time_limit=time_limit)
    current_time = time.perf_counter()
    comp_time = current_time - global_start_time  # Cumulative elapsed time
    fit = fitness_function(best_sol, graph)
    solution_str = "".join(str(bit) for bit in best_sol)
    results.append((i, solution_str, fit, comp_time, "", "", ""))
    print(f"MLS runtime run {i}: Total Time={comp_time:.6f}, Fitness={fit}")
    return results


def run_ILS_runtime_experiment(graph, time_limit, mutation_size):
    results = []
    i = 0
    global_start_time = time.perf_counter()  # Global start time
    best_sol, unchanged_count = ILS(
        graph, generate_random_solution(), mutation_size, time_limit=time_limit
    )
    current_time = time.perf_counter()
    comp_time = current_time - global_start_time  # Cumulative elapsed time
    fit = fitness_function(best_sol, graph)
    solution_str = "".join(str(bit) for bit in best_sol)
    results.append(
        (i, solution_str, fit, comp_time, unchanged_count, mutation_size, "")
    )
    print(
        f"ILS runtime run {i}: Total Time={comp_time:.6f}, Fitness={fit}, Unchanged={unchanged_count}"
    )

    return results


def run_GLS_runtime_experiment(graph, time_limit, stopping_crit):
    results = []

    i = 0
    global_start_time = time.perf_counter()  # Global start time
    best_sol = GLS(graph, POPULATION_SIZE, stopping_crit, time_limit=time_limit)
    current_time = time.perf_counter()
    comp_time = current_time - global_start_time  # Cumulative elapsed time
    fit = fitness_function(best_sol, graph)
    solution_str = "".join(str(bit) for bit in best_sol)
    results.append((i, solution_str, fit, comp_time, "", "", stopping_crit))
    print(f"GLS runtime run {i}: Total Time={comp_time:.6f}, Fitness={fit}")

    return results


def run_ILS_adaptive_runtime_experiment(graph, time_limit):
    """Runs ILS_adaptive for a specified clock time using cumulative timing."""
    results = []
    i = 0
    global_start_time = time.perf_counter()  # Global start time

    (
        best_sol,
        best_fit,
        mutation_history,
        iterations,
        final_stagnation_count,
        final_mut_mean,
    ) = ILS_adaptive(graph, generate_random_solution(), time_limit=time_limit)
    current_time = time.perf_counter()
    comp_time = current_time - global_start_time  # Cumulative elapsed time
    fit = fitness_function(best_sol, graph)
    solution_str = "".join(str(bit) for bit in best_sol)
    results.append(
        (
            i,
            solution_str,
            fit,
            comp_time,
            final_stagnation_count,
            final_mut_mean,
            mutation_history,
        )
    )
    print(
        f"ILS_Adaptive runtime run {i}: Total Time={comp_time:.6f}, Fitness={fit}, FinalMutSize={final_mut_mean}, Unchanged={final_stagnation_count}"
    )

    return results


def run_ILS_annealing_runtime_experiment(graph, time_limit, mutation_size):
    results = []
    i = 0
    global_start_time = time.perf_counter()  # Global start time

    best_sol, unchanged_count = ILS_annealing(
        graph, generate_random_solution(), mutation_size, time_limit=time_limit
    )
    current_time = time.perf_counter()
    comp_time = current_time - global_start_time  # Cumulative elapsed time
    fit = fitness_function(best_sol, graph)
    solution_str = "".join(str(bit) for bit in best_sol)
    results.append(
        (i, solution_str, fit, comp_time, unchanged_count, mutation_size, "")
    )
    print(
        f"ILS_Annealing runtime run {i}: Total Time={comp_time:.6f}, Fitness={fit}, Unchanged={unchanged_count}"
    )

    return results


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    graph = read_graph_data(os.path.join(script_dir, "Graph500.txt"))

    # -----------------------------
    # Parameter experiments (fixed FM passes)
    # -----------------------------
    mls_results = run_MLS_experiment(graph)
    ils_annealing_results = run_ILS_annealing_experiment(graph)
    simple_ils_results = run_simple_ILS_experiment(graph)
    gls_results = run_GLS_experiment(graph)
    ils_adaptive_results = run_ILS_adaptive_experiment(graph)

    # -----------------------------
    # Runtime experiments (equal clock time)
    # -----------------------------

    TIME_LIMIT = 10  # Set the time limit to 60 seconds

    mls_runtime = run_MLS_runtime_experiment(graph, TIME_LIMIT)
    ils_annealing_runtime = run_ILS_annealing_runtime_experiment(
        graph, TIME_LIMIT, mutation_size=50
    )
    ils_simple_runtime = run_ILS_runtime_experiment(graph, TIME_LIMIT, mutation_size=50)
    ils_adaptive_runtime = run_ILS_adaptive_runtime_experiment(graph, TIME_LIMIT)
    gls_runtime = run_GLS_runtime_experiment(graph, TIME_LIMIT, 2)

    # -----------------------------
    # Combine all results into a single CSV file.
    # -----------------------------
    all_rows = []

    # Parameter Experiments:
    for run, sol, fitness, comp_time, _, _, _ in mls_results:
        all_rows.append(
            {
                "Experiment": "MLS",
                "Run": run,
                "Mutation_Size": "",
                "Stopping_Crit": "",
                "Fitness": fitness,
                "Comp_Time": comp_time,
                "Unchanged_Count": "",
                "Solution": sol,
            }
        )
    for (
        run,
        sol,
        fitness,
        comp_time,
        unchanged_count,
        mutation_size,
        _,
    ) in ils_annealing_results:
        all_rows.append(
            {
                "Experiment": "ILS_Annealing",
                "Run": run,
                "Mutation_Size": mutation_size,
                "Stopping_Crit": "",
                "Fitness": fitness,
                "Comp_Time": comp_time,
                "Unchanged_Count": unchanged_count,
                "Solution": sol,
            }
        )

    for (
        run,
        sol,
        fitness,
        comp_time,
        unchanged_count,
        mutation_size,
        _,
    ) in simple_ils_results:
        all_rows.append(
            {
                "Experiment": "Simple_ILS",
                "Run": run,
                "Mutation_Size": mutation_size,
                "Stopping_Crit": "",
                "Fitness": fitness,
                "Comp_Time": comp_time,
                "Unchanged_Count": unchanged_count,
                "Solution": sol,
            }
        )
    for run, sol, fitness, comp_time, _, _, stopping_crit in gls_results:
        all_rows.append(
            {
                "Experiment": "GLS",
                "Run": run,
                "Mutation_Size": "",
                "Stopping_Crit": stopping_crit,
                "Fitness": fitness,
                "Comp_Time": comp_time,
                "Unchanged_Count": "",
                "Solution": sol,
            }
        )
    for (
        run,
        sol,
        fitness,
        comp_time,
        unchanged_count,
        final_mutation_size,
        _,
    ) in ils_adaptive_results:
        all_rows.append(
            {
                "Experiment": "ILS_Adaptive",
                "Run": run,
                "Mutation_Size": final_mutation_size,
                "Stopping_Crit": "",
                "Fitness": fitness,
                "Comp_Time": comp_time,
                "Unchanged_Count": unchanged_count,
                "Solution": sol,
            }
        )

    # Runtime Experiments:
    for run, sol, fitness, comp_time, _, _, _ in mls_runtime:
        all_rows.append(
            {
                "Experiment": "MLS_Runtime",
                "Run": run,
                "Mutation_Size": "",
                "Stopping_Crit": "",
                "Fitness": fitness,
                "Comp_Time": comp_time,
                "Unchanged_Count": "",
                "Solution": sol,
            }
        )
    for (
        run,
        sol,
        fitness,
        comp_time,
        unchanged_count,
        mutation_size,
        _,
    ) in ils_annealing_runtime:
        all_rows.append(
            {
                "Experiment": "ILS_Annealing_Runtime",
                "Run": run,
                "Mutation_Size": mutation_size,
                "Stopping_Crit": "",
                "Fitness": fitness,
                "Comp_Time": comp_time,
                "Unchanged_Count": unchanged_count,
                "Solution": sol,
            }
        )
    for (
        run,
        sol,
        fitness,
        comp_time,
        unchanged_count,
        mutation_size,
        _,
    ) in ils_simple_runtime:
        all_rows.append(
            {
                "Experiment": "ILS_Simple_Runtime",
                "Run": run,
                "Mutation_Size": mutation_size,
                "Stopping_Crit": "",
                "Fitness": fitness,
                "Comp_Time": comp_time,
                "Unchanged_Count": unchanged_count,
                "Solution": sol,
            }
        )
    for run, sol, fitness, comp_time, _, _, stopping_crit in gls_runtime:
        all_rows.append(
            {
                "Experiment": "GLS_Runtime",
                "Run": run,
                "Mutation_Size": "",
                "Stopping_Crit": stopping_crit,
                "Fitness": fitness,
                "Comp_Time": comp_time,
                "Unchanged_Count": "",
                "Solution": sol,
            }
        )

    for (
        run,
        sol,
        fitness,
        comp_time,
        unchanged_count,
        final_mutation_size,
        _,
    ) in ils_adaptive_runtime:
        all_rows.append(
            {
                "Experiment": "ILS_Adaptive_Runtime",
                "Run": run,
                "Mutation_Size": final_mutation_size,
                "Stopping_Crit": "",
                "Fitness": fitness,
                "Comp_Time": comp_time,
                "Unchanged_Count": unchanged_count,
                "Solution": sol,
            }
        )

    df_experiments = pd.DataFrame(
        all_rows,
        columns=[
            "Experiment",
            "Run",
            "Mutation_Size",
            "Stopping_Crit",
            "Fitness",
            "Comp_Time",
            "Unchanged_Count",
            "Solution",
        ],
    )

    df_experiments.to_csv("experiment_results.csv", index=False)
    print("All experiment results saved in 'experiment_results.csv'.")
