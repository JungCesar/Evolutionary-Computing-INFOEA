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


# --- Node Class (as provided by user) ---
class Node:
    __slots__ = ("gain", "neighbors", "prev", "next", "partition", "moved")

    def __init__(self, gain=0, neighbors=None, prev=-1, next=-1, partition=0):
        self.gain = gain
        self.neighbors = neighbors if neighbors is not None else []
        self.prev = prev
        self.next = next
        self.partition = partition
        self.moved = False

# --- Node Class (required by FM) ---
class Node:
    __slots__ = ("gain", "neighbors", "prev", "next", "partition", "moved")

    def __init__(self, gain=0, neighbors=None, prev=-1, next=-1, partition=0):
        self.gain = gain
        # Ensure neighbors is a set for efficient lookups if needed later
        self.neighbors = set(neighbors) if neighbors is not None else set()
        self.prev = prev
        self.next = next
        self.partition = partition
        self.moved = False

# --- Helper: Calculate Gain (required by FM) ---
def calculate_gain(node_idx, solution, graph):
    """ Calculates the gain for moving node_idx to the opposite partition. """
    gain = 0
    n = len(solution)
    if not (0 <= node_idx < n and node_idx < len(graph)):
         # print(f"Warning: Invalid node_idx {node_idx} in calculate_gain.")
         return 0 # Return neutral gain on error

    current_partition = solution[node_idx]
    # Ensure graph[node_idx] is iterable
    neighbors = graph[node_idx] if hasattr(graph[node_idx], '__iter__') else []

    for neighbor in neighbors:
        if not (0 <= neighbor < n):
             # print(f"Warning: Invalid neighbor index {neighbor} for node {node_idx}.")
             continue # Skip invalid neighbor

        if solution[neighbor] == current_partition:
            gain -= 1 # Internal edge decreases gain
        else:
            gain += 1 # External edge increases gain
    return gain

# --- Helper: Insert node into gain bucket list (required by FM) ---
def insert_node(nodes, gain_lists, max_gain, node_index, max_degree_adj):
    """ Inserts node_index into the correct gain bucket's linked list. """
    # Add basic safety checks
    if not (0 <= node_index < len(nodes)): return

    node = nodes[node_index]
    gain = node.gain
    part = node.partition
    gain_index = gain + max_degree_adj # Calculate index in the gain array

    if not (0 <= part < len(gain_lists) and 0 <= gain_index < len(gain_lists[part])):
         # print(f"Error: Invalid index in insert_node. Node: {node_index}, Part: {part}, Gain: {gain}, MaxDegree: {max_degree_adj}, Index: {gain_index}")
         return

    node.next = gain_lists[part][gain_index] # Point to the current head
    node.prev = -1 # This node is the new head

    head_node_idx = gain_lists[part][gain_index]
    if head_node_idx != -1: # If list wasn't empty
         if 0 <= head_node_idx < len(nodes): # Check index validity
             nodes[head_node_idx].prev = node_index # Old head points back to new node
         # else: print(f"Error: Invalid head node index {head_node_idx} in insert_node.")

    gain_lists[part][gain_index] = node_index # Update head pointer in gain array

    if gain > max_gain[part]:
        max_gain[part] = gain

# --- Helper: Remove node from gain bucket list (required by FM) ---
def remove_node(nodes, gain_lists, node_index, max_degree_adj):
    """ Removes node_index from its gain bucket's linked list. """
    if not (0 <= node_index < len(nodes)): return

    node = nodes[node_index]
    gain = node.gain
    part = node.partition
    gain_index = gain + max_degree_adj

    if not (0 <= part < len(gain_lists) and 0 <= gain_index < len(gain_lists[part])):
        # print(f"Error: Invalid index in remove_node. Node: {node_index}, Part: {part}, Gain: {gain}, MaxDegree: {max_degree_adj}, Index: {gain_index}")
        return

    prev_node_idx = node.prev
    next_node_idx = node.next

    # Update next pointer of the previous node (or head pointer)
    if prev_node_idx != -1:
         if 0 <= prev_node_idx < len(nodes):
             nodes[prev_node_idx].next = next_node_idx
         # else: print(f"Error: Invalid prev_node index {prev_node_idx} in remove_node.")
    else:
        # This node was the head of the list
        gain_lists[part][gain_index] = next_node_idx

    # Update previous pointer of the next node
    if next_node_idx != -1:
        if 0 <= next_node_idx < len(nodes):
             nodes[next_node_idx].prev = prev_node_idx
        # else: print(f"Error: Invalid next_node index {next_node_idx} in remove_node.")

    node.prev = -1
    node.next = -1


# --- Helper: Update max gain pointer (required by FM) ---
def update_max_gain(gain_lists, max_gain, max_degree_adj):
    """ Finds the new highest non-empty gain bucket. """
    min_gain = -max_degree_adj
    for part in range(2):
        current_max = max_gain[part]
        while current_max >= min_gain:
            gain_index = current_max + max_degree_adj
            if not (0 <= gain_index < len(gain_lists[part])):
                 # print(f"Error: Invalid index in update_max_gain search. Part: {part}, Gain: {current_max}, MaxDegree: {max_degree_adj}, Index: {gain_index}")
                 current_max = min_gain - 1 # Force exit
                 break

            if gain_lists[part][gain_index] != -1:
                max_gain[part] = current_max
                break
            current_max -= 1
        else:
            # Loop finished without break, all buckets empty or invalid gain
            max_gain[part] = min_gain - 1 # Indicate no nodes available


# --- Helper: Basic Balance Function (required by FM if input unbalanced / fallback) ---
def balance_child(child_list):
    """ Fallback basic balancer if needed """
    n = len(child_list)
    if n == 0: return child_list
    target_ones = n // 2
    try:
        # Ensure list contains only 0s and 1s before counting
        child_list_clean = [int(bit) for bit in child_list if int(bit) in (0, 1)]
        if len(child_list_clean) != n:
             print("Warning: balance_child input contained non-binary values.")
             # Decide how to handle - returning original might be safest
             return child_list

        current_ones = child_list_clean.count(1)
    except (TypeError, ValueError):
         print("Error: Non-numeric type found during balance_child count.")
         return child_list # Return original on error

    diff = current_ones - target_ones
    indices = list(range(n)) # Get indices once

    if diff > 0: # Too many ones
        ones_indices = [i for i in indices if child_list_clean[i] == 1]
        # Check if we have enough nodes to flip
        if len(ones_indices) >= diff:
             indices_to_flip = random.sample(ones_indices, diff)
             for i in indices_to_flip: child_list_clean[i] = 0
        else:
             print(f"Error: balance_child logic error - Cannot flip {diff} ones, only {len(ones_indices)} found.")
             # Cannot balance, return original or potentially raise error
             return child_list
    elif diff < 0: # Too many zeros
        num_to_flip = -diff
        zeros_indices = [i for i in indices if child_list_clean[i] == 0]
        # Check if we have enough nodes to flip
        if len(zeros_indices) >= num_to_flip:
             indices_to_flip = random.sample(zeros_indices, num_to_flip)
             for i in indices_to_flip: child_list_clean[i] = 1
        else:
             print(f"Error: balance_child logic error - Cannot flip {num_to_flip} zeros, only {len(zeros_indices)} found.")
             # Cannot balance, return original or potentially raise error
             return child_list

    return child_list_clean


# --- Helper: Fitness Function (required by FM) ---
def fitness_function(solution, graph):
    n = len(solution)
    if n == 0: return 0
    total = 0
    for i in range(n):
        # Add checks similar to calculate_gain for robustness
        if i < len(graph) and hasattr(graph[i], '__iter__') and i < len(solution):
             part_i = solution[i]
             for nb in graph[i]:
                 if 0 <= nb < n:
                     if part_i != solution[nb]:
                         total += 1
                 # else: Optional warning for invalid neighbor index
        # else: Optional warning for invalid node index
    return total // 2

# --- Main FM Heuristic Function ---
def fm_heuristic(initial_solution, graph, max_passes=FM_PASSES): # Default max_passes to 10 as often used
    """
    Fiduccia-Mattheyses heuristic based on Prac2.pdf description.
    Ensures returned solution is balanced by discarding passes if best state is imbalanced.
    Uses gain buckets for efficiency.

    Args:
        initial_solution (list): Initial balanced partition (0s and 1s).
        graph (list): Adjacency list/set representation.
        max_passes (int): Max outer passes (rebuild lists and find best prefix).

    Returns:
        tuple: (optimized_balanced_solution, passes_performed)
    """
    n = len(initial_solution)
    if n == 0: return initial_solution.copy(), 0
    target_part_size = n // 2

    working_solution = initial_solution.copy()
    if working_solution.count(0) != target_part_size:
        print("Warning: fm_heuristic_new received imbalanced input. Balancing using balance_child.")
        working_solution = balance_child(working_solution) # Ensure start is balanced

    # --- Pre-calculate Max Degree ---
    max_degree = 0
    for i in range(n):
        if i < len(graph) and hasattr(graph[i], '__iter__'):
             max_degree = max(max_degree, len(graph[i]))
    max_degree_adj = max(1, max_degree)
    gain_list_size = 2 * max_degree_adj + 1
    min_gain = -max_degree_adj

    nodes = [Node() for _ in range(n)]

    best_solution_overall = working_solution.copy()
    best_overall_cut_size = fitness_function(best_solution_overall, graph)

    # --- Outer Loop (Passes) ---
    improved_in_cycle = True # Assume improvement possible initially
    passes_done = 0
    while improved_in_cycle and passes_done < max_passes:
        #improved_in_cycle = False # Reset for this pass
        passes_done += 1
        print(passes_done)
        solution_at_pass_start = working_solution.copy()

        # --- Initialize for the pass ---
        gain_lists = [[-1] * gain_list_size for _ in range(2)]
        max_gain = [min_gain - 1] * 2

        indices = list(range(n))
        random.shuffle(indices)
        for i in indices:
            nodes[i].moved = False
            nodes[i].partition = working_solution[i]
            nodes[i].neighbors = graph[i] if i < len(graph) and graph[i] else set()
            nodes[i].gain = calculate_gain(i, working_solution, graph)
            if min_gain <= nodes[i].gain <= max_degree_adj:
                 insert_node(nodes, gain_lists, max_gain, i, max_degree_adj)
            # else: Warning about gain out of bounds could be added

        # --- Inner Loop (Sequence of N moves) ---
        moves_sequence = []
        cumulative_gains = [0.0] # Gain relative to start of pass

        for k in range(n):
            best_node_to_move = -1
            selected_gain = -float('inf')

            # Find best valid (unlocked, balance-permitting) node
            for part_from in range(2):
                current_max_g = max_gain[part_from]
                while current_max_g >= min_gain:
                     gain_idx = current_max_g + max_degree_adj
                     if not (0 <= gain_idx < gain_list_size): break

                     node_idx_in_bucket = gain_lists[part_from][gain_idx]
                     found_candidate_this_bucket = False
                     while node_idx_in_bucket != -1:
                         if not nodes[node_idx_in_bucket].moved:
                             # Relaxed Balance Check (check counts *before* the move)
                             current_part0_count = working_solution.count(0) # Count in current state
                             is_move_valid = False
                             if part_from == 0: # Moving from 0
                                 if current_part0_count > target_part_size - 1: is_move_valid = True
                             else: # Moving from 1
                                 if (n - current_part0_count) > target_part_size - 1: is_move_valid = True

                             if is_move_valid:
                                 best_node_to_move = node_idx_in_bucket
                                 selected_gain = nodes[best_node_to_move].gain
                                 found_candidate_this_bucket = True
                                 break
                         node_idx_in_bucket = nodes[node_idx_in_bucket].next
                     if found_candidate_this_bucket: break
                     current_max_g -= 1
                if best_node_to_move != -1: break

            if best_node_to_move == -1: break # No more valid moves

            # --- Perform the move ---
            node_to_move_idx = best_node_to_move
            original_partition = nodes[node_to_move_idx].partition

            nodes[node_to_move_idx].moved = True
            remove_node(nodes, gain_lists, node_to_move_idx, max_degree_adj)
            working_solution[node_to_move_idx] = 1 - original_partition # Update working solution
            nodes[node_to_move_idx].partition = working_solution[node_to_move_idx]

            moves_sequence.append(node_to_move_idx)
            cumulative_gains.append(cumulative_gains[-1] + selected_gain)

            # Update gains of unlocked neighbors
            for neighbor_idx in nodes[node_to_move_idx].neighbors:
                if not nodes[neighbor_idx].moved:
                    # Delta depends on neighbor's partition relative to moved node's *original* partition
                    gain_delta = 2 if working_solution[neighbor_idx] == original_partition else -2

                    neighbor_gain_before = nodes[neighbor_idx].gain
                    neighbor_gain_index_before = neighbor_gain_before + max_degree_adj
                    if 0 <= neighbor_gain_index_before < gain_list_size:
                        remove_node(nodes, gain_lists, neighbor_idx, max_degree_adj)

                    nodes[neighbor_idx].gain += gain_delta
                    neighbor_gain_index_after = nodes[neighbor_idx].gain + max_degree_adj
                    if 0 <= neighbor_gain_index_after < gain_list_size:
                        insert_node(nodes, gain_lists, max_gain, neighbor_idx, max_degree_adj)

            update_max_gain(gain_lists, max_gain, max_degree_adj)
        # --- End of Inner Loop ---

        # --- Find Best Prefix and Rollback ---
        if not moves_sequence: continue # No moves, no improvement this pass

        best_k = np.argmax(cumulative_gains) # Index in cumulative_gains list
        best_num_moves = best_k

        # Rollback: Reconstruct the solution state after 'best_num_moves'
        solution_after_rollback = solution_at_pass_start.copy()
        for i in range(best_num_moves):
            node_idx = moves_sequence[i]
            solution_after_rollback[node_idx] = 1 - solution_after_rollback[node_idx]

        # --- Check Balance and Decide ---
        final_solution_this_pass = solution_after_rollback # Tentative solution
        current_cut_size = fitness_function(final_solution_this_pass, graph)

        if final_solution_this_pass.count(0) != target_part_size:
            # Best state was imbalanced. Discard pass results.
            working_solution = solution_at_pass_start.copy()
            # improved_in_cycle remains False
        else:
            # Best state was balanced. Update overall best if better.
            if current_cut_size < best_overall_cut_size:
                 best_overall_cut_size = current_cut_size
                 best_solution_overall = final_solution_this_pass.copy()
                 improved_in_cycle = True # Improvement found!
            # Set working solution for next pass
            working_solution = final_solution_this_pass.copy()

    # --- End of Outer Loop (Passes) ---

    # Final check (should ideally pass now)
    if best_solution_overall.count(0) != target_part_size:
         print(f"CRITICAL ERROR: fm_heuristic_new final solution is not balanced!")

    return best_solution_overall, passes_done



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


def MLS(graph, num_starts, max_total_passes=None, time_limit=None):
    """
    Multi-start Local Search with cumulative pass limit and time limit.

    Args:
        graph (list): Adjacency list/set representation.
        num_starts (int): Number of random restarts.
        max_total_passes (int, optional): Cumulative FM pass limit for the entire run.
        time_limit (float, optional): Wall-clock time limit in seconds.

    Returns:
        tuple: (best_solution_found, total_passes_consumed)
               Returns (None, total_passes_consumed) if time/pass limit reached before finding any solution.
    """
    best_solution = None
    best_fit = float("inf")
    start_time = time.perf_counter()
    total_passes_consumed = 0

    for i in range(num_starts):
        current_time = time.perf_counter()
        # --- Check Limits BEFORE starting a new FM run ---
        if time_limit is not None and current_time - start_time >= time_limit:
            print(f"  MLS: Time limit reached after {i} starts.")
            break
        if max_total_passes is not None and total_passes_consumed >= max_total_passes:
            print(f"  MLS: Pass limit reached after {i} starts.")
            break

        # --- Prepare for FM call ---
        sol = generate_random_solution() # Generate new initial solution
        remaining_passes = float('inf') # Default if no pass limit
        if max_total_passes is not None:
            remaining_passes = max_total_passes - total_passes_consumed
            if remaining_passes <= 0: # Should be caught by check above, but safety
                 print(f"  MLS: No passes remaining before start {i+1}.")
                 break
            # FM heuristic internally stops after max_passes_per_call passes
            # OR when no improvement is found in a pass. Pass the remaining budget.

        # --- Call FM Heuristic ---
        try:
            # Assuming fm_heuristic_new is the correct function name
            local_opt, passes_this_call = fm_heuristic(
                sol,
                graph,
                max_passes=int(remaining_passes) # Pass remaining budget as the limit
            )
            total_passes_consumed += passes_this_call # Add passes actually used

            if local_opt is not None:
                fit = fitness_function(local_opt, graph)
                if fit < best_fit:
                    best_fit = fit
                    best_solution = local_opt.copy()
            # else: fm_heuristic potentially returned None (e.g., if input was bad)

        except Exception as e:
            print(f"  Error during fm_heuristic_new call in MLS start {i+1}: {e}")
            # Decide how to handle: continue, break? Continuing for now.
            continue

    # --- End of Loop ---
    final_time = time.perf_counter()
    print(f"  MLS finished. Starts completed: {i+1}/{num_starts}. Total Passes: {total_passes_consumed}. Time: {final_time-start_time:.4f}s. Best Fitness: {best_fit if best_solution else 'N/A'}")

    # Return None if no solution was ever found (e.g., limits too tight)
    return best_solution, total_passes_consumed


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
    initial_mutation_mean = 15
    mutation_std_dev = 2
    decay_factor = 0.98
    stagnation_threshold = 50
    max_iterations = 1000  # Keep a max iteration count, even with time limit
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
    Genetic Local Search. Includes a time limit check during initialization and evolution.
    """
    start_time = time.perf_counter() # Start timer BEFORE initialization
    num_starts = 5 # As used in your original code for initializing population members
    population = []
    fitness_values = []

    # --- Population Initialization with Time Limit Check ---
    print(f"  GLS Initializing population (size {population_size})...")
    for i in range(population_size):
        # Check time limit before starting the next MLS run for initialization
        if time_limit is not None and time.perf_counter() - start_time >= time_limit:
            print(f"  GLS Time limit reached during population initialization (after {i} individuals).")
            # If time runs out during init, we might have an incomplete population.
            # Depending on requirements, either proceed with smaller pop or handle error.
            # For now, we proceed with the partially initialized population if any.
            break

        # Run MLS for one individual - pass a portion of the remaining time?
        # Or let MLS run without time limit for init? Simpler: let MLS run normally for init.
        # The outer check will eventually stop the initialization loop.
        try:
             # We do not pass the time_limit to the MLS calls during initialization
             # as the main GLS time limit check handles stopping the overall process.
             sol = MLS(graph, num_starts)
             if sol is not None: # Ensure MLS returned a valid solution
                 population.append(sol)
                 fitness_values.append(fitness_function(sol, graph))
             else:
                 print(f"  Warning: MLS during GLS init (individual {i+1}) returned None.")
                 # Optionally, retry or break if this happens often
        except Exception as e:
            print(f"  Error during MLS in GLS init (individual {i+1}): {e}")
            # Decide how to handle: continue, break, etc. Continuing for now.
            continue

    # Check if population was initialized at all
    if not population:
        print("  GLS Error: Population initialization failed or stopped early by time limit. Returning None.")
        # Return a sensible default or None if no solution could be generated in time
        return generate_random_solution() # Or return None

    print(f"  GLS Initialization complete. Population size: {len(population)}. Time elapsed: {time.perf_counter() - start_time:.4f}s")

    # Find initial best from the generated population
    best_fit = min(fitness_values)
    best_solution = population[fitness_values.index(best_fit)]
    generation_without_improvement = 0

    # --- Evolutionary Loop with Time Limit Check ---
    # Use a large number, rely on time_limit or stopping_crit
    MAX_GENERATIONS = 100000
    print("  GLS Starting evolutionary loop...")
    for gen in range(MAX_GENERATIONS):
        # Check time limit at the start of each generation
        if time_limit is not None and time.perf_counter() - start_time >= time_limit:
            print(f"  GLS Time limit reached during evolution at generation {gen}.")
            break # Exit evolutionary loop

        # Check stopping criterion based on generations without improvement
        # Make sure stopping_crit is not None if used
        if stopping_crit is not None and generation_without_improvement >= stopping_crit:
            print(f"  GLS Stopping criterion met at generation {gen}.")
            break # Exit evolutionary loop

        # Standard GLS steps
        parent1_idx = random.randrange(len(population))
        parent2_idx = random.randrange(len(population))
        parent1 = population[parent1_idx]
        parent2 = population[parent2_idx]
        child = crossover(parent1, parent2)

        # Run FM heuristic on the child
        # We assume fm_heuristic itself doesn't need a time limit here,
        # as the main loop's time check is the primary control.
        try:
            child_opt, _ = fm_heuristic(child, graph)
            if child_opt is None: # Handle potential issues in fm_heuristic if needed
                 print(f"  Warning: fm_heuristic returned None in generation {gen}. Skipping replacement.")
                 continue
            fit_child = fitness_function(child_opt, graph)
        except Exception as e:
             print(f"  Error during fm_heuristic or fitness calculation in generation {gen}: {e}")
             continue # Skip this generation's replacement

        # Find worst individual for replacement
        worst_fit_idx = np.argmax(fitness_values) # More efficient way to find index of max
        worst_fit = fitness_values[worst_fit_idx]

        # Replacement (Steady-State)
        if fit_child <= worst_fit: # Replace if better or equal
            population[worst_fit_idx] = child_opt.copy()
            fitness_values[worst_fit_idx] = fit_child
            if fit_child < best_fit:
                best_fit = fit_child
                best_solution = child_opt.copy()
                generation_without_improvement = 0
                # Optional: print improvement
                # print(f"  GLS Gen {gen}: New best fitness {best_fit}")
            else:
                 # Fitness improved vs worst, but not overall best
                 generation_without_improvement += 1
        else:
            # Child was not better than the worst
            generation_without_improvement += 1

    print(f"  GLS Finished. Total time: {time.perf_counter() - start_time:.4f}s. Best fitness: {best_fit}")
    # Return the best solution found within the time limit / stopping criteria
    return best_solution


def get_hamming_distance(parent1, parent2):
    """Compute the Hamming distance between two solutions."""
    return sum(1 for a, b in zip(parent1, parent2) if a != b)


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
    # --- Essential Setup ---
    # Make sure necessary imports like 'time', 'os', 'numpy', 'pandas', 'random' are done above
    # Make sure all required functions like 'read_graph_data', 'MLS', 'ILS', 'GLS',
    # 'fm_heuristic', 'fitness_function', etc., are defined above.

    try:
        # Determine script directory for reliable file access
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the full path to the graph file
        graph_file_path = os.path.join(script_dir, "Graph500.txt")
        # Check if graph file exists before trying to read
        if not os.path.exists(graph_file_path):
             raise FileNotFoundError(f"Graph file not found at: {graph_file_path}")
        graph = read_graph_data(graph_file_path)
        print("Graph data loaded successfully.")
    except Exception as e:
        print(f"Error during setup: {e}")
        print("Please ensure 'Graph500.txt' is in the same directory as the script and functions are defined.")
        exit() # Exit if setup fails
    # === TEMPORARY BLOCK FOR TIMING 10000 FM PASSES ===
    print("\n--- Measuring Time for 10000 FM Passes ---")
    NUM_TIMING_RUNS = 10  # Number of times to measure for stability
    PASSES_TO_TIME = 10000 # The exact number of passes to time
    fm_times = []

    # Ensure graph is loaded from setup above
    if 'graph' not in locals():
         print("Error: Graph not loaded. Cannot run timing.")
    else:
         for i in range(NUM_TIMING_RUNS):
             print(f"  Timing Run {i + 1}/{NUM_TIMING_RUNS}...")
             # Generate a fresh random balanced solution for each timing run
             initial_sol = generate_random_solution()
             if initial_sol.count(0) != len(initial_sol)//2: # Double check balance
                  print("Error: generate_random_solution did not produce balanced solution.")
                  continue

             start_time = time.perf_counter()
             # Call fm_heuristic_new forcing it to run exactly PASSES_TO_TIME
             try:
                  _, passes_executed = fm_heuristic(
                      initial_sol,
                      graph,
                      max_passes=PASSES_TO_TIME,

                  )
                  end_time = time.perf_counter()

                  # Verify if it actually ran the intended number of passes
                  if passes_executed == PASSES_TO_TIME:
                       duration = end_time - start_time
                       fm_times.append(duration)
                       print(f"    Time for {passes_executed} passes: {duration:.6f} seconds")
                  else:
                       print(f"    Warning: FM did not execute expected {PASSES_TO_TIME} passes (executed {passes_executed}). Skipping this run.")

             except Exception as e:
                  print(f"    Error during fm_heuristic_new timing run: {e}")

         # Calculate and print median/average time
         if fm_times:
             median_fm_time = np.median(fm_times)
             average_fm_time = np.mean(fm_times)
             print("\n  --- FM Timing Summary ---")
             print(f"  Recorded Times: {[float(f'{t:.6f}') for t in fm_times]}")
             print(f"  Median Time for {PASSES_TO_TIME} passes: {median_fm_time:.6f} seconds")
             print(f"  Average Time for {PASSES_TO_TIME} passes: {average_fm_time:.6f} seconds")
             print("\n  Use this Median Time as the DYNAMIC_TIME_LIMIT for runtime experiments.")
             # You can assign it directly here if this code stays before the runtime experiments
             # DYNAMIC_TIME_LIMIT = median_fm_time
         else:
             print("\n  No valid FM times were recorded.")

    print("--- End of FM Timing Measurement ---")
    # === END OF TEMPORARY BLOCK ===

    # --- Your original main experiment code follows ---
    # DYNAMIC_TIME_LIMIT = ... # Make sure this is set using the median time above
    # print(f"Using Dynamic Time Limit: {DYNAMIC_TIME_LIMIT:.6f} seconds...")
    # ... (Rest of your main block running parameter and runtime experiments) ...
    # --- Constants ---
    # EXPERIMENT_RUNS = 10 # For pass-based comparison
    # RUNTIME_RUNS = 25    # Number of runs for the time-based comparison
    # Use the median time you measured as the dynamic time limit
    DYNAMIC_TIME_LIMIT = 3.261090 # seconds (based on your median measurement)
    print(f"Using Dynamic Time Limit: {DYNAMIC_TIME_LIMIT:.6f} seconds for runtime experiments.")


    # -----------------------------
    # Parameter experiments (Comparison a)
    # Runs based on your interpretation (e.g., FM_PASSES per FM call)
    # -----------------------------
    print("\n--- Running Parameter Experiments (Comparison a) ---")
    # These run EXPERIMENT_RUNS (10) times each
    mls_results = run_MLS_experiment(graph)
    ils_annealing_results = run_ILS_annealing_experiment(graph)
    simple_ils_results = run_simple_ILS_experiment(graph)
    gls_results = run_GLS_experiment(graph)
    ils_adaptive_results = run_ILS_adaptive_experiment(graph)


    # -----------------------------
    # Runtime experiments (Comparison b - Fixed Time Limit)
    # Run RUNTIME_RUNS (25) times each, using DYNAMIC_TIME_LIMIT
    # -----------------------------
    print(f"\n--- Running Runtime Experiments ({RUNTIME_RUNS} runs each, Time Limit: {DYNAMIC_TIME_LIMIT:.6f}s) ---")

    all_runtime_results_list = [] # Store results directly in the final format

    # --- Define parameters needed for specific runtime algorithms ---
    # Choose best performing mutation size based on parameter experiments if possible
    # Placeholder: Determine these from analysis of simple_ils_results / ils_annealing_results
    best_ils_mutation_size = 50 # Example: Choose based on median fitness from comparison (a)
    best_ils_annealing_mutation_size = 50 # Example: Choose based on median fitness from comparison (a)
    # Choose best GLS stopping criterion based on comparison (a)
    best_gls_stopping_crit = 2 # Example: Choose based on median fitness from comparison (a)
    # ILS_adaptive parameters are internal to the function in your current script

    # Loop for the 25 runtime repetitions
    for i in range(RUNTIME_RUNS):
        print(f"\nRuntime Repetition {i + 1}/{RUNTIME_RUNS}")

        # MLS Runtime
        print("  Running MLS (runtime)...")
        mls_run_res = run_MLS_runtime_experiment(graph, DYNAMIC_TIME_LIMIT)
        for res_tuple in mls_run_res:
             all_runtime_results_list.append({
                 "Experiment": "MLS_Runtime", "Run": i, "Mutation_Size": "", "Stopping_Crit": "",
                 "Fitness": res_tuple[2], "Comp_Time": res_tuple[3], "Unchanged_Count": "", "Solution": res_tuple[1]
             })

        # ILS Annealing Runtime
        print("  Running ILS Annealing (runtime)...")
        ils_a_run_res = run_ILS_annealing_runtime_experiment(graph, DYNAMIC_TIME_LIMIT, best_ils_annealing_mutation_size)
        for res_tuple in ils_a_run_res:
             all_runtime_results_list.append({
                 "Experiment": "ILS_Annealing_Runtime", "Run": i, "Mutation_Size": res_tuple[5], "Stopping_Crit": "",
                 "Fitness": res_tuple[2], "Comp_Time": res_tuple[3], "Unchanged_Count": res_tuple[4], "Solution": res_tuple[1]
             })

        # ILS Simple Runtime
        print("  Running ILS Simple (runtime)...")
        ils_s_run_res = run_ILS_runtime_experiment(graph, DYNAMIC_TIME_LIMIT, best_ils_mutation_size)
        for res_tuple in ils_s_run_res:
             all_runtime_results_list.append({
                 "Experiment": "ILS_Simple_Runtime", "Run": i, "Mutation_Size": res_tuple[5], "Stopping_Crit": "",
                 "Fitness": res_tuple[2], "Comp_Time": res_tuple[3], "Unchanged_Count": res_tuple[4], "Solution": res_tuple[1]
             })

        # GLS Runtime
        print("  Running GLS (runtime)...")
        gls_run_res = run_GLS_runtime_experiment(graph, DYNAMIC_TIME_LIMIT, best_gls_stopping_crit)
        for res_tuple in gls_run_res:
             all_runtime_results_list.append({
                 "Experiment": "GLS_Runtime", "Run": i, "Mutation_Size": "", "Stopping_Crit": res_tuple[6],
                 "Fitness": res_tuple[2], "Comp_Time": res_tuple[3], "Unchanged_Count": "", "Solution": res_tuple[1]
             })

        # ILS Adaptive Runtime
        print("  Running ILS Adaptive (runtime)...")
        ils_ad_run_res = run_ILS_adaptive_runtime_experiment(graph, DYNAMIC_TIME_LIMIT)
        for res_tuple in ils_ad_run_res:
             all_runtime_results_list.append({
                 "Experiment": "ILS_Adaptive_Runtime", "Run": i, "Mutation_Size": res_tuple[5], # final_mut_mean
                 "Stopping_Crit": "", "Fitness": res_tuple[2], "Comp_Time": res_tuple[3],
                 "Unchanged_Count": res_tuple[4], # stagnation count
                 "Solution": res_tuple[1]
             })


    # -----------------------------
    # Combine all results into a single CSV file.
    # -----------------------------
    print("\n--- Aggregating All Results ---")
    all_rows = [] # Start with an empty list

    # --- Append Parameter Experiment Results (Comparison a) ---
    print("  Adding results from Parameter Experiments...")
    # Add results from the pass-based experiments run earlier
    for run, sol, fitness, comp_time, _, _, _ in mls_results:
         all_rows.append({
             "Experiment": "MLS_PassBased", "Run": run, "Mutation_Size": "", "Stopping_Crit": "",
             "Fitness": fitness, "Comp_Time": comp_time, "Unchanged_Count": "", "Solution": sol
         })
    for run, sol, fitness, comp_time, unchanged_count, mutation_size, _ in ils_annealing_results:
         all_rows.append({
             "Experiment": "ILS_Annealing_PassBased", "Run": run, "Mutation_Size": mutation_size, "Stopping_Crit": "",
             "Fitness": fitness, "Comp_Time": comp_time, "Unchanged_Count": unchanged_count, "Solution": sol
         })
    for run, sol, fitness, comp_time, unchanged_count, mutation_size, _ in simple_ils_results:
         all_rows.append({
             "Experiment": "Simple_ILS_PassBased", "Run": run, "Mutation_Size": mutation_size, "Stopping_Crit": "",
             "Fitness": fitness, "Comp_Time": comp_time, "Unchanged_Count": unchanged_count, "Solution": sol
         })
    for run, sol, fitness, comp_time, _, _, stopping_crit in gls_results:
         all_rows.append({
             "Experiment": "GLS_PassBased", "Run": run, "Mutation_Size": "", "Stopping_Crit": stopping_crit,
             "Fitness": fitness, "Comp_Time": comp_time, "Unchanged_Count": "", "Solution": sol
         })
    for run, sol, fitness, comp_time, unchanged_count, final_mutation_size, _ in ils_adaptive_results:
         all_rows.append({
             "Experiment": "ILS_Adaptive_PassBased", "Run": run, "Mutation_Size": final_mutation_size, "Stopping_Crit": "",
             "Fitness": fitness, "Comp_Time": comp_time, "Unchanged_Count": unchanged_count, # Using Unchanged for stagnation
             "Solution": sol
         })

    # --- Append Runtime Experiment Results (Comparison b) ---
    print(f"  Adding results from {RUNTIME_RUNS} Runtime Experiment repetitions...")
    all_rows.extend(all_runtime_results_list)


    # --- Create and Save DataFrame ---
    df_experiments = pd.DataFrame(
        all_rows,
        columns=[
            "Experiment", "Run", "Mutation_Size", "Stopping_Crit",
            "Fitness", "Comp_Time", "Unchanged_Count", "Solution"
        ]
    )

    # Define the output file path relative to the script directory
    output_csv_path = os.path.join(script_dir, "experiment_results_combined.csv")

    df_experiments.to_csv(output_csv_path, index=False)
    print(f"\nAll combined experiment results saved in '{output_csv_path}'.")
    print("\n--- Experiment Script Finished ---")