import random
import math
import time
import os
import numpy as np
import pandas as pd
import warnings # Import warnings module

# --- Global Constants ---
NUM_VERTICES = 500
FM_PASSES_LIMIT = 10000 # Cumulative pass limit for comparison (a)
POPULATION_SIZE = 50
EXPERIMENT_RUNS = 10 # Runs for comparison (a)
RUNTIME_RUNS = 25   # Runs for comparison (b)

# --- Node Class ---
class Node:
    """ Stores info for a vertex during FM pass """
    __slots__ = ("gain", "neighbors", "prev", "next", "partition", "moved")

    def __init__(self, gain=0, neighbors=None, prev=-1, next=-1, partition=0):
        self.gain = gain
        self.neighbors = set(neighbors) if neighbors is not None else set()
        self.prev = prev
        self.next = next
        self.partition = partition
        self.moved = False

# --- Helper Functions ---

def calculate_gain(node_idx, solution, graph):
    """ Calculates the gain for moving node_idx to the opposite partition. """
    gain = 0
    n = len(solution)
    if not (0 <= node_idx < n and node_idx < len(graph)): return 0
    if not isinstance(solution, (list, np.ndarray)): return 0
    try: current_partition = solution[node_idx]
    except IndexError: return 0
    neighbors = graph[node_idx] if hasattr(graph[node_idx], '__iter__') else []
    for neighbor in neighbors:
        if not (0 <= neighbor < n): continue
        try:
            neighbor_partition = solution[neighbor]
            if neighbor_partition == current_partition: gain -= 1
            else: gain += 1
        except IndexError: continue
    return gain

def insert_node(nodes, gain_lists, max_gain, node_index, max_degree_adj):
    """ Inserts node_index into the correct gain bucket's linked list. """
    if not (0 <= node_index < len(nodes)): return
    node = nodes[node_index]
    gain = node.gain; part = node.partition
    if part not in (0, 1): return
    gain_index = gain + max_degree_adj
    if not (0 <= gain_index < len(gain_lists[part])): return

    node.next = gain_lists[part][gain_index]; node.prev = -1
    head_node_idx = gain_lists[part][gain_index]
    if head_node_idx != -1:
         if 0 <= head_node_idx < len(nodes): nodes[head_node_idx].prev = node_index
    gain_lists[part][gain_index] = node_index
    if -max_degree_adj <= gain <= max_degree_adj:
        if gain > max_gain[part]: max_gain[part] = gain

def remove_node(nodes, gain_lists, node_index, max_degree_adj):
    """ Removes node_index from its gain bucket's linked list. """
    if not (0 <= node_index < len(nodes)): return
    node = nodes[node_index]
    gain = node.gain; part = node.partition
    if part not in (0, 1): return
    gain_index = gain + max_degree_adj
    if not (0 <= gain_index < len(gain_lists[part])): return

    prev_node_idx = node.prev; next_node_idx = node.next
    if prev_node_idx != -1:
        if 0 <= prev_node_idx < len(nodes): nodes[prev_node_idx].next = next_node_idx
    else: gain_lists[part][gain_index] = next_node_idx
    if next_node_idx != -1:
        if 0 <= next_node_idx < len(nodes): nodes[next_node_idx].prev = prev_node_idx
    node.prev = -1; node.next = -1

def update_max_gain(gain_lists, max_gain, max_degree_adj):
    """ Finds the new highest non-empty gain bucket after potential changes. """
    min_gain = -max_degree_adj
    for part in range(2):
        current_max = max_gain[part]
        current_max = min(current_max, max_degree_adj)
        current_max = max(current_max, min_gain -1)
        found_new_max = False
        while current_max >= min_gain:
            gain_index = current_max + max_degree_adj
            if not (0 <= gain_index < len(gain_lists[part])): current_max = min_gain - 1; break
            if gain_lists[part][gain_index] != -1:
                max_gain[part] = current_max; found_new_max = True; break
            current_max -= 1
        if not found_new_max: max_gain[part] = min_gain - 1

def fitness_function(solution, graph):
    """ Calculates cut size. """
    n = len(solution)
    if n == 0: return 0
    total = 0
    try: sol_list = list(map(int, solution))
    except (TypeError, ValueError): return float('inf')
    if len(sol_list) != n: return float('inf')
    for i in range(n):
        if i < len(graph) and hasattr(graph[i], '__iter__'):
             part_i = sol_list[i]
             for nb in graph[i]:
                 if 0 <= nb < n:
                     if part_i != sol_list[nb]: total += 1
    return total // 2

def balance_child(child_list_input):
    """ Basic random balancing function. Ensures output is list of ints. """
    n = len(child_list_input)
    if n == 0: return []
    target_ones = n // 2
    child_list = []
    try: child_list = [int(bit) for bit in child_list_input]
    except (TypeError, ValueError): return list(child_list_input)
    if len(child_list) != n: return list(child_list_input)
    try: current_ones = child_list.count(1)
    except TypeError: return child_list
    if child_list.count(0) + current_ones != n: return child_list

    diff = current_ones - target_ones
    indices = list(range(n))
    if diff > 0:
        ones_indices = [i for i in indices if child_list[i] == 1]
        if len(ones_indices) >= diff:
             indices_to_flip = random.sample(ones_indices, diff)
             for i in indices_to_flip: child_list[i] = 0
        else: return child_list
    elif diff < 0:
        num_to_flip = -diff
        zeros_indices = [i for i in indices if child_list[i] == 0]
        if len(zeros_indices) >= num_to_flip:
             indices_to_flip = random.sample(zeros_indices, num_to_flip)
             for i in indices_to_flip: child_list[i] = 1
        else: return child_list
    return child_list

def read_graph_data(filename):
    """Reads the graph and returns adjacency lists (list of sets)."""
    graph = [set() for _ in range(NUM_VERTICES)]
    print(f"Reading graph data from: {filename}")
    try:
        with open(filename, "r") as f:
            ln = 0
            for line in f:
                ln += 1; line = line.strip()
                if not line: continue
                parts = line.split()
                if not parts: continue
                try:
                    vertex_index = int(parts[0]) - 1
                    if not (0 <= vertex_index < NUM_VERTICES): continue
                    num_neighbors_idx = -1
                    for idx, part in enumerate(parts):
                         if idx > 0 and ')' in parts[idx-1] and part.isdigit(): num_neighbors_idx = idx; break
                    if num_neighbors_idx == -1:
                        for idx, part in enumerate(parts):
                            if idx > 0 and part.isdigit(): num_neighbors_idx = idx; break
                    if num_neighbors_idx == -1 or num_neighbors_idx + 1 >= len(parts): continue
                    connected_vertices = [int(n) - 1 for n in parts[num_neighbors_idx + 1:]]
                    valid_neighbors = {nb for nb in connected_vertices if 0 <= nb < NUM_VERTICES}
                    graph[vertex_index].update(valid_neighbors)
                except (ValueError, IndexError) as e: print(f"Skipping line {ln}: {e}"); continue
        print("Ensuring graph symmetry...")
        for i in range(NUM_VERTICES):
             current_neighbors = list(graph[i])
             for neighbor in current_neighbors:
                  if 0 <= neighbor < NUM_VERTICES:
                      if i not in graph[neighbor]: graph[neighbor].add(i)
                  else: graph[i].discard(neighbor)
        print("Graph symmetry ensured.")
    except FileNotFoundError: print(f"Error: Graph file not found at {filename}"); raise
    return graph

def generate_random_solution():
    """Generates a balanced random solution (list of 0s and 1s)."""
    half = NUM_VERTICES // 2
    solution = [0] * half + [1] * (NUM_VERTICES - half)
    random.shuffle(solution)
    return solution

def mutate(solution, mutation_size):
    """Perturbs by swapping bits while preserving balance."""
    mutated = solution.copy()
    zeros = [i for i, bit in enumerate(mutated) if bit == 0]
    ones = [i for i, bit in enumerate(mutated) if bit == 1]
    num_mutations = min(mutation_size, len(zeros), len(ones))
    if num_mutations > 0:
         zeros_to_swap = random.sample(zeros, num_mutations)
         ones_to_swap = random.sample(ones, num_mutations)
         for i in range(num_mutations):
             mutated[zeros_to_swap[i]] = 1
             mutated[ones_to_swap[i]] = 0
    return mutated

def get_hamming_distance(parent1, parent2):
    """Compute the Hamming distance between two solutions."""
    dist = 0
    if len(parent1) != len(parent2): return len(parent1)
    for i in range(len(parent1)):
        if parent1[i] != parent2[i]: dist +=1
    return dist

def crossover(parent1, parent2):
    """ Performs uniform crossover respecting balance constraint. Returns list of ints. """
    n = len(parent1)
    child = [0] * n
    if len(parent1) != n or len(parent2) != n: return generate_random_solution()

    p1_eff = parent1
    hd = get_hamming_distance(parent1, parent2)
    if hd > n / 2: p1_eff = [1 - bit for bit in parent1]

    disagree_indices = []; ones_needed = n // 2; zeros_needed = n - ones_needed
    ones_count = 0; zeros_count = 0
    for i in range(n):
        if str(p1_eff[i]) == str(parent2[i]):
            child[i] = int(p1_eff[i])
            if child[i] == 1: ones_count += 1
            else: zeros_count += 1
        else: disagree_indices.append(i)

    ones_to_add = ones_needed - ones_count; zeros_to_add = zeros_needed - zeros_count
    if ones_to_add < 0 or zeros_to_add < 0 or (ones_to_add + zeros_to_add != len(disagree_indices)):
         return balance_child(child)
    else:
         random.shuffle(disagree_indices)
         for i in range(len(disagree_indices)):
             child[disagree_indices[i]] = 1 if i < ones_to_add else 0
    return child

# --- FM HEURISTIC ---
def fm_heuristic(initial_solution, graph, max_passes=10):
    """
    Fiduccia-Mattheyses heuristic based on Prac2.pdf description.
    Ensures returned solution is balanced by discarding passes if best state is imbalanced.
    Uses gain buckets for efficiency. Optimized balance check.

    Args:
        initial_solution (list): Initial partition (list of 0s and 1s).
        graph (list): Adjacency list/set representation.
        max_passes (int): Max outer passes allowed for *this specific call*.

    Returns:
        tuple: (optimized_balanced_solution, passes_performed_this_call)
    """
    n = len(initial_solution)
    if n == 0: return initial_solution.copy(), 0
    target_part_size = n // 2

    try: working_solution = [int(bit) for bit in initial_solution]
    except (ValueError, TypeError): return initial_solution.copy(), 0
    if working_solution.count(0) != target_part_size:
        working_solution = balance_child(working_solution)

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

    improved_in_cycle = True
    passes_done = 0
    while improved_in_cycle and passes_done < max_passes:
        improved_in_cycle = False
        passes_done += 1
        solution_at_pass_start = working_solution.copy()

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

        moves_sequence = []; cumulative_gains = [0.0]
        current_solution_in_pass = working_solution.copy()
        current_part0_count_in_pass = current_solution_in_pass.count(0)

        for k in range(n):
            best_node_to_move = -1; selected_gain = -float('inf')
            for part_from in range(2):
                current_max_g = max_gain[part_from]
                found_candidate_this_part = False
                while current_max_g >= min_gain:
                    gain_idx = current_max_g + max_degree_adj
                    if not (0 <= gain_idx < gain_list_size): break
                    node_idx_in_bucket = gain_lists[part_from][gain_idx]
                    while node_idx_in_bucket != -1:
                        if not nodes[node_idx_in_bucket].moved:
                            is_move_valid = False
                            if part_from == 0: # Moving from 0
                                if current_part0_count_in_pass > target_part_size - 1: is_move_valid = True
                            else: # Moving from 1
                                if (n - current_part0_count_in_pass) > target_part_size - 1: is_move_valid = True
                            if is_move_valid:
                                best_node_to_move = node_idx_in_bucket
                                selected_gain = nodes[best_node_to_move].gain
                                found_candidate_this_part = True; break
                        node_idx_in_bucket = nodes[node_idx_in_bucket].next
                    if found_candidate_this_part: break
                    current_max_g -= 1
                if best_node_to_move != -1: break
            if best_node_to_move == -1: break

            node_to_move_idx = best_node_to_move
            original_partition = nodes[node_to_move_idx].partition
            nodes[node_to_move_idx].moved = True
            remove_node(nodes, gain_lists, node_to_move_idx, max_degree_adj)
            current_solution_in_pass[node_to_move_idx] = 1 - original_partition
            nodes[node_to_move_idx].partition = current_solution_in_pass[node_to_move_idx]
            if original_partition == 0: current_part0_count_in_pass -= 1
            else: current_part0_count_in_pass += 1
            moves_sequence.append(node_to_move_idx)
            cumulative_gains.append(cumulative_gains[-1] + selected_gain)

            for neighbor_idx in nodes[node_to_move_idx].neighbors:
                if not nodes[neighbor_idx].moved:
                    gain_delta = 2 if current_solution_in_pass[neighbor_idx] == original_partition else -2
                    neighbor_gain_before = nodes[neighbor_idx].gain
                    neighbor_gain_idx_before = neighbor_gain_before + max_degree_adj
                    if 0 <= neighbor_gain_idx_before < gain_list_size:
                        remove_node(nodes, gain_lists, neighbor_idx, max_degree_adj)
                    nodes[neighbor_idx].gain += gain_delta
                    neighbor_gain_idx_after = nodes[neighbor_idx].gain + max_degree_adj
                    if 0 <= neighbor_gain_idx_after < gain_list_size:
                        insert_node(nodes, gain_lists, max_gain, neighbor_idx, max_degree_adj)
            update_max_gain(gain_lists, max_gain, max_degree_adj)

        if not moves_sequence: continue

        best_k = np.argmax(cumulative_gains); best_num_moves = best_k
        solution_after_rollback = solution_at_pass_start.copy()
        for i in range(best_num_moves):
            if i < len(moves_sequence):
                 node_idx = moves_sequence[i]
                 if 0 <= node_idx < n: solution_after_rollback[node_idx] = 1 - solution_after_rollback[node_idx]
                 else: solution_after_rollback = solution_at_pass_start.copy(); break
            else: solution_after_rollback = solution_at_pass_start.copy(); break

        final_solution_this_pass = solution_after_rollback
        if final_solution_this_pass.count(0) != target_part_size:
            working_solution = solution_at_pass_start.copy()
        else:
            working_solution = final_solution_this_pass.copy()
            current_cut_size = fitness_function(final_solution_this_pass, graph)
            if current_cut_size < best_overall_cut_size:
                 best_overall_cut_size = current_cut_size
                 best_solution_overall = final_solution_this_pass.copy()
                 improved_in_cycle = True

    if best_solution_overall.count(0) != target_part_size:
         print(f"CRITICAL ERROR: fm_heuristic final solution is not balanced!")

    return best_solution_overall, passes_done


# --- Revised Metaheuristics (Handling Pass Limits) ---

def MLS(graph, num_starts, max_total_passes=None, time_limit=None):
    """ Multi-start Local Search with cumulative pass limit and time limit. """
    best_solution = None; best_fit = float("inf")
    start_time = time.perf_counter(); total_passes_consumed = 0
    force_run_to_pass_limit = (max_total_passes is not None and time_limit is None)
    effective_num_starts = float('inf') if force_run_to_pass_limit else num_starts
    i = 0
    while i < effective_num_starts:
        current_time = time.perf_counter()
        if time_limit is not None and current_time - start_time >= time_limit: break
        if max_total_passes is not None and total_passes_consumed >= max_total_passes: break
        sol = generate_random_solution()
        remaining_passes = float('inf')
        if max_total_passes is not None:
            remaining_passes = max(1, max_total_passes - total_passes_consumed) # Min 1 pass
            if remaining_passes == 0: break # Should be caught by >= check
            remaining_passes = int(remaining_passes)
        try:
            local_opt, passes_this_call = fm_heuristic(sol, graph, max_passes=remaining_passes)
            total_passes_consumed += passes_this_call
            if local_opt is not None:
                if local_opt.count(0) != len(local_opt)//2: local_opt = balance_child(local_opt)
                fit = fitness_function(local_opt, graph)
                if fit < best_fit: best_fit = fit; best_solution = local_opt.copy()
        except Exception as e: print(f"Error in MLS fm_heuristic call: {e}");
        finally: i += 1
    if best_solution is None and i > 0: best_solution = generate_random_solution()
    return best_solution, total_passes_consumed

def ILS(graph, initial_solution, mutation_size, max_total_passes=None, time_limit=None):
    """ Simple Iterated Local Search with limits. """
    if initial_solution.count(0) != len(initial_solution) // 2:
        best_solution = balance_child(initial_solution.copy())
    else: best_solution = initial_solution.copy()
    best_fit = fitness_function(best_solution, graph)
    start_time = time.perf_counter(); total_passes_consumed = 0
    no_improvement = 0; max_no_improvement = 10; unchanged_count = 0; iteration = 0
    force_run_to_pass_limit = (max_total_passes is not None and time_limit is None)
    while True:
        iteration += 1; current_time = time.perf_counter()
        if time_limit is not None and current_time - start_time >= time_limit: break
        if max_total_passes is not None and total_passes_consumed >= max_total_passes: break
        if not force_run_to_pass_limit and no_improvement >= max_no_improvement: break
        mutated = mutate(best_solution, mutation_size)
        if mutated.count(0) != len(mutated) // 2: mutated = balance_child(mutated)
        remaining_passes = float('inf')
        if max_total_passes is not None:
            remaining_passes = max(1, max_total_passes - total_passes_consumed)
            if remaining_passes == 0: break
            remaining_passes = int(remaining_passes)
        try:
            local_opt, passes_this_call = fm_heuristic(mutated, graph, max_passes=remaining_passes)
            total_passes_consumed += passes_this_call
            if local_opt is not None:
                if local_opt.count(0) != len(local_opt)//2: local_opt = balance_child(local_opt)
                fit = fitness_function(local_opt, graph)
                if fit == best_fit: unchanged_count += 1
                if fit < best_fit: best_solution = local_opt.copy(); best_fit = fit; no_improvement = 0; unchanged_count = 0
                else: no_improvement += 1
            else: no_improvement += 1
        except Exception as e: print(f"Error in ILS fm_heuristic call: {e}"); no_improvement += 1; continue
    return best_solution, unchanged_count, total_passes_consumed

def ILS_annealing(graph, initial_solution, mutation_size, max_total_passes=None, time_limit=None):
    """ ILS with Annealing and limits. """
    if initial_solution.count(0) != len(initial_solution) // 2:
        current_solution = balance_child(initial_solution.copy())
    else: current_solution = initial_solution.copy()
    best_solution = current_solution.copy()
    current_fit = fitness_function(current_solution, graph); best_fit = current_fit
    start_time = time.perf_counter(); total_passes_consumed = 0
    no_improvement_on_best = 0; max_no_improvement = 20; unchanged_count = 0
    temperature = 10.0; cooling_rate = 0.98; min_temperature = 0.1; iteration = 0
    force_run_to_pass_limit = (max_total_passes is not None and time_limit is None)
    while True:
        iteration += 1; current_time = time.perf_counter()
        if time_limit is not None and current_time - start_time >= time_limit: break
        if max_total_passes is not None and total_passes_consumed >= max_total_passes: break
        if not force_run_to_pass_limit:
             if no_improvement_on_best >= max_no_improvement: break
             if temperature <= min_temperature: break
        mutated = mutate(current_solution, mutation_size)
        if mutated.count(0) != len(mutated) // 2: mutated = balance_child(mutated)
        remaining_passes = float('inf')
        if max_total_passes is not None:
            remaining_passes = max(1, max_total_passes - total_passes_consumed)
            if remaining_passes == 0: break
            remaining_passes = int(remaining_passes)
        try:
            local_opt, passes_this_call = fm_heuristic(mutated, graph, max_passes=remaining_passes)
            total_passes_consumed += passes_this_call
            if local_opt is not None:
                if local_opt.count(0) != len(local_opt)//2: local_opt = balance_child(local_opt)
                fit = fitness_function(local_opt, graph)
                if fit == current_fit: unchanged_count += 1
                delta_e = fit - current_fit
                if delta_e < 0 or (temperature > 0 and random.random() < math.exp(-delta_e / temperature)):
                     current_solution = local_opt.copy(); current_fit = fit
                     if fit < best_fit: best_solution = local_opt.copy(); best_fit = fit; no_improvement_on_best = 0
                     else: no_improvement_on_best += 1
                else: no_improvement_on_best += 1
            else: no_improvement_on_best += 1
        except Exception as e: print(f"Error in ILS_A fm_heuristic call: {e}"); no_improvement_on_best += 1; continue
        if not force_run_to_pass_limit: temperature *= cooling_rate
    return best_solution, unchanged_count, total_passes_consumed

def ILS_adaptive(graph, initial_solution, max_total_passes=None, time_limit=None):
    """ Adaptive ILS with limits and adjusted parameters. """
    initial_mutation_mean = 80; mutation_std_dev = 2.0; decay_factor = 0.99
    increase_factor = 1.2; stagnation_threshold = 50; max_iterations = 100000
    min_mutation_size = 1; max_mutation_size = NUM_VERTICES // 4
    stagnation_window=15; stagnation_tolerance=0.0001; restart_reset=True

    if initial_solution.count(0) != len(initial_solution) // 2:
        current_solution = balance_child(initial_solution.copy())
    else: current_solution = initial_solution.copy()
    best_solution = current_solution.copy()
    current_fit = fitness_function(current_solution, graph); best_fit = current_fit
    current_mutation_mean = float(initial_mutation_mean); current_mutation_std_dev = float(mutation_std_dev)
    mutation_history = []; stagnation_count = 0; iteration = 0; fitness_history = [current_fit]
    start_time = time.perf_counter(); total_passes_consumed = 0
    force_run_to_pass_limit = (max_total_passes is not None and time_limit is None)
    effective_max_iterations = float('inf') if force_run_to_pass_limit else max_iterations

    while iteration < effective_max_iterations:
        iteration += 1; current_time = time.perf_counter()
        if time_limit is not None and current_time - start_time >= time_limit: break
        if max_total_passes is not None and total_passes_consumed >= max_total_passes: break

        mutation_size = int(np.random.normal(current_mutation_mean, current_mutation_std_dev))
        mutation_size = max(min_mutation_size, min(mutation_size, max_mutation_size))
        mutated = mutate(current_solution, mutation_size)
        if mutated.count(0) != len(mutated) // 2: mutated = balance_child(mutated)

        remaining_passes = float('inf')
        if max_total_passes is not None:
            remaining_passes = max(1, max_total_passes - total_passes_consumed)
            if remaining_passes == 0: break
            remaining_passes = int(remaining_passes)

        try:
            local_opt, passes_this_call = fm_heuristic(mutated, graph, max_passes=remaining_passes)
            total_passes_consumed += passes_this_call
        except Exception as e: print(f"Error in ILS_Ad fm_heuristic call: {e}"); stagnation_count += 1; continue
        if local_opt is None: stagnation_count += 1; continue

        if local_opt.count(0) != len(local_opt)//2: local_opt = balance_child(local_opt)
        local_opt_fit = fitness_function(local_opt, graph)
        fitness_history.append(local_opt_fit)

        if local_opt_fit < current_fit:
            current_solution = local_opt.copy(); current_fit = local_opt_fit
            stagnation_count = 0
            current_mutation_mean = max(min_mutation_size, current_mutation_mean * decay_factor)
            current_mutation_std_dev = max(1.0, current_mutation_std_dev * decay_factor)
            if current_fit < best_fit: best_solution = current_solution.copy(); best_fit = current_fit
        else: stagnation_count += 1
        # Store history regardless of acceptance for analysis
        mutation_history.append((iteration, mutation_size, current_mutation_mean, current_mutation_std_dev, current_fit))

        if not force_run_to_pass_limit and stagnation_count >= stagnation_threshold:
             is_stuck = True
             if len(fitness_history) >= stagnation_window:
                  if np.mean(np.abs(np.diff(fitness_history[-stagnation_window:]))) >= stagnation_tolerance: is_stuck = False
             if is_stuck:
                  current_mutation_mean = min(max_mutation_size, current_mutation_mean * increase_factor)
                  current_mutation_std_dev = min(current_mutation_mean / 2, current_mutation_std_dev * increase_factor)
                  current_mutation_std_dev = max(1.0, current_mutation_std_dev)
                  stagnation_count = 0

    return (best_solution, best_fit, mutation_history, iteration,
            stagnation_count, int(current_mutation_mean), total_passes_consumed)

def GLS(graph, population_size, stopping_crit=None, max_total_passes=None, time_limit=None):
    """ Genetic Local Search with limits. """
    start_time = time.perf_counter(); total_passes_consumed = 0
    population = []; fitness_values = []
    force_run_to_pass_limit = (max_total_passes is not None and time_limit is None)

    # --- Population Initialization ---
    for i in range(population_size):
        current_time = time.perf_counter()
        if time_limit is not None and current_time - start_time >= time_limit: break
        if max_total_passes is not None and total_passes_consumed >= max_total_passes: break
        initial_sol = generate_random_solution()
        remaining_passes = float('inf')
        if max_total_passes is not None:
            remaining_passes = max(1, max_total_passes - total_passes_consumed)
            if remaining_passes == 0: break
            remaining_passes = int(remaining_passes)
        try:
            sol_opt, passes_this_call = fm_heuristic(initial_sol, graph, max_passes=remaining_passes)
            total_passes_consumed += passes_this_call
            if sol_opt is not None:
                if sol_opt.count(0) != len(sol_opt)//2: sol_opt = balance_child(sol_opt)
                population.append(sol_opt)
                fitness_values.append(fitness_function(sol_opt, graph))
        except Exception as e: print(f"  Error during GLS init fm_heuristic: {e}"); continue
    if not population: return None, total_passes_consumed

    best_fit_idx = np.argmin(fitness_values) if fitness_values else -1
    if best_fit_idx == -1 : return None, total_passes_consumed
    best_fit = fitness_values[best_fit_idx]; best_solution = population[best_fit_idx]
    generation_without_improvement = 0; gen = 0

    # --- Evolutionary Loop ---
    effective_max_generations = float('inf') if force_run_to_pass_limit else 100000

    while gen < effective_max_generations:
        gen += 1; current_time = time.perf_counter()
        if time_limit is not None and current_time - start_time >= time_limit: break
        if max_total_passes is not None and total_passes_consumed >= max_total_passes: break
        if not force_run_to_pass_limit and stopping_crit is not None and generation_without_improvement >= stopping_crit: break
        if len(population) < 2: break

        parent1_idx = random.randrange(len(population)); parent2_idx = random.randrange(len(population))
        parent1 = population[parent1_idx]; parent2 = population[parent2_idx]
        child = crossover(parent1, parent2)

        remaining_passes = float('inf')
        if max_total_passes is not None:
            remaining_passes = max(1, max_total_passes - total_passes_consumed)
            if remaining_passes == 0: break
            remaining_passes = int(remaining_passes)
        try:
            child_opt, passes_this_call = fm_heuristic(child, graph, max_passes=remaining_passes)
            total_passes_consumed += passes_this_call
        except Exception as e: print(f"  Error during GLS evolve fm_heuristic: {e}"); continue
        if child_opt is None: continue

        if child_opt.count(0) != len(child_opt)//2: child_opt = balance_child(child_opt)
        fit_child = fitness_function(child_opt, graph)
        worst_fit_idx = np.argmax(fitness_values)
        worst_fit = fitness_values[worst_fit_idx]
        improvement_found_this_gen = False
        if fit_child <= worst_fit:
            population[worst_fit_idx] = child_opt.copy()
            fitness_values[worst_fit_idx] = fit_child
            if fit_child < best_fit:
                best_fit = fit_child; best_solution = child_opt.copy()
                improvement_found_this_gen = True
        if improvement_found_this_gen: generation_without_improvement = 0
        else: generation_without_improvement += 1

    return best_solution, total_passes_consumed


# --- Revised Experiment Runners ---

def run_MLS_experiment(graph):
    """Runs MLS 10 times, limited by FM_PASSES_LIMIT cumulative passes."""
    print("--- Running MLS Pass-Limited Experiment ---")
    num_starts = 10000 # High num_starts to ensure pass limit is the primary stop condition
    results = []
    for i in range(EXPERIMENT_RUNS):
        print(f"  MLS Pass-Limited Run {i+1}/{EXPERIMENT_RUNS}")
        start_time = time.perf_counter()
        best_sol, passes_consumed = MLS(graph, num_starts, max_total_passes=FM_PASSES_LIMIT, time_limit=None)
        end_time = time.perf_counter()
        comp_time = end_time - start_time
        fit = fitness_function(best_sol, graph) if best_sol else float('inf')
        solution_str = "".join(map(str, best_sol)) if best_sol else ""
        results.append((i, solution_str, fit, comp_time, passes_consumed, "", "")) # Pass index 4
        print(f"    Run {i+1}: Fitness={fit}, Passes={passes_consumed}, Time={comp_time:.4f}s")
    return results

def run_simple_ILS_experiment(graph):
    """Runs Simple ILS 10 times per mutation size, limited by FM_PASSES_LIMIT cumulative passes."""
    print("--- Running Simple ILS Pass-Limited Experiment ---")
    mutation_sizes = [10, 25, 40, 50, 60, 75, 100] # Refined range
    results = []
    for mutation_size in mutation_sizes:
        print(f"  Simple ILS Testing Mutation Size: {mutation_size}")
        for i in range(EXPERIMENT_RUNS):
            initial_sol = generate_random_solution()
            start_time = time.perf_counter()
            best_sol, unchanged_count, passes_consumed = ILS(
                graph, initial_sol, mutation_size, max_total_passes=FM_PASSES_LIMIT, time_limit=None
            )
            end_time = time.perf_counter()
            comp_time = end_time - start_time
            fit = fitness_function(best_sol, graph) if best_sol else float('inf')
            solution_str = "".join(map(str, best_sol)) if best_sol else ""
            unchanged_count = unchanged_count if best_sol else -1 # Indicate error if no solution
            # Results tuple: (run_idx, solution, fitness, time, passes, mut_size, unchanged)
            results.append((i, solution_str, fit, comp_time, passes_consumed, mutation_size, unchanged_count))
            print(f"      Run {i+1}: Fitness={fit}, Passes={passes_consumed}, Unchanged={unchanged_count}, Time={comp_time:.4f}s")
    return results

def run_GLS_experiment(graph):
    """Runs GLS 10 times for stopping_crit=1, limited by FM_PASSES_LIMIT cumulative passes."""
    print("--- Running GLS Pass-Limited Experiment ---")
    stopping_crit = 1 # Use only stopping_crit=1 as requested
    results = []
    print(f"  GLS Testing Stopping Criterion: {stopping_crit}")
    for i in range(EXPERIMENT_RUNS):
         start_time = time.perf_counter()
         best_sol, passes_consumed = GLS(
             graph, POPULATION_SIZE, stopping_crit, max_total_passes=FM_PASSES_LIMIT, time_limit=None
         )
         end_time = time.perf_counter()
         comp_time = end_time - start_time
         fit = fitness_function(best_sol, graph) if best_sol else float('inf')
         solution_str = "".join(map(str, best_sol)) if best_sol else ""
         # Results tuple: (run_idx, solution, fitness, time, passes, "", stop_crit)
         results.append((i, solution_str, fit, comp_time, passes_consumed, "", stopping_crit))
         print(f"      Run {i+1}: Fitness={fit}, Passes={passes_consumed}, Time={comp_time:.4f}s")
    return results

def run_ILS_adaptive_experiment(graph):
    """Runs Adaptive ILS 10 times, limited by FM_PASSES_LIMIT cumulative passes."""
    print("--- Running ILS Adaptive Pass-Limited Experiment ---")
    results = []
    for i in range(EXPERIMENT_RUNS):
        initial_sol = generate_random_solution()
        start_time = time.perf_counter()
        res_tuple = ILS_adaptive(
             graph, initial_sol, max_total_passes=FM_PASSES_LIMIT, time_limit=None
        )
        end_time = time.perf_counter()
        comp_time = end_time - start_time
        best_sol, best_fit_val, _, _, final_stag_count, final_mut_mean, passes_consumed = res_tuple
        if best_sol is None: fit = float('inf'); solution_str = ""; final_stag_count = -1; final_mut_mean = -1
        else: fit = best_fit_val; solution_str = "".join(map(str, best_sol))
        # Results tuple: (run_idx, solution, fitness, time, passes, mut_mean, stagnation)
        results.append((i, solution_str, fit, comp_time, passes_consumed, final_mut_mean, final_stag_count))
        print(f"    Run {i+1}: Fitness={fit}, Passes={passes_consumed}, Stagnation={final_stag_count}, MutMean={final_mut_mean}, Time={comp_time:.4f}s")
    return results

def run_ILS_annealing_experiment(graph):
    """Runs ILS Annealing 10 times per mutation size, limited by FM_PASSES_LIMIT cumulative passes."""
    print("--- Running ILS Annealing Pass-Limited Experiment ---")
    mutation_sizes = [10, 25, 40, 50, 60, 75, 100] # Refined range
    results = []
    for mutation_size in mutation_sizes:
        print(f"  ILS Annealing Testing Mutation Size: {mutation_size}")
        for i in range(EXPERIMENT_RUNS):
             initial_sol = generate_random_solution()
             start_time = time.perf_counter()
             best_sol, unchanged_count, passes_consumed = ILS_annealing(
                 graph, initial_sol, mutation_size, max_total_passes=FM_PASSES_LIMIT, time_limit=None
             )
             end_time = time.perf_counter()
             comp_time = end_time - start_time
             if best_sol is None: fit = float('inf'); solution_str = ""; unchanged_count = -1
             else: fit = fitness_function(best_sol, graph); solution_str = "".join(map(str, best_sol))
             # Results tuple: (run_idx, solution, fitness, time, passes, mut_size, unchanged)
             results.append((i, solution_str, fit, comp_time, passes_consumed, mutation_size, unchanged_count))
             print(f"      Run {i+1}: Fitness={fit}, Passes={passes_consumed}, Unchanged={unchanged_count}, Time={comp_time:.4f}s")
    return results


# --- Runtime Experiment Runners ---

def run_MLS_runtime_experiment(graph, time_limit):
    """Runs MLS with a time limit."""
    best_sol, passes_consumed = MLS(graph, num_starts=1000000, time_limit=time_limit, max_total_passes=None)
    if best_sol is None: fit = float('inf'); solution_str = ""
    else: fit = fitness_function(best_sol, graph); solution_str = "".join(map(str, best_sol))
    # Return tuple matching pass-based structure where possible
    return [(0, solution_str, fit, time_limit, passes_consumed, "", "")]

def run_ILS_runtime_experiment(graph, time_limit, mutation_size):
    """Runs Simple ILS with a time limit."""
    initial_sol = generate_random_solution()
    best_sol, unchanged_count, passes_consumed = ILS(graph, initial_sol, mutation_size, time_limit=time_limit, max_total_passes=None)
    if best_sol is None: fit = float('inf'); solution_str = ""; unchanged_count = -1
    else: fit = fitness_function(best_sol, graph); solution_str = "".join(map(str, best_sol))
    return [(0, solution_str, fit, time_limit, passes_consumed, mutation_size, unchanged_count)]

def run_GLS_runtime_experiment(graph, time_limit, stopping_crit):
    """Runs GLS with a time limit."""
    best_sol, passes_consumed = GLS(graph, POPULATION_SIZE, stopping_crit, time_limit=time_limit, max_total_passes=None)
    if best_sol is None: fit = float('inf'); solution_str = ""
    else: fit = fitness_function(best_sol, graph); solution_str = "".join(map(str, best_sol))
    return [(0, solution_str, fit, time_limit, passes_consumed, "", stopping_crit)]

def run_ILS_adaptive_runtime_experiment(graph, time_limit):
    """Runs Adaptive ILS with a time limit."""
    initial_sol = generate_random_solution()
    res_tuple = ILS_adaptive(graph, initial_sol, time_limit=time_limit, max_total_passes=None)
    best_sol, best_fit_val, _, _, final_stag_count, final_mut_mean, passes_consumed = res_tuple
    if best_sol is None: fit = float('inf'); solution_str = ""; final_stag_count = -1; final_mut_mean = -1
    else: fit = best_fit_val; solution_str = "".join(map(str, best_sol))
    return [(0, solution_str, fit, time_limit, passes_consumed, final_mut_mean, final_stag_count)]

def run_ILS_annealing_runtime_experiment(graph, time_limit, mutation_size):
    """Runs ILS Annealing with a time limit."""
    initial_sol = generate_random_solution()
    best_sol, unchanged_count, passes_consumed = ILS_annealing(graph, initial_sol, mutation_size, time_limit=time_limit, max_total_passes=None)
    if best_sol is None: fit = float('inf'); solution_str = ""; unchanged_count = -1
    else: fit = fitness_function(best_sol, graph); solution_str = "".join(map(str, best_sol))
    return [(0, solution_str, fit, time_limit, passes_consumed, mutation_size, unchanged_count)]


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Setup ---
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not script_dir: script_dir = os.getcwd()
        graph_file_path = os.path.join(script_dir, "Graph500.txt")
        if not os.path.exists(graph_file_path): raise FileNotFoundError(f"Graph file not found: {graph_file_path}")
        graph = read_graph_data(graph_file_path)
        print("Graph data loaded successfully.")
    except Exception as e: print(f"Error during setup: {e}"); exit()

    # --- Determine Time Limit ---
    # --- !!! IMPORTANT: Uncomment and run this block ONCE to find your DYNAMIC_TIME_LIMIT !!! ---
    # print("\n--- Measuring Time for 10000 FM Passes ---")
    # NUM_TIMING_RUNS = 10; PASSES_TO_TIME = 10000; fm_times = []
    # print(f"(Running MLS {NUM_TIMING_RUNS} times with cumulative pass limit of {PASSES_TO_TIME})")
    # for i in range(NUM_TIMING_RUNS):
    #      print(f"  Timing Run {i + 1}/{NUM_TIMING_RUNS}...")
    #      initial_sol = generate_random_solution()
    #      start_time = time.perf_counter()
    #      try:
    #           _, passes_executed = MLS(graph, num_starts=10000, max_total_passes=PASSES_TO_TIME)
    #           end_time = time.perf_counter()
    #           if passes_executed >= PASSES_TO_TIME * 0.95 : # Allow slight undershoot
    #               duration = end_time - start_time
    #               fm_times.append(duration)
    #               print(f"    Time for {passes_executed} passes: {duration:.6f} seconds")
    #           else: print(f"    Warning: MLS run finished too early ({passes_executed} passes).")
    #      except Exception as e: print(f"    Error during MLS timing run: {e}")
    # if fm_times:
    #      median_fm_time = np.median(fm_times)
    #      print(f"\n  Median Time for ~{PASSES_TO_TIME} passes: {median_fm_time:.6f} seconds")
    #      print(f"  >>> Set DYNAMIC_TIME_LIMIT = {median_fm_time:.6f} below <<<")
    # else: print("\n  No valid FM timing runs completed.")
    # print("--- End of FM Timing Measurement ---")
    # exit()
    # --- !!! End of Temporary Timing Block !!! ---

    # --- HARDCODE DYNAMIC TIME LIMIT HERE ---
    DYNAMIC_TIME_LIMIT = 60 # 
    print(f"Using Dynamic Time Limit: {DYNAMIC_TIME_LIMIT:.6f} seconds for runtime experiments.")
    # -----------------------------------------

    all_results_list = [] # Stores dicts

    # -----------------------------
    # Pass-Limited Experiments (Comparison a)
    # -----------------------------
    print(f"\n--- Running Pass-Limited Experiments ({EXPERIMENT_RUNS} runs each, Limit: {FM_PASSES_LIMIT} passes) ---")
    mls_results_pass = run_MLS_experiment(graph)
    for r in mls_results_pass: all_results_list.append({"Experiment": "MLS_PassBased", "Run": r[0], "Fitness": r[2], "Comp_Time": r[3], "Actual_Passes": r[4], "Solution": r[1]})
    simple_ils_results_pass = run_simple_ILS_experiment(graph)
    for r in simple_ils_results_pass: all_results_list.append({"Experiment": "Simple_ILS_PassBased", "Run": r[0], "Mutation_Size": r[5], "Fitness": r[2], "Comp_Time": r[3], "Actual_Passes": r[4], "Unchanged_Count": r[6], "Solution": r[1]})
    gls_results_pass = run_GLS_experiment(graph)
    for r in gls_results_pass: all_results_list.append({"Experiment": "GLS_PassBased", "Run": r[0], "Stopping_Crit": r[6], "Fitness": r[2], "Comp_Time": r[3], "Actual_Passes": r[4], "Solution": r[1]})
    ils_adaptive_results_pass = run_ILS_adaptive_experiment(graph)
    for r in ils_adaptive_results_pass: all_results_list.append({"Experiment": "ILS_Adaptive_PassBased", "Run": r[0], "Mutation_Size": r[5], "Fitness": r[2], "Comp_Time": r[3], "Actual_Passes": r[4], "Unchanged_Count": r[6], "Solution": r[1]})
    ils_annealing_results_pass = run_ILS_annealing_experiment(graph)
    for r in ils_annealing_results_pass: all_results_list.append({"Experiment": "ILS_Annealing_PassBased", "Run": r[0], "Mutation_Size": r[5], "Fitness": r[2], "Comp_Time": r[3], "Actual_Passes": r[4], "Unchanged_Count": r[6], "Solution": r[1]})

    # --- Determine best parameters for Runtime tests ---
    print("\n--- Determining Best Parameters for Runtime Tests ---")
    best_ils_mutation_size = 50; best_ils_annealing_mutation_size = 50; best_gls_stopping_crit = 1 # Default GLS to 1 now
    try:
        df_simple_ils = pd.DataFrame([r for r in all_results_list if r['Experiment'] == 'Simple_ILS_PassBased'])
        if not df_simple_ils.empty:
             df_simple_ils['Mutation_Size'] = pd.to_numeric(df_simple_ils['Mutation_Size']); df_simple_ils['Fitness'] = pd.to_numeric(df_simple_ils['Fitness'])
             median_fits = df_simple_ils.groupby('Mutation_Size')['Fitness'].median()
             if not median_fits.empty: best_ils_mutation_size = int(median_fits.idxmin())
    except Exception as e: print(f"Could not determine best Simple ILS param: {e}")
    try:
        df_ils_anneal = pd.DataFrame([r for r in all_results_list if r['Experiment'] == 'ILS_Annealing_PassBased'])
        if not df_ils_anneal.empty:
             df_ils_anneal['Mutation_Size'] = pd.to_numeric(df_ils_anneal['Mutation_Size']); df_ils_anneal['Fitness'] = pd.to_numeric(df_ils_anneal['Fitness'])
             median_fits = df_ils_anneal.groupby('Mutation_Size')['Fitness'].median()
             if not median_fits.empty: best_ils_annealing_mutation_size = int(median_fits.idxmin())
    except Exception as e: print(f"Could not determine best ILS Annealing param: {e}")

    best_gls_stopping_crit = 1
    print(f"Using parameters for Runtime: ILS Mut={best_ils_mutation_size}, ILS_A Mut={best_ils_annealing_mutation_size}, GLS Stop={best_gls_stopping_crit}")

    # -----------------------------
    # Runtime experiments (Comparison b - Fixed Time Limit)
    # -----------------------------
    print(f"\n--- Running Runtime Experiments ({RUNTIME_RUNS} runs each, Limit: {DYNAMIC_TIME_LIMIT:.6f}s) ---")
    for i in range(RUNTIME_RUNS):
        print(f"\nRuntime Repetition {i + 1}/{RUNTIME_RUNS}")
        mls_run_res = run_MLS_runtime_experiment(graph, DYNAMIC_TIME_LIMIT)
        ils_a_run_res = run_ILS_annealing_runtime_experiment(graph, DYNAMIC_TIME_LIMIT, best_ils_annealing_mutation_size)
        ils_s_run_res = run_ILS_runtime_experiment(graph, DYNAMIC_TIME_LIMIT, best_ils_mutation_size)
        gls_run_res = run_GLS_runtime_experiment(graph, DYNAMIC_TIME_LIMIT, best_gls_stopping_crit)
        ils_ad_run_res = run_ILS_adaptive_runtime_experiment(graph, DYNAMIC_TIME_LIMIT)
        # Append results
        all_results_list.append({"Experiment": "MLS_Runtime", "Run": i, "Fitness": mls_run_res[0][2], "Comp_Time": mls_run_res[0][3], "Actual_Passes": mls_run_res[0][4], "Solution": mls_run_res[0][1]})
        all_results_list.append({"Experiment": "ILS_Annealing_Runtime", "Run": i, "Mutation_Size": ils_a_run_res[0][5], "Fitness": ils_a_run_res[0][2], "Comp_Time": ils_a_run_res[0][3], "Actual_Passes": ils_a_run_res[0][4], "Unchanged_Count": ils_a_run_res[0][6], "Solution": ils_a_run_res[0][1]})
        all_results_list.append({"Experiment": "ILS_Simple_Runtime", "Run": i, "Mutation_Size": ils_s_run_res[0][5], "Fitness": ils_s_run_res[0][2], "Comp_Time": ils_s_run_res[0][3], "Actual_Passes": ils_s_run_res[0][4], "Unchanged_Count": ils_s_run_res[0][6], "Solution": ils_s_run_res[0][1]})
        all_results_list.append({"Experiment": "GLS_Runtime", "Run": i, "Stopping_Crit": gls_run_res[0][6], "Fitness": gls_run_res[0][2], "Comp_Time": gls_run_res[0][3], "Actual_Passes": gls_run_res[0][4], "Solution": gls_run_res[0][1]})
        all_results_list.append({"Experiment": "ILS_Adaptive_Runtime", "Run": i, "Mutation_Size": ils_ad_run_res[0][5], "Fitness": ils_ad_run_res[0][2], "Comp_Time": ils_ad_run_res[0][3], "Actual_Passes": ils_ad_run_res[0][4], "Unchanged_Count": ils_ad_run_res[0][6], "Solution": ils_ad_run_res[0][1]})

    # -----------------------------
    # Final DataFrame Creation and Saving
    # -----------------------------
    print("\n--- Aggregating All Results ---")
    df_experiments = pd.DataFrame(all_results_list)
    final_columns = ["Experiment", "Run", "Mutation_Size", "Stopping_Crit", "Fitness", "Comp_Time", "Actual_Passes", "Unchanged_Count", "Solution"]
    df_experiments = df_experiments.reindex(columns=final_columns)
    df_experiments.fillna({"Mutation_Size": "", "Stopping_Crit": "", "Unchanged_Count": "", "Actual_Passes": -1 }, inplace=True)

    output_csv_path = os.path.join(script_dir, "experiment_results_combined.csv")
    try:
        df_experiments.to_csv(output_csv_path, index=False, float_format='%.6f')
        print(f"\nAll combined experiment results saved in '{output_csv_path}'.")
    except Exception as e: print(f"\nError saving results to CSV: {e}")
    print("\n--- Experiment Script Finished ---")

