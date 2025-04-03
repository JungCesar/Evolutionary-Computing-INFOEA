import random
import math
import time
import os
import numpy as np
import pandas as pd
import warnings  # Import warnings module
import itertools  # For grid search
import json  # For saving best params

# --- Global Constants ---
NUM_VERTICES = 500
FM_PASSES_LIMIT = 10000  # Cumulative pass limit for comparison (a)
POPULATION_SIZE = 50
EXPERIMENT_RUNS = 10  # Runs for comparison (a)
RUNTIME_RUNS = 25  # Runs for comparison (b)
GRID_SEARCH_RUNS = 5  # Runs per parameter combination in grid search
GRID_SEARCH_PASS_LIMIT = 1000


class Node:
    """Stores info for a vertex during FM pass"""

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
    """Calculates the gain for moving node_idx to the opposite partition."""
    gain = 0
    n = len(solution)
    if not (0 <= node_idx < n and node_idx < len(graph)):
        return 0
    if not isinstance(solution, (list, np.ndarray)):
        return 0
    try:
        current_partition = solution[node_idx]
    except IndexError:
        return 0
    neighbors = graph[node_idx] if hasattr(graph[node_idx], "__iter__") else []
    for neighbor in neighbors:
        if not (0 <= neighbor < n):
            continue
        try:
            neighbor_partition = solution[neighbor]
            if neighbor_partition == current_partition:
                gain -= 1
            else:
                gain += 1
        except IndexError:
            continue
    return gain


def insert_node(nodes, gain_lists, max_gain, node_index, max_degree_adj):
    """Inserts node_index into the correct gain bucket's linked list."""
    if not (0 <= node_index < len(nodes)):
        return
    node = nodes[node_index]
    gain = node.gain
    part = node.partition
    if part not in (0, 1):
        return
    gain_index = gain + max_degree_adj
    if not (0 <= gain_index < len(gain_lists[part])):
        return

    node.next = gain_lists[part][gain_index]
    node.prev = -1
    head_node_idx = gain_lists[part][gain_index]
    if head_node_idx != -1:
        if 0 <= head_node_idx < len(nodes):
            nodes[head_node_idx].prev = node_index
    gain_lists[part][gain_index] = node_index
    if -max_degree_adj <= gain <= max_degree_adj:
        if gain > max_gain[part]:
            max_gain[part] = gain


def remove_node(nodes, gain_lists, node_index, max_degree_adj):
    """Removes node_index from its gain bucket's linked list."""
    if not (0 <= node_index < len(nodes)):
        return
    node = nodes[node_index]
    gain = node.gain
    part = node.partition
    if part not in (0, 1):
        return
    gain_index = gain + max_degree_adj
    if not (0 <= gain_index < len(gain_lists[part])):
        return

    prev_node_idx = node.prev
    next_node_idx = node.next
    if prev_node_idx != -1:
        if 0 <= prev_node_idx < len(nodes):
            nodes[prev_node_idx].next = next_node_idx
    else:
        gain_lists[part][gain_index] = next_node_idx
    if next_node_idx != -1:
        if 0 <= next_node_idx < len(nodes):
            nodes[next_node_idx].prev = prev_node_idx
    node.prev = -1
    node.next = -1


def update_max_gain(gain_lists, max_gain, max_degree_adj):
    """Finds the new highest non-empty gain bucket after potential changes."""
    min_gain = -max_degree_adj
    for part in range(2):
        current_max = max_gain[part]
        current_max = min(current_max, max_degree_adj)
        current_max = max(current_max, min_gain - 1)
        found_new_max = False
        while current_max >= min_gain:
            gain_index = current_max + max_degree_adj
            if not (0 <= gain_index < len(gain_lists[part])):
                current_max = min_gain - 1
                break
            if gain_lists[part][gain_index] != -1:
                max_gain[part] = current_max
                found_new_max = True
                break
            current_max -= 1
        if not found_new_max:
            max_gain[part] = min_gain - 1


def fitness_function(solution, graph):
    """Calculates cut size."""
    n = len(solution)
    if n == 0:
        return 0
    total = 0
    try:
        sol_list = list(map(int, solution))
    except (TypeError, ValueError):
        return float("inf")
    if len(sol_list) != n:
        return float("inf")
    for i in range(n):
        if i < len(graph) and hasattr(graph[i], "__iter__"):
            part_i = sol_list[i]
            for nb in graph[i]:
                if 0 <= nb < n:
                    if part_i != sol_list[nb]:
                        total += 1
    return total // 2


def balance_child(child_list_input):
    """Basic random balancing function. Ensures output is list of ints."""
    n = len(child_list_input)
    if n == 0:
        return []
    target_ones = n // 2
    child_list = []
    try:
        child_list = [int(bit) for bit in child_list_input]
    except (TypeError, ValueError):
        return list(child_list_input)
    if len(child_list) != n:
        return list(child_list_input)
    try:
        current_ones = child_list.count(1)
    except TypeError:
        return child_list
    if child_list.count(0) + current_ones != n:
        return child_list

    diff = current_ones - target_ones
    indices = list(range(n))
    if diff > 0:
        ones_indices = [i for i in indices if child_list[i] == 1]
        if len(ones_indices) >= diff:
            indices_to_flip = random.sample(ones_indices, diff)
            for i in indices_to_flip:
                child_list[i] = 0
        else:
            return child_list
    elif diff < 0:
        num_to_flip = -diff
        zeros_indices = [i for i in indices if child_list[i] == 0]
        if len(zeros_indices) >= num_to_flip:
            indices_to_flip = random.sample(zeros_indices, num_to_flip)
            for i in indices_to_flip:
                child_list[i] = 1
        else:
            return child_list
    return child_list


def read_graph_data(filename):
    """Reads the graph and returns adjacency lists (list of sets)."""
    graph = [set() for _ in range(NUM_VERTICES)]
    print(f"Reading graph data from: {filename}")
    try:
        with open(filename, "r") as f:
            ln = 0
            for line in f:
                ln += 1
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if not parts:
                    continue
                try:
                    vertex_index = int(parts[0]) - 1
                    if not (0 <= vertex_index < NUM_VERTICES):
                        continue
                    num_neighbors_idx = -1
                    for idx, part in enumerate(parts):
                        if idx > 0 and ")" in parts[idx - 1] and part.isdigit():
                            num_neighbors_idx = idx
                            break
                    if num_neighbors_idx == -1:
                        for idx, part in enumerate(parts):
                            if idx > 0 and part.isdigit():
                                num_neighbors_idx = idx
                                break
                    if num_neighbors_idx == -1 or num_neighbors_idx + 1 >= len(parts):
                        continue
                    connected_vertices = [
                        int(n) - 1 for n in parts[num_neighbors_idx + 1 :]
                    ]
                    valid_neighbors = {
                        nb for nb in connected_vertices if 0 <= nb < NUM_VERTICES
                    }
                    graph[vertex_index].update(valid_neighbors)
                except (ValueError, IndexError) as e:
                    print(f"Skipping line {ln}: {e}")
                    continue
        print("Ensuring graph symmetry...")
        for i in range(NUM_VERTICES):
            current_neighbors = list(graph[i])
            for neighbor in current_neighbors:
                if 0 <= neighbor < NUM_VERTICES:
                    if i not in graph[neighbor]:
                        graph[neighbor].add(i)
                else:
                    graph[i].discard(neighbor)
        print("Graph symmetry ensured.")
    except FileNotFoundError:
        print(f"Error: Graph file not found at {filename}")
        raise
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
    if len(parent1) != len(parent2):
        return len(parent1)
    for i in range(len(parent1)):
        if parent1[i] != parent2[i]:
            dist += 1
    return dist


def crossover(parent1, parent2):
    """Performs uniform crossover respecting balance constraint. Returns list of ints."""
    n = len(parent1)
    child = [0] * n
    if len(parent1) != n or len(parent2) != n:
        return generate_random_solution()

    p1_eff = parent1
    hd = get_hamming_distance(parent1, parent2)
    if hd > n / 2:
        p1_eff = [1 - bit for bit in parent1]

    disagree_indices = []
    ones_needed = n // 2
    zeros_needed = n - ones_needed
    ones_count = 0
    zeros_count = 0
    for i in range(n):
        if str(p1_eff[i]) == str(parent2[i]):
            child[i] = int(p1_eff[i])
            if child[i] == 1:
                ones_count += 1
            else:
                zeros_count += 1
        else:
            disagree_indices.append(i)

    ones_to_add = ones_needed - ones_count
    zeros_to_add = zeros_needed - zeros_count
    if (
        ones_to_add < 0
        or zeros_to_add < 0
        or (ones_to_add + zeros_to_add != len(disagree_indices))
    ):
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
    if n == 0:
        return initial_solution.copy(), 0
    target_part_size = n // 2

    try:
        working_solution = [int(bit) for bit in initial_solution]
    except (ValueError, TypeError):
        return initial_solution.copy(), 0
    if working_solution.count(0) != target_part_size:
        working_solution = balance_child(working_solution)

    max_degree = 0
    for i in range(n):
        if i < len(graph) and hasattr(graph[i], "__iter__"):
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

        moves_sequence = []
        cumulative_gains = [0.0]
        current_solution_in_pass = working_solution.copy()
        current_part0_count_in_pass = current_solution_in_pass.count(0)

        for k in range(n):
            best_node_to_move = -1
            selected_gain = -float("inf")
            for part_from in range(2):
                current_max_g = max_gain[part_from]
                found_candidate_this_part = False
                while current_max_g >= min_gain:
                    gain_idx = current_max_g + max_degree_adj
                    if not (0 <= gain_idx < gain_list_size):
                        break
                    node_idx_in_bucket = gain_lists[part_from][gain_idx]
                    while node_idx_in_bucket != -1:
                        if not nodes[node_idx_in_bucket].moved:
                            is_move_valid = False
                            if part_from == 0:  # Moving from 0
                                if current_part0_count_in_pass > target_part_size - 1:
                                    is_move_valid = True
                            else:  # Moving from 1
                                if (
                                    n - current_part0_count_in_pass
                                ) > target_part_size - 1:
                                    is_move_valid = True
                            if is_move_valid:
                                best_node_to_move = node_idx_in_bucket
                                selected_gain = nodes[best_node_to_move].gain
                                found_candidate_this_part = True
                                break
                        node_idx_in_bucket = nodes[node_idx_in_bucket].next
                    if found_candidate_this_part:
                        break
                    current_max_g -= 1
                if best_node_to_move != -1:
                    break
            if best_node_to_move == -1:
                break

            node_to_move_idx = best_node_to_move
            original_partition = nodes[node_to_move_idx].partition
            nodes[node_to_move_idx].moved = True
            remove_node(nodes, gain_lists, node_to_move_idx, max_degree_adj)
            current_solution_in_pass[node_to_move_idx] = 1 - original_partition
            nodes[node_to_move_idx].partition = current_solution_in_pass[
                node_to_move_idx
            ]
            if original_partition == 0:
                current_part0_count_in_pass -= 1
            else:
                current_part0_count_in_pass += 1
            moves_sequence.append(node_to_move_idx)
            cumulative_gains.append(cumulative_gains[-1] + selected_gain)

            for neighbor_idx in nodes[node_to_move_idx].neighbors:
                if not nodes[neighbor_idx].moved:
                    gain_delta = (
                        2
                        if current_solution_in_pass[neighbor_idx] == original_partition
                        else -2
                    )
                    neighbor_gain_before = nodes[neighbor_idx].gain
                    neighbor_gain_idx_before = neighbor_gain_before + max_degree_adj
                    if 0 <= neighbor_gain_idx_before < gain_list_size:
                        remove_node(nodes, gain_lists, neighbor_idx, max_degree_adj)
                    nodes[neighbor_idx].gain += gain_delta
                    neighbor_gain_idx_after = nodes[neighbor_idx].gain + max_degree_adj
                    if 0 <= neighbor_gain_idx_after < gain_list_size:
                        insert_node(
                            nodes, gain_lists, max_gain, neighbor_idx, max_degree_adj
                        )
            update_max_gain(gain_lists, max_gain, max_degree_adj)

        if not moves_sequence:
            continue

        best_k = np.argmax(cumulative_gains)
        best_num_moves = best_k
        solution_after_rollback = solution_at_pass_start.copy()
        for i in range(best_num_moves):
            if i < len(moves_sequence):
                node_idx = moves_sequence[i]
                if 0 <= node_idx < n:
                    solution_after_rollback[node_idx] = (
                        1 - solution_after_rollback[node_idx]
                    )
                else:
                    solution_after_rollback = solution_at_pass_start.copy()
                    break
            else:
                solution_after_rollback = solution_at_pass_start.copy()
                break

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
        print("CRITICAL ERROR: fm_heuristic final solution is not balanced!")

    return best_solution_overall, passes_done


# --- Revised Metaheuristics (Handling Pass Limits & Time Limits) ---


def MLS(graph, num_starts, max_total_passes=None, time_limit=None):
    """
    Multi-start Local Search.
    Applies FM heuristic from multiple random starting points.
    Stops based on num_starts, max_total_passes (cumulative FM passes), or time_limit.
    """
    best_solution = None
    best_fit = float("inf")
    start_time = time.perf_counter()
    total_passes_consumed = 0  # Cumulative FM passes across all starts

    # Determine the effective number of starts based on limits
    # If only max_total_passes is set, run indefinitely until passes are consumed.
    force_run_to_pass_limit = max_total_passes is not None and time_limit is None
    effective_num_starts = float("inf") if force_run_to_pass_limit else num_starts

    i = 0  # Start counter
    while i < effective_num_starts:
        current_time = time.perf_counter()
        # Check time limit first
        if time_limit is not None and current_time - start_time >= time_limit:
            # print(f"MLS: Time limit ({time_limit}s) reached.")
            break
        # Check cumulative pass limit
        if max_total_passes is not None and total_passes_consumed >= max_total_passes:
            # print(f"MLS: Pass limit ({max_total_passes}) reached.")
            break

        # Generate a new random (balanced) starting solution
        sol = generate_random_solution()

        # Determine max_passes for this specific FM call
        fm_max_passes = float("inf")  # Default: let FM run until convergence
        if max_total_passes is not None:
            # Calculate remaining passes allowed
            remaining_passes = max(1, max_total_passes - total_passes_consumed)
            # Check again if limit already reached before FM call
            if total_passes_consumed >= max_total_passes:
                break
            fm_max_passes = int(remaining_passes)  # Pass the remaining budget to FM

        try:
            # Run FM heuristic
            local_opt, passes_this_call = fm_heuristic(
                sol, graph, max_passes=fm_max_passes
            )
            total_passes_consumed += passes_this_call  # Accumulate passes used

            if local_opt is not None:
                # Ensure FM result is balanced (should be, but safety check)
                if local_opt.count(0) != len(local_opt) // 2:
                    # print("Warning: MLS received imbalanced solution from FM. Re-balancing.")
                    local_opt = balance_child(local_opt)

                # Only consider if balanced
                if local_opt.count(0) == len(local_opt) // 2:
                    fit = fitness_function(local_opt, graph)
                    # Update best solution found so far
                    if fit < best_fit:
                        best_fit = fit
                        best_solution = local_opt.copy()
                # else: print("Warning: MLS could not use FM result due to imbalance after re-balancing.")

        except Exception as e:
            print(f"Error in MLS fm_heuristic call (start {i+1}): {e}")
            # Continue to the next start if an error occurs in FM

        finally:
            # Increment start counter regardless of FM success/failure, unless limits were hit before starting
            if (time_limit is None or current_time - start_time < time_limit) and (
                max_total_passes is None or total_passes_consumed < max_total_passes
            ):
                i += 1

    # Handle cases where no valid solution was found
    if best_solution is None:
        if i > 0:
            print("Warning: MLS completed without finding any valid solution.")
        else:
            print("Warning: MLS did not complete any starts due to immediate limit.")
        best_solution = (
            generate_random_solution()
        )  # Return a random balanced solution as fallback

    return best_solution, total_passes_consumed


# ***************************************************************************
# * REVISED ILS (Simple) - Ignores stagnation when time_limit is active    *
# ***************************************************************************
def ILS(graph, initial_solution, mutation_size, max_total_passes=None, time_limit=None):
    """
    Simple Iterated Local Search.
    Applies mutation and FM heuristic iteratively.
    Accepts only improving solutions.
    Stops based on max_total_passes, time_limit, or stagnation (if no time_limit).
    """
    # --- Initialization ---
    if initial_solution.count(0) != len(initial_solution) // 2:
        best_solution = balance_child(initial_solution.copy())
    else:
        best_solution = initial_solution.copy()

    if best_solution.count(0) != len(best_solution) // 2:  # If balancing failed
        print("Error: ILS could not start with a balanced solution.")
        return generate_random_solution(), 0, 0  # Return random, 0 unchanged, 0 passes

    best_fit = fitness_function(best_solution, graph)
    start_time = time.perf_counter()
    total_passes_consumed = 0
    no_improvement = 0
    max_no_improvement = 10  # Fixed stagnation limit for simple ILS
    unchanged_count = 0  # Counts consecutive iterations with same fitness
    iteration = 0
    # Determine if internal stagnation check should be active
    check_stagnation = time_limit is None and max_total_passes is None

    # --- Iteration Loop ---
    while True:
        iteration += 1
        current_time = time.perf_counter()

        # --- Check Stopping Conditions ---
        # Priority: Time Limit
        if time_limit is not None and current_time - start_time >= time_limit:
            # print(f"ILS: Time limit ({time_limit}s) reached.")
            break
        # Priority: Pass Limit
        if max_total_passes is not None and total_passes_consumed >= max_total_passes:
            # print(f"ILS: Pass limit ({max_total_passes}) reached.")
            break
        # Check Stagnation *only* if no time or pass limit is active
        if check_stagnation and no_improvement >= max_no_improvement:
            # print(f"ILS: Stagnation limit ({max_no_improvement}) reached.")
            break

        # --- Mutation ---
        mutated = mutate(best_solution, mutation_size)
        # Ensure mutation result is balanced
        if mutated.count(0) != len(mutated) // 2:
            mutated = balance_child(mutated)
        if mutated.count(0) != len(mutated) // 2:  # If still not balanced, skip FM
            print("Warning: ILS mutation resulted in imbalance, skipping FM.")
            no_improvement += 1
            continue

        # --- Local Search (FM) ---
        fm_max_passes = float("inf")  # Let FM run until convergence
        if max_total_passes is not None:
            remaining_passes = max(1, max_total_passes - total_passes_consumed)
            # Check again if limit already reached before FM call
            if total_passes_consumed >= max_total_passes:
                break
            fm_max_passes = int(remaining_passes)

        try:
            local_opt, passes_this_call = fm_heuristic(
                mutated, graph, max_passes=fm_max_passes
            )
            total_passes_consumed += passes_this_call

            if local_opt is not None:
                # Ensure FM result is balanced
                if local_opt.count(0) != len(local_opt) // 2:
                    local_opt = balance_child(local_opt)

                if (
                    local_opt.count(0) != len(local_opt) // 2
                ):  # Skip if still imbalanced
                    print("Warning: ILS FM result imbalanced, skipping update.")
                    no_improvement += 1
                    continue

                # --- Acceptance Criterion ---
                fit = fitness_function(local_opt, graph)
                if fit == best_fit:
                    unchanged_count += 1
                else:
                    unchanged_count = 0  # Reset if fitness changes

                # Accept only if strictly better
                if fit < best_fit:
                    best_solution = local_opt.copy()
                    best_fit = fit
                    no_improvement = 0  # Reset stagnation counter
                else:
                    no_improvement += 1  # Increment stagnation counter
            else:
                # FM failed or returned None
                no_improvement += 1
                unchanged_count = 0  # Reset unchanged count if FM fails

        except Exception as e:
            print(f"Error in ILS fm_heuristic call (iteration {iteration}): {e}")
            no_improvement += 1
            unchanged_count = 0
            continue  # Skip to next iteration

    return best_solution, unchanged_count, total_passes_consumed


# ***************************************************************************
# * REVISED ILS Annealing - Ignores stagnation/temp when time_limit active *
# ***************************************************************************
def ILS_annealing(
    graph,
    initial_solution,
    mutation_size,
    temperature=10.0,
    cooling_rate=0.98,
    min_temperature=0.1,
    max_no_improvement=20,
    max_total_passes=None,
    time_limit=None,
):
    """
    Iterated Local Search with Simulated Annealing acceptance criterion.
    Applies mutation and FM heuristic iteratively.
    Accepts improving solutions always, and worsening solutions probabilistically.
    Stops based on max_total_passes, time_limit, stagnation, or temperature (if no time/pass limit).
    """
    # --- Initialization ---
    if initial_solution.count(0) != len(initial_solution) // 2:
        current_solution = balance_child(initial_solution.copy())
    else:
        current_solution = initial_solution.copy()

    if current_solution.count(0) != len(current_solution) // 2:
        print("Error: ILS Annealing could not start with a balanced solution.")
        return generate_random_solution(), 0, 0  # Return random, 0 unchanged, 0 passes

    best_solution = current_solution.copy()
    current_fit = fitness_function(current_solution, graph)
    best_fit = current_fit
    start_time = time.perf_counter()
    total_passes_consumed = 0
    no_improvement_on_best = 0  # Tracks iterations since *best* solution improved
    unchanged_count = 0  # Tracks consecutive iterations with same current_fit
    current_temperature = temperature  # Local variable for temperature decay
    iteration = 0
    # Determine if internal stopping checks (stagnation, temp) should be active
    check_internal_stops = time_limit is None and max_total_passes is None

    # --- Iteration Loop ---
    while True:
        iteration += 1
        current_time = time.perf_counter()

        # --- Check Stopping Conditions ---
        # Priority: Time Limit
        if time_limit is not None and current_time - start_time >= time_limit:
            # print(f"ILS_A: Time limit ({time_limit}s) reached.")
            break
        # Priority: Pass Limit
        if max_total_passes is not None and total_passes_consumed >= max_total_passes:
            # print(f"ILS_A: Pass limit ({max_total_passes}) reached.")
            break
        # Check Internal Stops *only* if no time or pass limit is active
        if check_internal_stops:
            if no_improvement_on_best >= max_no_improvement:
                # print(f"ILS_A: Stagnation limit ({max_no_improvement}) reached.")
                break
            if current_temperature <= min_temperature:
                # print(f"ILS_A: Minimum temperature ({min_temperature}) reached.")
                break

        # --- Mutation ---
        mutated = mutate(
            current_solution, mutation_size
        )  # Mutate the *current* solution
        if mutated.count(0) != len(mutated) // 2:
            mutated = balance_child(mutated)
        if mutated.count(0) != len(mutated) // 2:
            print("Warning: ILS Annealing mutation resulted in imbalance, skipping FM.")
            no_improvement_on_best += 1  # Count as non-improving step for stagnation
            continue

        # --- Local Search (FM) ---
        fm_max_passes = float("inf")  # Let FM run until convergence
        if max_total_passes is not None:
            remaining_passes = max(1, max_total_passes - total_passes_consumed)
            if total_passes_consumed >= max_total_passes:
                break  # Check again
            fm_max_passes = int(remaining_passes)

        try:
            local_opt, passes_this_call = fm_heuristic(
                mutated, graph, max_passes=fm_max_passes
            )
            total_passes_consumed += passes_this_call

            if local_opt is not None:
                if local_opt.count(0) != len(local_opt) // 2:
                    local_opt = balance_child(local_opt)

                if local_opt.count(0) != len(local_opt) // 2:
                    print(
                        "Warning: ILS Annealing FM result imbalanced, skipping update."
                    )
                    no_improvement_on_best += 1
                    continue

                # --- Acceptance Criterion (Simulated Annealing) ---
                fit = fitness_function(local_opt, graph)
                if fit == current_fit:
                    unchanged_count += 1
                else:
                    unchanged_count = 0  # Reset if fitness changes

                delta_e = fit - current_fit  # Change in fitness (positive if worse)

                # Accept if better (delta_e < 0) OR probabilistically if worse
                accept_prob = 0.0
                if delta_e < 0:
                    accept = True
                elif current_temperature > 1e-9:  # Avoid division by zero/near-zero
                    try:
                        accept_prob = math.exp(-delta_e / current_temperature)
                        accept = random.random() < accept_prob
                    except (
                        OverflowError
                    ):  # math.exp can overflow for large delta_e/temp
                        accept = False
                else:  # Temperature is effectively zero, only accept improvements
                    accept = False

                if accept:
                    current_solution = local_opt.copy()
                    current_fit = fit
                    # Update best solution if the accepted solution is better than the overall best
                    if fit < best_fit:
                        best_solution = local_opt.copy()
                        best_fit = fit
                        no_improvement_on_best = (
                            0  # Reset stagnation counter for *best*
                        )
                    else:
                        # Accepted a non-improving solution (or same fitness)
                        no_improvement_on_best += 1
                else:
                    # Rejected worse solution
                    no_improvement_on_best += 1
            else:
                # FM failed or returned None
                no_improvement_on_best += 1
                unchanged_count = 0

        except Exception as e:
            print(f"Error in ILS_A fm_heuristic call (iteration {iteration}): {e}")
            no_improvement_on_best += 1
            unchanged_count = 0
            continue

        # --- Cool Down ---
        # Only cool down if internal stops are active (otherwise temp becomes irrelevant)
        if check_internal_stops:
            current_temperature *= cooling_rate

    return best_solution, unchanged_count, total_passes_consumed


# --- MODIFIED ILS ADAPTIVE ---
def ILS_adaptive(
    graph,
    initial_solution,
    initial_mutation_mean=40,
    mutation_std_dev=5.0,
    decay_factor=0.98,
    increase_factor=1.2,
    stagnation_threshold=50,
    max_iterations=100000,
    min_mutation_size=1,
    max_mutation_size=None,  # Allow setting max mutation
    stagnation_window=15,
    stagnation_tolerance=0.0001,
    restart_reset=True,  # Added params
    max_total_passes=None,
    time_limit=None,
):
    """
    Adaptive Iterated Local Search.
    Adjusts mutation strength based on search progress (stagnation).
    Accepts only improving solutions.
    Stops based on max_total_passes, time_limit, or max_iterations (if no time/pass limit).
    """
    # --- Initialization ---
    if max_mutation_size is None:
        max_mutation_size = NUM_VERTICES // 4  # Default based on original code

    if initial_solution.count(0) != len(initial_solution) // 2:
        current_solution = balance_child(initial_solution.copy())
    else:
        current_solution = initial_solution.copy()

    if current_solution.count(0) != len(current_solution) // 2:
        print("Error: ILS Adaptive could not start with a balanced solution.")
        # Return structure matching success case but indicating failure
        return (
            generate_random_solution(),
            float("inf"),
            [],
            0,
            0,
            int(initial_mutation_mean),
            0,
        )

    best_solution = current_solution.copy()
    current_fit = fitness_function(current_solution, graph)
    best_fit = current_fit
    current_mutation_mean = float(initial_mutation_mean)
    current_mutation_std_dev = float(mutation_std_dev)
    mutation_history = []  # Optional: track mutation parameters over time
    stagnation_count = 0  # Iterations since last improvement in *current* solution
    iteration = 0
    fitness_history = [current_fit]  # Track fitness for stagnation detection
    start_time = time.perf_counter()
    total_passes_consumed = 0
    # Determine if internal iteration limit should be used
    use_max_iterations = time_limit is None and max_total_passes is None
    effective_max_iterations = max_iterations if use_max_iterations else float("inf")

    # --- Iteration Loop ---
    while iteration < effective_max_iterations:
        iteration += 1
        current_time = time.perf_counter()

        # --- Check Stopping Conditions ---
        # Priority: Time Limit
        if time_limit is not None and current_time - start_time >= time_limit:
            # print(f"ILS_Ad: Time limit ({time_limit}s) reached.")
            break
        # Priority: Pass Limit
        if max_total_passes is not None and total_passes_consumed >= max_total_passes:
            # print(f"ILS_Ad: Pass limit ({max_total_passes}) reached.")
            break
        # Max iterations checked by loop condition `while iteration < effective_max_iterations`

        # --- Adaptive Mutation ---
        # Generate mutation size from normal distribution, clamp within bounds
        mutation_size = int(
            np.random.normal(current_mutation_mean, current_mutation_std_dev)
        )
        mutation_size = max(min_mutation_size, min(mutation_size, max_mutation_size))

        mutated = mutate(current_solution, mutation_size)
        if mutated.count(0) != len(mutated) // 2:
            mutated = balance_child(mutated)
        if mutated.count(0) != len(mutated) // 2:
            print("Warning: ILS Adaptive mutation resulted in imbalance, skipping FM.")
            stagnation_count += 1  # Count as stagnation
            continue

        # --- Local Search (FM) ---
        fm_max_passes = float("inf")  # Let FM run until convergence
        if max_total_passes is not None:
            remaining_passes = max(1, max_total_passes - total_passes_consumed)
            if total_passes_consumed >= max_total_passes:
                break  # Check again
            fm_max_passes = int(remaining_passes)

        try:
            local_opt, passes_this_call = fm_heuristic(
                mutated, graph, max_passes=fm_max_passes
            )
            total_passes_consumed += passes_this_call
        except Exception as e:
            print(f"Error in ILS_Ad fm_heuristic call (iteration {iteration}): {e}")
            stagnation_count += 1
            continue

        if local_opt is None:  # FM failed
            stagnation_count += 1
            continue

        if local_opt.count(0) != len(local_opt) // 2:
            local_opt = balance_child(local_opt)
        if local_opt.count(0) != len(local_opt) // 2:
            print("Warning: ILS Adaptive FM result imbalanced, skipping update.")
            stagnation_count += 1
            continue

        # --- Acceptance & Adaptation ---
        local_opt_fit = fitness_function(local_opt, graph)
        fitness_history.append(local_opt_fit)  # Add to history for stagnation check

        # Accept only if strictly better
        if local_opt_fit < current_fit:
            current_solution = local_opt.copy()
            current_fit = local_opt_fit
            stagnation_count = 0  # Reset stagnation counter

            # Adapt mutation parameters downwards (intensification)
            current_mutation_mean = max(
                min_mutation_size, current_mutation_mean * decay_factor
            )
            # Ensure std_dev doesn't go below 1 or too small relative to mean
            current_mutation_std_dev = max(
                1.0,
                min(
                    current_mutation_mean * 0.5, current_mutation_std_dev * decay_factor
                ),
            )

            # Update best solution if current is better than overall best
            if current_fit < best_fit:
                best_solution = current_solution.copy()
                best_fit = current_fit
        else:
            # Did not accept (not better)
            stagnation_count += 1  # Increment stagnation counter

        # Optional: Store history for analysis
        mutation_history.append(
            (
                iteration,
                mutation_size,
                current_mutation_mean,
                current_mutation_std_dev,
                current_fit,
            )
        )

        # --- Stagnation Handling (Parameter Increase) ---
        # Increase mutation strength if stagnation threshold is reached (diversification)
        # Only adapt if internal stops are potentially active (i.e., not limited by time/passes)
        if use_max_iterations and stagnation_count >= stagnation_threshold:
            is_stuck = True  # Assume stuck unless recent fitness changes significantly
            # Check if fitness has actually plateaued recently
            if len(fitness_history) >= stagnation_window:
                # Calculate mean absolute difference in the recent window
                recent_diff = np.mean(
                    np.abs(np.diff(fitness_history[-stagnation_window:]))
                )
                # If average change is above tolerance, not truly stuck
                if recent_diff >= stagnation_tolerance:
                    is_stuck = False

            if is_stuck:
                # Increase mutation mean and std dev
                current_mutation_mean = min(
                    max_mutation_size, current_mutation_mean * increase_factor
                )
                # Increase std dev, but keep it relative to mean and not too small
                new_std_dev = min(
                    current_mutation_mean / 2,
                    current_mutation_std_dev * increase_factor,
                )
                current_mutation_std_dev = max(
                    1.0, new_std_dev
                )  # Ensure std_dev is at least 1

                stagnation_count = 0  # Reset stagnation counter after adapting

                # Optional: Restart from best solution found so far
                if restart_reset:
                    current_solution = best_solution.copy()
                    current_fit = best_fit
                    # Keep the increased mutation parameters for diversification after restart

    # Return tuple includes key results and final parameters
    return (
        best_solution,
        best_fit,
        mutation_history,
        iteration,
        stagnation_count,
        int(current_mutation_mean),
        total_passes_consumed,
    )


def GLS(
    graph, population_size, stopping_crit=None, max_total_passes=None, time_limit=None
):
    """
    Genetic Local Search (Memetic Algorithm).
    Combines evolutionary approach (crossover) with local search (FM).
    Stops based on max_total_passes, time_limit, or generations without improvement (if no time/pass limit).
    """
    start_time = time.perf_counter()
    total_passes_consumed = 0
    population = []
    fitness_values = []
    # Determine if internal stopping criterion (stagnation) should be active
    check_stagnation = (
        time_limit is None and max_total_passes is None and stopping_crit is not None
    )

    # --- Population Initialization ---
    pop_initialized_count = 0
    print(f"GLS: Initializing population (size {population_size})...")
    init_start_time = time.perf_counter()
    while pop_initialized_count < population_size:
        current_time = time.perf_counter()
        # Check limits during initialization
        if time_limit is not None and current_time - start_time >= time_limit:
            break
        if max_total_passes is not None and total_passes_consumed >= max_total_passes:
            break

        initial_sol = generate_random_solution()
        fm_max_passes = float("inf")  # Let FM run fully during init
        if max_total_passes is not None:
            remaining_passes = max(1, max_total_passes - total_passes_consumed)
            if total_passes_consumed >= max_total_passes:
                break
            fm_max_passes = int(remaining_passes)

        try:
            sol_opt, passes_this_call = fm_heuristic(
                initial_sol, graph, max_passes=fm_max_passes
            )
            total_passes_consumed += passes_this_call
            if sol_opt is not None:
                if sol_opt.count(0) != len(sol_opt) // 2:
                    sol_opt = balance_child(sol_opt)
                if sol_opt.count(0) == len(sol_opt) // 2:  # Only add if balanced
                    population.append(sol_opt)
                    fitness_values.append(fitness_function(sol_opt, graph))
                    pop_initialized_count += 1
                # else: print("Warning: GLS init FM produced imbalanced solution.")
        except Exception as e:
            print(
                f"  Error during GLS init fm_heuristic (individual {pop_initialized_count+1}): {e}"
            )
            # Continue trying to initialize the population

    init_end_time = time.perf_counter()
    print(
        f"GLS: Initialization finished ({pop_initialized_count}/{population_size} individuals). Passes: {total_passes_consumed}, Time: {init_end_time - init_start_time:.2f}s"
    )

    # Check if initialization failed or was cut short
    if not population:
        print("Error: GLS could not initialize any population members.")
        return generate_random_solution(), total_passes_consumed  # Return random

    # Find best initial solution
    best_fit_idx = np.argmin(fitness_values) if fitness_values else -1
    if best_fit_idx == -1:
        print("Error: GLS population exists but fitness values are missing.")
        return population[0], total_passes_consumed  # Return first member as fallback

    best_fit = fitness_values[best_fit_idx]
    best_solution = population[best_fit_idx]
    generation_without_improvement = 0
    gen = 0
    # Set a high default max generations if no other limit applies
    effective_max_generations = 100000 if check_stagnation else float("inf")

    # --- Evolutionary Loop ---
    print("GLS: Starting evolutionary loop...")
    while gen < effective_max_generations:
        gen += 1
        current_time = time.perf_counter()

        # --- Check Stopping Conditions ---
        if time_limit is not None and current_time - start_time >= time_limit:
            # print(f"GLS: Time limit ({time_limit}s) reached.")
            break
        if max_total_passes is not None and total_passes_consumed >= max_total_passes:
            # print(f"GLS: Pass limit ({max_total_passes}) reached.")
            break
        if check_stagnation and generation_without_improvement >= stopping_crit:
            # print(f"GLS: Stopping criterion ({stopping_crit} gens without improvement) reached.")
            break
        # Need at least two distinct individuals for crossover
        if len(population) < 2:
            print("GLS: Population size dropped below 2. Stopping.")
            break

        # --- Selection and Crossover ---
        # Simple random selection (could be replaced with tournament, roulette wheel etc.)
        parent1_idx = random.randrange(len(population))
        parent2_idx = random.randrange(len(population))
        # Ensure parents are different for meaningful crossover
        tries = 0
        while parent1_idx == parent2_idx and len(population) > 1 and tries < 10:
            parent2_idx = random.randrange(len(population))
            tries += 1
        # If parents are still the same after tries, proceed anyway (effectively cloning)
        if parent1_idx == parent2_idx:
            print("Warning: GLS selecting same parent twice.")

        parent1 = population[parent1_idx]
        parent2 = population[parent2_idx]
        child = crossover(parent1, parent2)  # Perform crossover
        if child.count(0) != len(child) // 2:
            child = balance_child(child)  # Ensure child is balanced
        if child.count(0) != len(child) // 2:
            print("Warning: GLS crossover produced imbalanced child, skipping FM.")
            # Don't increment stagnation here, as it wasn't a failed *improvement* attempt
            continue

        # --- Local Search (FM) on Child ---
        fm_max_passes = float("inf")  # Let FM run fully
        if max_total_passes is not None:
            remaining_passes = max(1, max_total_passes - total_passes_consumed)
            if total_passes_consumed >= max_total_passes:
                break  # Check again
            fm_max_passes = int(remaining_passes)
        try:
            child_opt, passes_this_call = fm_heuristic(
                child, graph, max_passes=fm_max_passes
            )
            total_passes_consumed += passes_this_call
        except Exception as e:
            print(f"  Error during GLS evolve fm_heuristic (gen {gen}): {e}")
            continue  # Skip replacement if FM fails

        if child_opt is None:
            continue  # FM failed

        if child_opt.count(0) != len(child_opt) // 2:
            child_opt = balance_child(child_opt)
        if child_opt.count(0) != len(child_opt) // 2:
            print(
                "Warning: GLS FM on child produced imbalanced result. Skipping replacement."
            )
            # Don't increment stagnation here
            continue

        # --- Replacement ---
        fit_child = fitness_function(child_opt, graph)
        # Simple steady-state replacement: Replace worst individual if child is better or equal
        # Could use other strategies (e.g., replace parent, generational replacement)
        worst_fit_idx = np.argmax(fitness_values)
        worst_fit = fitness_values[worst_fit_idx]

        improvement_found_this_gen = False
        if fit_child <= worst_fit:  # Using <= allows replacing same-fitness individuals
            population[worst_fit_idx] = child_opt.copy()
            fitness_values[worst_fit_idx] = fit_child
            # Check if this child is the new overall best
            if fit_child < best_fit:
                best_fit = fit_child
                best_solution = child_opt.copy()
                improvement_found_this_gen = True
                # print(f"GLS Gen {gen}: New best found! Fitness: {best_fit}") # Optional log

        # Update stagnation counter
        if improvement_found_this_gen:
            generation_without_improvement = 0
        else:
            generation_without_improvement += 1

    print(f"GLS: Evolutionary loop finished after {gen} generations.")
    return best_solution, total_passes_consumed


# --- Grid Search Function ---
def run_grid_search(graph, grid_search_pass_limit, grid_search_runs):
    """Performs grid search for ILS_adaptive and ILS_annealing"""
    print("\n--- Starting Grid Search for ILS Parameters ---")
    print(
        f"(Using Pass Limit: {grid_search_pass_limit}, Runs per Combo: {grid_search_runs})"
    )

    best_adaptive_params = {}
    best_adaptive_fitness = float("inf")
    adaptive_results = []

    best_annealing_params = {}
    best_annealing_fitness = float("inf")
    annealing_results = []

    # --- ILS Adaptive Grid --- (Revised grid)
    adaptive_param_grid = {
        "initial_mutation_mean": [40, 60, 80, 100],  # Added larger mean
        "mutation_std_dev": [
            2.0,
            5.0,
            10.0,
            20.0,
        ],  # Switched to absolute std dev, wider range
        "decay_factor": [0.9, 0.95, 0.98, 0.99],  # Added slower decay
        "increase_factor": [1.2, 1.5, 2.0],  # Added stronger increase
        "stagnation_threshold": [30, 50, 80],  # Kept as is
        # Fixed params (can be tuned separately if needed)
        # 'stagnation_window': [15],
        # 'stagnation_tolerance': [0.0001]
    }
    adaptive_keys = list(adaptive_param_grid.keys())
    adaptive_combinations = list(itertools.product(*adaptive_param_grid.values()))
    print(f"Adaptive ILS Grid Search: {len(adaptive_combinations)} combinations")

    for combo_idx, combo in enumerate(adaptive_combinations):
        params = dict(zip(adaptive_keys, combo))
        # Add fixed params not in the grid
        params["stagnation_window"] = 15
        params["stagnation_tolerance"] = 0.0001
        params["min_mutation_size"] = 1
        params["max_mutation_size"] = NUM_VERTICES // 4  # ~125
        params["restart_reset"] = True
        params["max_iterations"] = 100000  # Will be limited by passes/time

        combo_fitnesses = []
        # Shortened param display for logging
        display_params = {
            k: f"{v:.2f}" if isinstance(v, float) else v
            for k, v in params.items()
            if k in adaptive_keys
        }
        print(
            f"  ({combo_idx+1}/{len(adaptive_combinations)}) Testing Adaptive Params: { display_params }"
        )

        run_passes = []
        for r in range(grid_search_runs):
            initial_sol = generate_random_solution()
            # Run ILS_adaptive with the current parameter combination
            # Pass limit is the primary stopping criterion here
            res_tuple = ILS_adaptive(
                graph,
                initial_sol,
                max_total_passes=grid_search_pass_limit,
                time_limit=None,  # No time limit for grid search
                **params,
            )  # Pass parameters using dictionary unpacking
            combo_fitnesses.append(res_tuple[1])  # Index 1 is best_fit
            run_passes.append(res_tuple[6])  # Index 6 is total_passes_consumed

        # Calculate average fitness for this parameter combination
        avg_fitness = np.mean(combo_fitnesses) if combo_fitnesses else float("inf")
        avg_passes = np.mean(run_passes) if run_passes else 0
        adaptive_results.append({"params": params, "avg_fitness": avg_fitness})
        print(f"     Avg Fitness: {avg_fitness:.2f}, Avg Passes: {avg_passes:.1f}")

        # Update best parameters if this combination is better
        if avg_fitness < best_adaptive_fitness:
            best_adaptive_fitness = avg_fitness
            best_adaptive_params = params

    print(f"\nBest Adaptive Params Found: {best_adaptive_params}")
    print(f"Best Adaptive Avg Fitness: {best_adaptive_fitness:.2f}")

    # --- ILS Annealing Grid --- (Revised grid)
    annealing_param_grid = {
        "mutation_size": [10, 25, 40, 60, 80],  # Added smaller size
        "temperature": [10.0, 50.0, 150.0, 300.0],  # Increased range significantly
        "cooling_rate": [0.9, 0.98, 0.99, 0.995],  # Added slower cooling
        "min_temperature": [0.1, 0.5],  # Kept as is
        "max_no_improvement": [10, 20, 40, 60],  # Increased range
    }
    annealing_keys = list(annealing_param_grid.keys())
    annealing_combinations = list(itertools.product(*annealing_param_grid.values()))
    print(f"\nAnnealing ILS Grid Search: {len(annealing_combinations)} combinations")

    for combo_idx, combo in enumerate(annealing_combinations):
        params = dict(zip(annealing_keys, combo))
        combo_fitnesses = []
        display_params = {
            k: f"{v:.3f}" if isinstance(v, float) else v for k, v in params.items()
        }  # Show more precision for cooling
        print(
            f"  ({combo_idx+1}/{len(annealing_combinations)}) Testing Annealing Params: { display_params }"
        )

        run_passes = []
        for r in range(grid_search_runs):
            initial_sol = generate_random_solution()
            # Run ILS_annealing with the current parameter combination
            best_sol, _, passes_consumed = ILS_annealing(
                graph,
                initial_sol,
                max_total_passes=grid_search_pass_limit,
                time_limit=None,  # No time limit
                **params,  # Pass parameters
            )
            # Calculate fitness of the result
            fit = fitness_function(best_sol, graph) if best_sol else float("inf")
            combo_fitnesses.append(fit)
            run_passes.append(passes_consumed)

        # Calculate average fitness
        avg_fitness = np.mean(combo_fitnesses) if combo_fitnesses else float("inf")
        avg_passes = np.mean(run_passes) if run_passes else 0
        annealing_results.append({"params": params, "avg_fitness": avg_fitness})
        print(f"     Avg Fitness: {avg_fitness:.2f}, Avg Passes: {avg_passes:.1f}")

        # Update best parameters
        if avg_fitness < best_annealing_fitness:
            best_annealing_fitness = avg_fitness
            best_annealing_params = params

    print(f"\nBest Annealing Params Found: {best_annealing_params}")
    print(f"Best Annealing Avg Fitness: {best_annealing_fitness:.2f}")

    print("--- Grid Search Finished ---")
    return best_adaptive_params, best_annealing_params


# --- Revised Experiment Runners ---
def run_MLS_experiment(graph):
    """Runs MLS 10 times, limited by FM_PASSES_LIMIT cumulative passes."""
    print("--- Running MLS Pass-Limited Experiment ---")
    # High num_starts ensures pass limit is the primary stop condition
    num_starts = FM_PASSES_LIMIT * 2  # Set high enough
    results = []
    for i in range(EXPERIMENT_RUNS):
        print(f"   MLS Pass-Limited Run {i+1}/{EXPERIMENT_RUNS}")
        start_time = time.perf_counter()
        # Run MLS with pass limit, no time limit
        best_sol, passes_consumed = MLS(
            graph, num_starts, max_total_passes=FM_PASSES_LIMIT, time_limit=None
        )
        end_time = time.perf_counter()
        comp_time = end_time - start_time
        fit = fitness_function(best_sol, graph) if best_sol else float("inf")
        solution_str = "".join(map(str, best_sol)) if best_sol else ""
        # Results tuple structure: (run_idx, solution, fitness, time, passes, param1_name, param1_val, param2_name, param2_val)
        results.append(
            (i, solution_str, fit, comp_time, passes_consumed, "", "", "", "")
        )  # Keep structure consistent
        print(
            f"     Run {i+1}: Fitness={fit}, Passes={passes_consumed}, Time={comp_time:.4f}s"
        )
    return results


def run_simple_ILS_experiment(graph):
    """Runs Simple ILS 10 times per mutation size, limited by FM_PASSES_LIMIT cumulative passes."""
    print("--- Running Simple ILS Pass-Limited Experiment ---")
    # Keep testing different mutation sizes for simple ILS as it has fewer params
    mutation_sizes = [10, 25, 40, 50, 60, 75, 100, 150]
    results = []
    for mutation_size in mutation_sizes:
        print(f"   Simple ILS Testing Mutation Size: {mutation_size}")
        for i in range(EXPERIMENT_RUNS):
            initial_sol = generate_random_solution()
            start_time = time.perf_counter()
            # Run simple ILS with pass limit, no time limit
            best_sol, unchanged_count, passes_consumed = ILS(
                graph,
                initial_sol,
                mutation_size,
                max_total_passes=FM_PASSES_LIMIT,
                time_limit=None,
            )
            end_time = time.perf_counter()
            comp_time = end_time - start_time
            fit = fitness_function(best_sol, graph) if best_sol else float("inf")
            solution_str = "".join(map(str, best_sol)) if best_sol else ""
            unchanged_count = (
                unchanged_count if best_sol else -1
            )  # Indicate error if no solution
            # Results tuple: (run_idx, solution, fitness, time, passes, param1_name, param1_val, param2_name, param2_val)
            results.append(
                (
                    i,
                    solution_str,
                    fit,
                    comp_time,
                    passes_consumed,
                    "Mutation_Size",
                    mutation_size,
                    "Unchanged_Count",
                    unchanged_count,
                )
            )
            print(
                f"       Run {i+1}: Fitness={fit}, Passes={passes_consumed}, Unchanged={unchanged_count}, Time={comp_time:.4f}s"
            )
    return results


def run_GLS_experiment(graph):
    """Runs GLS 10 times for stopping_crit=1, limited by FM_PASSES_LIMIT cumulative passes."""
    print("--- Running GLS Pass-Limited Experiment ---")
    stopping_crit = 1  # Use only stopping_crit=1 as requested
    results = []
    print(f"   GLS Testing Stopping Criterion: {stopping_crit}")
    for i in range(EXPERIMENT_RUNS):
        start_time = time.perf_counter()
        # Run GLS with pass limit, no time limit
        best_sol, passes_consumed = GLS(
            graph,
            POPULATION_SIZE,
            stopping_crit,
            max_total_passes=FM_PASSES_LIMIT,
            time_limit=None,
        )
        end_time = time.perf_counter()
        comp_time = end_time - start_time
        fit = fitness_function(best_sol, graph) if best_sol else float("inf")
        solution_str = "".join(map(str, best_sol)) if best_sol else ""
        # Results tuple: (run_idx, solution, fitness, time, passes, param1_name, param1_val, param2_name, param2_val)
        results.append(
            (
                i,
                solution_str,
                fit,
                comp_time,
                passes_consumed,
                "Stopping_Crit",
                stopping_crit,
                "",
                "",
            )
        )
        print(
            f"       Run {i+1}: Fitness={fit}, Passes={passes_consumed}, Time={comp_time:.4f}s"
        )
    return results


# MODIFIED Experiment Runner for Adaptive ILS
def run_ILS_adaptive_experiment(graph, best_params):
    """Runs Adaptive ILS 10 times with best params, limited by FM_PASSES_LIMIT."""
    print("--- Running ILS Adaptive Pass-Limited Experiment (Best Params) ---")
    results = []
    if not best_params:
        print(
            "   WARNING: No best parameters found for Adaptive ILS. Skipping experiment."
        )
        return results
    # Create a clean copy for the experiment run if needed, or ensure params are not modified
    exp_params = best_params.copy()
    print(f"   Using Parameters: {exp_params}")
    for i in range(EXPERIMENT_RUNS):
        initial_sol = generate_random_solution()
        start_time = time.perf_counter()
        # Run adaptive ILS with pass limit, no time limit, using best params
        res_tuple = ILS_adaptive(
            graph,
            initial_sol,
            max_total_passes=FM_PASSES_LIMIT,
            time_limit=None,
            **exp_params,
        )
        end_time = time.perf_counter()
        comp_time = end_time - start_time
        # Unpack results from ILS_adaptive return tuple
        (
            best_sol,
            best_fit_val,
            _,
            _,
            final_stag_count,
            final_mut_mean,
            passes_consumed,
        ) = res_tuple

        if best_sol is None:
            fit = float("inf")
            solution_str = ""
            final_stag_count = -1
            final_mut_mean = -1
        else:
            fit = best_fit_val
            solution_str = "".join(map(str, best_sol))
        # Results tuple: (run_idx, solution, fitness, time, passes, param1_name, param1_val, param2_name, param2_val)
        # Store key final parameters for analysis
        results.append(
            (
                i,
                solution_str,
                fit,
                comp_time,
                passes_consumed,
                "Final_MutMean",
                final_mut_mean,
                "Final_Stagnation",
                final_stag_count,
            )
        )
        print(
            f"     Run {i+1}: Fitness={fit}, Passes={passes_consumed}, Stagnation={final_stag_count}, MutMean={final_mut_mean}, Time={comp_time:.4f}s"
        )
    return results


# MODIFIED Experiment Runner for Annealing ILS
def run_ILS_annealing_experiment(graph, best_params):
    """Runs ILS Annealing 10 times with best params, limited by FM_PASSES_LIMIT."""
    print("--- Running ILS Annealing Pass-Limited Experiment (Best Params) ---")
    results = []
    if not best_params:
        print(
            "   WARNING: No best parameters found for Annealing ILS. Skipping experiment."
        )
        return results
    # Create a clean copy for the experiment run
    exp_params = best_params.copy()
    print(f"   Using Parameters: {exp_params}")
    mutation_size = exp_params.get(
        "mutation_size", 50
    )  # Extract mutation size for logging if needed

    for i in range(EXPERIMENT_RUNS):
        initial_sol = generate_random_solution()
        start_time = time.perf_counter()
        # Run annealing ILS with pass limit, no time limit, using best params
        best_sol, unchanged_count, passes_consumed = ILS_annealing(
            graph,
            initial_sol,
            max_total_passes=FM_PASSES_LIMIT,
            time_limit=None,
            **exp_params,
        )
        end_time = time.perf_counter()
        comp_time = end_time - start_time
        if best_sol is None:
            fit = float("inf")
            solution_str = ""
            unchanged_count = -1
        else:
            fit = fitness_function(best_sol, graph)
            solution_str = "".join(map(str, best_sol))
        # Results tuple: (run_idx, solution, fitness, time, passes, param1_name, param1_val, param2_name, param2_val)
        results.append(
            (
                i,
                solution_str,
                fit,
                comp_time,
                passes_consumed,
                "Mutation_Size",
                mutation_size,
                "Unchanged_Count",
                unchanged_count,
            )
        )
        print(
            f"     Run {i+1}: Fitness={fit}, Passes={passes_consumed}, Unchanged={unchanged_count}, Time={comp_time:.4f}s"
        )
    return results


# --- Runtime Experiment Runners ---
def run_MLS_runtime_experiment(graph, time_limit):
    """Runs MLS with a time limit and returns the actual duration."""
    start_run_time = time.perf_counter()  # Start timer
    # Set num_starts very high so time_limit is the effective constraint
    best_sol, passes_consumed = MLS(
        graph, num_starts=1000000, time_limit=time_limit, max_total_passes=None
    )
    end_run_time = time.perf_counter()  # End timer
    duration = end_run_time - start_run_time  # Calculate actual duration

    if best_sol is None:
        fit = float("inf")
        solution_str = ""
    else:
        fit = fitness_function(best_sol, graph)
        solution_str = "".join(map(str, best_sol))

    # Return tuple: (run_idx, solution, fitness, actual_duration, passes, ...)
    return [(0, solution_str, fit, duration, passes_consumed, "", "", "", "")]


def run_ILS_runtime_experiment(graph, time_limit, mutation_size):
    """Runs Simple ILS with a time limit and returns the actual duration."""
    initial_sol = generate_random_solution()
    start_run_time = time.perf_counter()  # Start timer
    # Run simple ILS with time limit, no pass limit
    best_sol, unchanged_count, passes_consumed = ILS(
        graph, initial_sol, mutation_size, time_limit=time_limit, max_total_passes=None
    )
    end_run_time = time.perf_counter()  # End timer
    duration = end_run_time - start_run_time  # Calculate actual duration

    if best_sol is None:
        fit = float("inf")
        solution_str = ""
        unchanged_count = -1
    else:
        fit = fitness_function(best_sol, graph)
        solution_str = "".join(map(str, best_sol))

    # Return tuple: (run_idx, solution, fitness, actual_duration, passes, ...)
    return [
        (
            0,
            solution_str,
            fit,
            duration,
            passes_consumed,
            "Mutation_Size",
            mutation_size,
            "Unchanged_Count",
            unchanged_count,
        )
    ]


def run_GLS_runtime_experiment(graph, time_limit, stopping_crit):
    """Runs GLS with a time limit and returns the actual duration."""
    start_run_time = time.perf_counter()  # Start timer
    # GLS stopping_crit is ignored when time_limit is set, but pass it anyway
    best_sol, passes_consumed = GLS(
        graph,
        POPULATION_SIZE,
        stopping_crit,
        time_limit=time_limit,
        max_total_passes=None,
    )
    end_run_time = time.perf_counter()  # End timer
    duration = end_run_time - start_run_time  # Calculate actual duration

    if best_sol is None:
        fit = float("inf")
        solution_str = ""
    else:
        fit = fitness_function(best_sol, graph)
        solution_str = "".join(map(str, best_sol))

    # Return tuple: (run_idx, solution, fitness, actual_duration, passes, ...)
    return [
        (
            0,
            solution_str,
            fit,
            duration,
            passes_consumed,
            "Stopping_Crit",
            stopping_crit,
            "",
            "",
        )
    ]


def run_ILS_adaptive_runtime_experiment(graph, time_limit, best_params):
    """Runs Adaptive ILS with a time limit using best params and returns the actual duration."""
    if not best_params:
        print(
            "   WARNING: No best parameters for Adaptive ILS runtime. Using defaults."
        )
        # Fallback to some reasonable defaults if grid search failed
        best_params = {
            "initial_mutation_mean": 40,
            "mutation_std_dev": 5.0,
            "decay_factor": 0.98,
            "increase_factor": 1.2,
            "stagnation_threshold": 50,
            "max_iterations": 100000,
            "min_mutation_size": 1,
            "max_mutation_size": NUM_VERTICES // 4,
            "stagnation_window": 15,
            "stagnation_tolerance": 0.0001,
            "restart_reset": True,
        }
    exp_params = best_params.copy()  # Use a copy

    initial_sol = generate_random_solution()
    start_run_time = time.perf_counter()  # Start timer
    # Run adaptive ILS with time limit, no pass limit
    res_tuple = ILS_adaptive(
        graph, initial_sol, time_limit=time_limit, max_total_passes=None, **exp_params
    )
    end_run_time = time.perf_counter()  # End timer
    duration = end_run_time - start_run_time  # Calculate actual duration

    best_sol, best_fit_val, _, _, final_stag_count, final_mut_mean, passes_consumed = (
        res_tuple
    )
    if best_sol is None:
        fit = float("inf")
        solution_str = ""
        final_stag_count = -1
        final_mut_mean = -1
    else:
        fit = best_fit_val
        solution_str = "".join(map(str, best_sol))

    # Return tuple: (run_idx, solution, fitness, actual_duration, passes, ...)
    return [
        (
            0,
            solution_str,
            fit,
            duration,
            passes_consumed,
            "Final_MutMean",
            final_mut_mean,
            "Final_Stagnation",
            final_stag_count,
        )
    ]


def run_ILS_annealing_runtime_experiment(graph, time_limit, best_params):
    """Runs ILS Annealing with a time limit using best params and returns the actual duration."""
    if not best_params:
        print(
            "   WARNING: No best parameters for Annealing ILS runtime. Using defaults."
        )
        # Fallback to some reasonable defaults
        best_params = {
            "mutation_size": 50,
            "temperature": 10.0,
            "cooling_rate": 0.98,
            "min_temperature": 0.1,
            "max_no_improvement": 20,
        }
    exp_params = best_params.copy()  # Use a copy
    mutation_size = exp_params.get("mutation_size", 50)  # Extract for result tuple

    initial_sol = generate_random_solution()
    start_run_time = time.perf_counter()  # Start timer
    # Run annealing ILS with time limit, no pass limit
    best_sol, unchanged_count, passes_consumed = ILS_annealing(
        graph, initial_sol, time_limit=time_limit, max_total_passes=None, **exp_params
    )
    end_run_time = time.perf_counter()  # End timer
    duration = end_run_time - start_run_time  # Calculate actual duration

    if best_sol is None:
        fit = float("inf")
        solution_str = ""
        unchanged_count = -1
    else:
        fit = fitness_function(best_sol, graph)
        solution_str = "".join(map(str, best_sol))

    # Return tuple: (run_idx, solution, fitness, actual_duration, passes, ...)
    return [
        (
            0,
            solution_str,
            fit,
            duration,
            passes_consumed,
            "Mutation_Size",
            mutation_size,
            "Unchanged_Count",
            unchanged_count,
        )
    ]


# --- Helper Function for Intermediate Saving ---
def save_intermediate_results(
    results_list, script_dir, filename="experiment_results_intermediate.csv"
):
    """Saves the current list of results to a CSV file."""
    if not results_list:
        print("No results to save yet.")
        return

    print(f"Attempting to save intermediate results ({len(results_list)} rows)...")
    df_intermediate = pd.DataFrame(results_list)
    # Define columns order, ensuring parameter columns are included
    final_columns = [
        "Experiment",
        "Run",
        "Param1_Name",
        "Param1_Value",
        "Param2_Name",
        "Param2_Value",
        "Fitness",
        "Comp_Time",
        "Actual_Passes",
        "Solution",
    ]
    # Add any columns present in the DataFrame but missing from the list
    for col in df_intermediate.columns:
        if col not in final_columns:
            final_columns.append(col)

    # Reindex and fill missing
    df_intermediate = df_intermediate.reindex(columns=final_columns)
    fill_values = {
        "Param1_Name": "",
        "Param1_Value": pd.NA,
        "Param2_Name": "",
        "Param2_Value": pd.NA,
        "Actual_Passes": -1,
        "Fitness": float("inf"),
        "Comp_Time": -1.0,
        "Solution": "",
    }
    for col, val in fill_values.items():
        if col in df_intermediate.columns:
            df_intermediate[col].fillna(val, inplace=True)
        else:
            df_intermediate[col] = val

    # Convert types
    df_intermediate["Run"] = pd.to_numeric(
        df_intermediate["Run"], errors="coerce"
    ).astype("Int64")
    df_intermediate["Fitness"] = pd.to_numeric(
        df_intermediate["Fitness"], errors="coerce"
    )
    df_intermediate["Comp_Time"] = pd.to_numeric(
        df_intermediate["Comp_Time"], errors="coerce"
    )
    df_intermediate["Actual_Passes"] = pd.to_numeric(
        df_intermediate["Actual_Passes"], errors="coerce"
    ).astype("Int64")
    df_intermediate["Param1_Value"] = pd.to_numeric(
        df_intermediate["Param1_Value"], errors="ignore"
    )
    df_intermediate["Param2_Value"] = pd.to_numeric(
        df_intermediate["Param2_Value"], errors="ignore"
    )

    output_path = os.path.join(script_dir, filename)
    try:
        df_intermediate.to_csv(output_path, index=False, float_format="%.6f")
        print(f"Intermediate results saved to '{output_path}'.")
    except Exception as e:
        print(f"\nError saving intermediate results to CSV: {e}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Setup ---
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Handle cases where __file__ might not be defined (e.g., interactive, notebooks)
        if not script_dir or not os.path.exists(script_dir):
            script_dir = os.getcwd()
            print(
                f"Warning: Could not determine script directory reliably. Using current working directory: {script_dir}"
            )

        graph_file_path = os.path.join(script_dir, "Graph500.txt")
        if not os.path.exists(graph_file_path):
            # Try looking in the current working directory as a fallback
            graph_file_path = os.path.join(os.getcwd(), "Graph500.txt")
            if not os.path.exists(graph_file_path):
                raise FileNotFoundError(
                    f"Graph file 'Graph500.txt' not found in script directory or CWD."
                )

        graph = read_graph_data(graph_file_path)
        print("Graph data loaded successfully.")
    except Exception as e:
        print(f"Error during setup: {e}")
        exit()

    # --- Determine Time Limit ---
    # --- !!! IMPORTANT: Uncomment and run this block ONCE to find your DYNAMIC_TIME_LIMIT !!! ---
    # print("\n--- Measuring Time for 10000 FM Passes ---")
    # NUM_TIMING_RUNS = 5; PASSES_TO_TIME = 10000; fm_times = [] # Reduced runs for faster timing
    # print(f"(Running MLS {NUM_TIMING_RUNS} times with cumulative pass limit of {PASSES_TO_TIME})")
    # for i in range(NUM_TIMING_RUNS):
    #     print(f"   Timing Run {i + 1}/{NUM_TIMING_RUNS}...")
    #     initial_sol = generate_random_solution()
    #     start_time = time.perf_counter()
    #     try:
    #         # Use MLS as it purely accumulates FM passes until the limit
    #         _, passes_executed = MLS(graph, num_starts=PASSES_TO_TIME*2, max_total_passes=PASSES_TO_TIME)
    #         end_time = time.perf_counter()
    #         # Check if it ran close to the limit
    #         if passes_executed >= PASSES_TO_TIME * 0.90 : # Allow 10% undershoot
    #             duration = end_time - start_time
    #             fm_times.append(duration)
    #             print(f"     Time for {passes_executed} passes: {duration:.6f} seconds")
    #         else: print(f"     Warning: MLS run finished too early ({passes_executed} passes). Discarding time.")
    #     except Exception as e: print(f"     Error during MLS timing run: {e}")
    # if fm_times:
    #     median_fm_time = np.median(fm_times)
    #     print(f"\n   Median Time for ~{PASSES_TO_TIME} passes: {median_fm_time:.6f} seconds")
    #     print(f"   >>> Set DYNAMIC_TIME_LIMIT = {median_fm_time:.6f} below <<<")
    # else: print("\n   No valid FM timing runs completed. Cannot set DYNAMIC_TIME_LIMIT automatically.")
    # print("--- End of FM Timing Measurement ---")
    # exit() # Exit after timing
    # --- !!! End of Temporary Timing Block !!! ---

    # --- HARDCODE DYNAMIC TIME LIMIT HERE (Replace with value from timing block) ---
    DYNAMIC_TIME_LIMIT = 60
    # -----------------------------------------
    print(
        f"\nUsing Dynamic Time Limit: {DYNAMIC_TIME_LIMIT:.6f} seconds for runtime experiments."
    )

    # --- Run Grid Search ---
    # Run grid search to find best parameters before the main experiments
    # Use a smaller pass limit for grid search to make it faster
    # --- Intermediate Save Filenames ---
    intermediate_results_file = "experiment_results_intermediate.csv"
    intermediate_params_file = "best_params_intermediate.json"

    # --- Run Grid Search ---
    best_adaptive_params, best_annealing_params = {}, {}  # Initialize
    try:
        best_adaptive_params, best_annealing_params = run_grid_search(
            graph, GRID_SEARCH_PASS_LIMIT, GRID_SEARCH_RUNS
        )

        # --- Save Best Params Immediately After Grid Search ---
        params_to_save = {
            "best_adaptive_params": best_adaptive_params,
            "best_annealing_params": best_annealing_params,
        }
        params_path = os.path.join(script_dir, intermediate_params_file)
        try:
            with open(params_path, "w") as f:
                json.dump(params_to_save, f, indent=4)
            print(f"Intermediate best parameters saved to '{params_path}'")
        except Exception as e:
            print(f"Error saving intermediate parameters: {e}")
        # -----------------------------------------------------

    except Exception as e:
        print(f"\n!!! Error during Grid Search: {e} !!!")
        print("Skipping subsequent experiments that depend on grid search results.")
        # Optionally exit or try to load params if they exist
        params_path = os.path.join(script_dir, intermediate_params_file)
        if os.path.exists(params_path):
            try:
                print(
                    f"Attempting to load previously saved parameters from {params_path}"
                )
                with open(params_path, "r") as f:
                    loaded_params = json.load(f)
                best_adaptive_params = loaded_params.get("best_adaptive_params", {})
                best_annealing_params = loaded_params.get("best_annealing_params", {})
                print("Successfully loaded parameters.")
            except Exception as load_e:
                print(
                    f"Failed to load parameters: {load_e}. ILS Adaptive/Annealing may use defaults."
                )
        else:
            print("No intermediate parameters file found.")
    # print(f"DEBUG: Best Adaptive Params from Grid Search: {best_adaptive_params}")
    # print(f"DEBUG: Best Annealing Params from Grid Search: {best_annealing_params}")

    # --- Prepare for Main Experiments ---
    all_results_list = []  # Stores dicts for final DataFrame

    # --- Function to handle appending results and intermediate saving ---
    def append_and_save(new_results, experiment_name):
        print(f"Appending results for {experiment_name}...")
        # Append results based on the structure returned by experiment runners
        for r in new_results:
            result_dict = {
                "Experiment": experiment_name,
                "Run": r[0],
                "Fitness": r[2],
                "Comp_Time": r[3],
                "Actual_Passes": r[4],
                "Solution": r[1],  # Solution string
            }
            # Add parameter info if available (check length of tuple)
            if len(r) > 5:
                result_dict["Param1_Name"] = r[5]
            if len(r) > 6:
                result_dict["Param1_Value"] = r[6]
            if len(r) > 7:
                result_dict["Param2_Name"] = r[7]
            if len(r) > 8:
                result_dict["Param2_Value"] = r[8]
            all_results_list.append(result_dict)
        # Save intermediate results after appending
        save_intermediate_results(
            all_results_list, script_dir, intermediate_results_file
        )

    # -----------------------------
    # Pass-Limited Experiments (Comparison a)
    # -----------------------------
    print(
        f"\n--- Running Pass-Limited Experiments ({EXPERIMENT_RUNS} runs each, Limit: {FM_PASSES_LIMIT} passes) ---"
    )

    # Run MLS
    mls_results_pass = run_MLS_experiment(graph)
    for r in mls_results_pass:
        append_and_save([r], "MLS_PassBased")

    # Run Simple ILS
    simple_ils_results_pass = run_simple_ILS_experiment(graph)
    for r in simple_ils_results_pass:
        append_and_save([r], "Simple_ILS_PassBased")

    # Run GLS
    gls_results_pass = run_GLS_experiment(graph)
    for r in gls_results_pass:
        append_and_save([r], "GLS_PassBased")

    # Run ILS Adaptive with BEST parameters found
    ils_adaptive_results_pass = run_ILS_adaptive_experiment(graph, best_adaptive_params)
    for r in ils_adaptive_results_pass:
        append_and_save([r], "ILS_Adaptive_PassBased")

    # Run ILS Annealing with BEST parameters found
    ils_annealing_results_pass = run_ILS_annealing_experiment(
        graph, best_annealing_params
    )
    for r in ils_annealing_results_pass:
        append_and_save([r], "ILS_Annealing_PassBased")

    # --- Determine best parameters for Simple ILS Runtime tests from pass-based runs ---
    # (Adaptive and Annealing use grid search results directly)
    print("\n--- Determining Best Parameters for Simple ILS Runtime Test ---")
    best_ils_mutation_size = 50  # Default fallback
    try:
        # Filter results for Simple ILS pass-based runs
        df_simple_ils = pd.DataFrame(
            [r for r in all_results_list if r["Experiment"] == "Simple_ILS_PassBased"]
        )
        if (
            not df_simple_ils.empty
            and "Param1_Value" in df_simple_ils.columns
            and "Fitness" in df_simple_ils.columns
        ):
            # Ensure types are numeric for aggregation
            df_simple_ils["Param1_Value"] = pd.to_numeric(
                df_simple_ils["Param1_Value"], errors="coerce"
            )
            df_simple_ils["Fitness"] = pd.to_numeric(
                df_simple_ils["Fitness"], errors="coerce"
            )
            df_simple_ils.dropna(subset=["Param1_Value", "Fitness"], inplace=True)

            if not df_simple_ils.empty:
                # Group by mutation size (Param1_Value) and find the median fitness
                median_fits = df_simple_ils.groupby("Param1_Value")["Fitness"].median()
                if not median_fits.empty:
                    # Find the mutation size corresponding to the minimum median fitness
                    best_ils_mutation_size = int(median_fits.idxmin())
                    print(
                        f"Best Simple ILS Mutation Size from Pass-Based runs: {best_ils_mutation_size} (Median Fitness: {median_fits.min():.2f})"
                    )
                else:
                    print("Could not calculate median fitness for Simple ILS.")
            else:
                print("No valid numeric data for Simple ILS parameter determination.")
        else:
            print("Simple ILS results missing required columns or empty.")
    except Exception as e:
        print(
            f"Could not determine best Simple ILS param: {e}. Using default: {best_ils_mutation_size}"
        )

    # Keep GLS stopping crit fixed at 1 as per original logic/request
    best_gls_stopping_crit = 1

    print(f"\nParameters for Runtime Experiments:")
    print(f"  Simple ILS Mutation Size: {best_ils_mutation_size}")
    print(f"  GLS Stopping Crit: {best_gls_stopping_crit}")
    print(f"  Adaptive ILS: Using best parameters from grid search.")
    print(f"  Annealing ILS: Using best parameters from grid search.")

    # -----------------------------
    # Runtime experiments (Comparison b - Fixed Time Limit)
    # -----------------------------
    print(
        f"\n--- Running Runtime Experiments ({RUNTIME_RUNS} runs each, Limit: {DYNAMIC_TIME_LIMIT:.6f}s) ---"
    )
    for i in range(RUNTIME_RUNS):
        print(f"\nRuntime Repetition {i + 1}/{RUNTIME_RUNS}")

        # MLS Runtime
        print("  Running MLS Runtime...")
        mls_run_res = run_MLS_runtime_experiment(graph, DYNAMIC_TIME_LIMIT)
        append_and_save([mls_run_res[0]], "MLS_Runtime")
        print(f"    MLS Done: Fitness={mls_run_res[0][2]}, Passes={mls_run_res[0][4]}")

        # Simple ILS Runtime (using best mutation size found)
        print(f"  Running Simple ILS Runtime (Mut={best_ils_mutation_size})...")
        ils_s_run_res = run_ILS_runtime_experiment(
            graph, DYNAMIC_TIME_LIMIT, best_ils_mutation_size
        )
        append_and_save([ils_s_run_res[0]], "ILS_Simple_Runtime")
        print(
            f"    Simple ILS Done: Fitness={ils_s_run_res[0][2]}, Passes={ils_s_run_res[0][4]}"
        )

        # GLS Runtime (using fixed stopping crit)
        print(f"  Running GLS Runtime (StopCrit={best_gls_stopping_crit})...")
        gls_run_res = run_GLS_runtime_experiment(
            graph, DYNAMIC_TIME_LIMIT, best_gls_stopping_crit
        )
        append_and_save([gls_run_res[0]], "GLS_Runtime")
        print(f"    GLS Done: Fitness={gls_run_res[0][2]}, Passes={gls_run_res[0][4]}")

        # Adaptive ILS Runtime (using best params from grid search)
        print("  Running Adaptive ILS Runtime (Best Params)...")
        ils_ad_run_res = run_ILS_adaptive_runtime_experiment(
            graph, DYNAMIC_TIME_LIMIT, best_adaptive_params
        )
        append_and_save([ils_ad_run_res[0]], "ILS_Adaptive_Runtime")
        print(
            f"    Adaptive ILS Done: Fitness={ils_ad_run_res[0][2]}, Passes={ils_ad_run_res[0][4]}"
        )

        # Annealing ILS Runtime (using best params from grid search)
        print("  Running Annealing ILS Runtime (Best Params)...")
        ils_a_run_res = run_ILS_annealing_runtime_experiment(
            graph, DYNAMIC_TIME_LIMIT, best_annealing_params
        )
        append_and_save([ils_a_run_res[0]], "ILS_Annealing_Runtime")
        print(
            f"    Annealing ILS Done: Fitness={ils_a_run_res[0][2]}, Passes={ils_a_run_res[0][4]}"
        )

    # -----------------------------
    # Final DataFrame Creation and Saving
    # -----------------------------
    print("\n--- Aggregating All Results ---")
    df_experiments = pd.DataFrame(all_results_list)

    # Define columns order, ensuring parameter columns are included
    final_columns = [
        "Experiment",
        "Run",
        "Param1_Name",
        "Param1_Value",
        "Param2_Name",
        "Param2_Value",  # Generic parameter columns
        "Fitness",
        "Comp_Time",
        "Actual_Passes",
        "Solution",  # Keep solution if needed, can make file large
    ]
    # Add any columns present in the DataFrame but missing from the list (just in case)
    for col in df_experiments.columns:
        if col not in final_columns:
            final_columns.append(col)

    # Reindex to ensure consistent column order and fill missing values
    df_experiments = df_experiments.reindex(columns=final_columns)
    fill_values = {
        "Param1_Name": "",
        "Param1_Value": pd.NA,  # Use pd.NA for missing numeric/object
        "Param2_Name": "",
        "Param2_Value": pd.NA,
        "Actual_Passes": -1,  # Use -1 for missing passes
        "Fitness": float("inf"),  # Use inf for missing fitness
        "Comp_Time": -1.0,
        "Solution": "",
    }
    # Fill NaN for specific columns that might be missing in some experiments
    for col, val in fill_values.items():
        if col in df_experiments.columns:
            df_experiments[col].fillna(val, inplace=True)
        else:
            # Add column if missing entirely and fill with default
            df_experiments[col] = val

    # Attempt to convert columns to appropriate types before saving
    print("Converting column types...")
    df_experiments["Run"] = pd.to_numeric(
        df_experiments["Run"], errors="coerce"
    ).astype("Int64")
    df_experiments["Fitness"] = pd.to_numeric(
        df_experiments["Fitness"], errors="coerce"
    )
    df_experiments["Comp_Time"] = pd.to_numeric(
        df_experiments["Comp_Time"], errors="coerce"
    )
    df_experiments["Actual_Passes"] = pd.to_numeric(
        df_experiments["Actual_Passes"], errors="coerce"
    ).astype("Int64")
    # Try converting Param values, but ignore errors as they can be strings or numbers
    df_experiments["Param1_Value"] = pd.to_numeric(
        df_experiments["Param1_Value"], errors="ignore"
    )
    df_experiments["Param2_Value"] = pd.to_numeric(
        df_experiments["Param2_Value"], errors="ignore"
    )

    # Define output path relative to script directory
    output_csv_path = os.path.join(
        script_dir, "experiment_results_combined_gridsearch_runtimefix.csv"
    )
    try:
        df_experiments.to_csv(output_csv_path, index=False, float_format="%.6f")
        print(f"\nAll combined experiment results saved in '{output_csv_path}'.")
    except Exception as e:
        print(f"\nError saving results to CSV: {e}")

    print("\n--- Experiment Script Finished ---")
