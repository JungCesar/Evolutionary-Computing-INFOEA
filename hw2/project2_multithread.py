import random
import math
import time
import os
import numpy as np
import pandas as pd
import warnings # Import warnings module
import concurrent.futures # Added for multithreading
import threading

# --- Global Constants ---
NUM_VERTICES = 500
FM_PASSES_LIMIT = 10000 # Cumulative pass limit for comparison (a)
POPULATION_SIZE = 50
EXPERIMENT_RUNS = 10 # Runs for comparison (a)
RUNTIME_RUNS = 25     # Runs for comparison (b)

# Determine the number of threads to use (based on logical cores)
# You have 12 threads, so this should be 12.
# You can manually set this, e.g., NUM_WORKERS = 12
NUM_WORKERS = os.cpu_count() if os.cpu_count() else 4 # Default to 4 if cpu_count fails
print(f"Using {NUM_WORKERS} worker threads for parallel execution.")


# --- Node Class ---
class Node:
    """ Stores info for a vertex during FM pass """
    # Use __slots__ for memory efficiency, especially with many nodes
    __slots__ = ("gain", "neighbors", "prev", "next", "partition", "moved")

    def __init__(self, gain=0, neighbors=None, prev=-1, next=-1, partition=0):
        self.gain = gain
        # Store neighbors as a set for efficient lookups
        self.neighbors = set(neighbors) if neighbors is not None else set()
        self.prev = prev # Index of previous node in the gain bucket list
        self.next = next # Index of next node in the gain bucket list
        self.partition = partition # 0 or 1
        self.moved = False # Flag indicating if the node has been moved in the current pass

# --- Helper Functions ---

def calculate_gain(node_idx, solution, graph):
    """
    Calculates the gain for moving node_idx to the opposite partition.
    Gain = (Number of neighbors in the *other* partition) - (Number of neighbors in the *same* partition)
    """
    gain = 0
    n = len(solution)
    # Basic checks for valid indices and graph structure
    if not (0 <= node_idx < n and node_idx < len(graph)): return 0
    # Ensure solution is valid list/array of numbers
    if not isinstance(solution, (list, np.ndarray)): return 0 # Or raise TypeError
    try:
        current_partition = solution[node_idx]
    except IndexError: return 0 # Index out of bounds

    # Ensure graph[node_idx] is iterable (e.g., a set or list)
    neighbors = graph[node_idx] if hasattr(graph[node_idx], '__iter__') else []
    for neighbor in neighbors:
        # Ensure neighbor index is valid
        if not (0 <= neighbor < n): continue # Skip invalid neighbors
        try:
            if solution[neighbor] == current_partition:
                gain -= 1 # Internal edge cost decreases if moved
            else:
                gain += 1 # External edge cost increases if moved
        except IndexError: continue # Skip if neighbor index is out of bounds for solution
    return gain

def insert_node(nodes, gain_lists, max_gain, node_index, max_degree_adj):
    """
    Inserts node_index into the correct gain bucket's doubly linked list.
    Updates the max_gain for that partition if necessary.
    """
    if not (0 <= node_index < len(nodes)): return # Basic bounds check
    node = nodes[node_index]
    gain = node.gain
    part = node.partition
    # Map gain to the index in the gain_lists array
    # gain_lists[partition][gain + max_degree_adj] stores the head of the list for that gain
    gain_index = gain + max_degree_adj

    # Check if indices are valid before accessing lists
    if not (0 <= part < len(gain_lists) and 0 <= gain_index < len(gain_lists[part])): return

    # Insert at the head of the linked list for this gain value
    node.next = gain_lists[part][gain_index] # Current head becomes the 'next' of the new node
    node.prev = -1 # New node is the head, so no previous node
    head_node_idx = gain_lists[part][gain_index] # Get the old head index

    # If the list was not empty, update the 'prev' of the old head
    if head_node_idx != -1:
        # Check bounds for the old head node index
         if 0 <= head_node_idx < len(nodes):
             nodes[head_node_idx].prev = node_index

    # Set the new head of the list to the inserted node
    gain_lists[part][gain_index] = node_index

    # Update the maximum gain tracker for this partition if the inserted node's gain is higher
    if gain > max_gain[part]:
        max_gain[part] = gain

def remove_node(nodes, gain_lists, node_index, max_degree_adj):
    """ Removes node_index from its gain bucket's doubly linked list. """
    if not (0 <= node_index < len(nodes)): return # Basic bounds check
    node = nodes[node_index]
    gain = node.gain
    part = node.partition
    gain_index = gain + max_degree_adj

    # Check if indices are valid before accessing lists
    if not (0 <= part < len(gain_lists) and 0 <= gain_index < len(gain_lists[part])): return

    prev_node_idx = node.prev
    next_node_idx = node.next

    # Update the 'next' pointer of the previous node
    if prev_node_idx != -1:
        if 0 <= prev_node_idx < len(nodes): # Bounds check
            nodes[prev_node_idx].next = next_node_idx
    else:
        # If this was the head node, update the head pointer in gain_lists
        gain_lists[part][gain_index] = next_node_idx

    # Update the 'prev' pointer of the next node
    if next_node_idx != -1:
        if 0 <= next_node_idx < len(nodes): # Bounds check
            nodes[next_node_idx].prev = prev_node_idx

    # Reset the removed node's pointers
    node.prev = -1
    node.next = -1


def update_max_gain(gain_lists, max_gain, max_degree_adj):
    """
    Finds the new highest non-empty gain bucket for each partition
    after a node move might have emptied the previous max gain bucket.
    Scans down from the current max_gain value.
    """
    min_gain = -max_degree_adj # The lowest possible gain value
    for part in range(2): # For partition 0 and 1
        current_max = max_gain[part]
        # Clamp current_max to valid range [-max_degree_adj, max_degree_adj]
        current_max = min(current_max, max_degree_adj)
        current_max = max(current_max, min_gain -1) # Ensure it's at least min_gain - 1

        # Scan downwards from the current max gain
        while current_max >= min_gain:
            gain_index = current_max + max_degree_adj
            # Check bounds for gain_index
            if not (0 <= gain_index < len(gain_lists[part])):
                # This shouldn't happen if max_degree_adj is correct, but acts as a safeguard
                current_max = min_gain - 1 # Set to invalid gain to stop loop
                break
            # If we find a non-empty bucket, this is the new max gain
            if gain_lists[part][gain_index] != -1:
                max_gain[part] = current_max
                break
            current_max -= 1
        else:
            # If the loop completes without finding a non-empty bucket, set max gain to invalid
            max_gain[part] = min_gain - 1


def fitness_function(solution, graph):
    """
    Calculates the cut size (number of edges crossing the partition).
    Handles potential list/array type for solution.
    """
    n = len(solution)
    if n == 0: return 0
    total_cut_edges = 0
    # Ensure solution is indexable like a list (handles numpy arrays too)
    sol_list = list(solution) # Convert to list if necessary

    for i in range(n):
        # Check if node i exists in the graph and has neighbors defined
        if i < len(graph) and hasattr(graph[i], '__iter__') and i < len(sol_list):
            part_i = sol_list[i]
            for neighbor_idx in graph[i]:
                # Check if neighbor index is valid within the solution length
                if 0 <= neighbor_idx < n:
                    # Check if neighbor is in a different partition
                    if part_i != sol_list[neighbor_idx]:
                        total_cut_edges += 1
                # else: ignore neighbors outside the defined vertex range

    # Each edge is counted twice (once for each endpoint), so divide by 2
    return total_cut_edges // 2


def balance_child(child_list_input):
    """
    Adjusts a child solution (list of 0s and 1s) to have a balanced number
    of 0s and 1s (n/2 each, handling odd n).
    Modifies the list in-place by randomly flipping bits.
    Ensures the output is a list of integers.
    """
    n = len(child_list_input)
    if n == 0: return []
    target_ones = n // 2 # Integer division handles odd n correctly

    # Work on a copy, ensure integer type
    child_list = [int(bit) for bit in child_list_input]

    try:
        current_ones = child_list.count(1)
        # Optional: Check for non-binary values if conversion didn't raise error
        # if child_list.count(0) + current_ones != n:
        #     print("Warning: balance_child input contained non-binary values after int conversion.")
        #     # Decide how to handle: return as is, raise error, or attempt further fixing
        #     # return child_list # Return potentially flawed list
    except (TypeError, ValueError):
        # This might happen if input contained non-numeric types before conversion
        print("Error: Non-numeric type found during balance_child count.")
        return child_list_input # Return original on error

    diff = current_ones - target_ones # Positive if too many 1s, negative if too few
    indices = list(range(n))

    if diff > 0: # Too many ones, need to flip 'diff' ones to zeros
        ones_indices = [i for i in indices if child_list[i] == 1]
        if len(ones_indices) >= diff:
            indices_to_flip = random.sample(ones_indices, diff)
            for i in indices_to_flip:
                child_list[i] = 0
        else:
            # This indicates a logic error or inconsistent state
            print(f"Error: balance_child logic error - Cannot flip {diff} ones from available {len(ones_indices)}.")
            # Consider returning the partially balanced list or raising an error
            return child_list # Return the list as is
    elif diff < 0: # Too few ones (too many zeros), need to flip '-diff' zeros to ones
        num_to_flip = -diff
        zeros_indices = [i for i in indices if child_list[i] == 0]
        if len(zeros_indices) >= num_to_flip:
            indices_to_flip = random.sample(zeros_indices, num_to_flip)
            for i in indices_to_flip:
                child_list[i] = 1
        else:
            # Logic error or inconsistent state
            print(f"Error: balance_child logic error - Cannot flip {num_to_flip} zeros from available {len(zeros_indices)}.")
            return child_list # Return the list as is

    # Final check (optional but recommended)
    # if child_list.count(1) != target_ones:
    #    print("Warning: balance_child did not achieve perfect balance.")

    return child_list # Return the balanced list of ints


def read_graph_data(filename):
    """Reads the graph from a file and returns adjacency lists (list of sets)."""
    # Initialize graph as a list of empty sets for NUM_VERTICES
    graph = [set() for _ in range(NUM_VERTICES)]
    try:
        with open(filename, "r") as f:
            ln = 0 # Line number for error reporting
            for line in f:
                ln += 1
                line = line.strip()
                if not line or line.startswith('%'): continue # Skip empty lines or comments

                parts = line.split()
                if not parts: continue

                try:
                    # Assuming the first part is the vertex index (1-based)
                    vertex_index = int(parts[0]) - 1

                    # Check if vertex_index is within the expected range
                    if not (0 <= vertex_index < NUM_VERTICES):
                        # warnings.warn(f"Vertex index {vertex_index + 1} on line {ln} is out of range [1, {NUM_VERTICES}]. Skipping neighbors.")
                        continue

                    # Find where the neighbors list starts
                    # This logic seems specific to a certain format, might need adjustment
                    # It tries to find the number of neighbors if present
                    num_neighbors_idx = -1
                    # Heuristic 1: Look for a digit after a ')' - common in some formats
                    for idx, part in enumerate(parts):
                        if idx > 0 and ')' in parts[idx-1] and part.isdigit():
                             num_neighbors_idx = idx
                             break
                    # Heuristic 2: If not found, look for the first digit after the vertex index
                    if num_neighbors_idx == -1:
                         for idx, part in enumerate(parts):
                             if idx > 0 and part.isdigit():
                                 num_neighbors_idx = idx
                                 break

                    # If neither heuristic worked, assume all remaining parts are neighbors
                    if num_neighbors_idx == -1:
                        start_neighbors_idx = 1
                    # If a number was found, neighbors start after it
                    elif num_neighbors_idx + 1 < len(parts):
                         start_neighbors_idx = num_neighbors_idx + 1
                    else:
                        # Number found but no neighbors listed after it
                        continue # Skip this line

                    # Extract and convert neighbors (1-based to 0-based)
                    connected_vertices = []
                    for neighbor_str in parts[start_neighbors_idx:]:
                        try:
                            neighbor_idx = int(neighbor_str) - 1
                            # Add only valid neighbors within the graph size
                            if 0 <= neighbor_idx < NUM_VERTICES:
                                connected_vertices.append(neighbor_idx)
                            # else: warnings.warn(f"Neighbor index {neighbor_idx + 1} for vertex {vertex_index + 1} on line {ln} is out of range.")
                        except ValueError:
                            # warnings.warn(f"Invalid neighbor format '{neighbor_str}' for vertex {vertex_index + 1} on line {ln}.")
                            pass # Ignore non-integer parts

                    # Add valid neighbors to the set for the current vertex
                    graph[vertex_index].update(connected_vertices)

                except (ValueError, IndexError) as e:
                    print(f"Skipping line {ln} due to parsing error: {e} - Line content: '{line}'")
                    continue

        print("Ensuring graph symmetry (adding missing edges)...")
        # Ensure symmetry: if A -> B exists, make sure B -> A also exists
        for i in range(NUM_VERTICES):
            # Iterate over a copy of the set to allow modification during iteration
            for neighbor in list(graph[i]):
                if 0 <= neighbor < NUM_VERTICES:
                    # If the reverse edge doesn't exist, add it
                    if i not in graph[neighbor]:
                        graph[neighbor].add(i)
                else:
                    # If a neighbor index is somehow invalid, remove it
                    graph[i].discard(neighbor)
        print("Graph symmetry ensured.")

    except FileNotFoundError:
        print(f"Error: Graph file not found at {filename}")
        raise # Re-raise the exception to stop the script
    except Exception as e:
        print(f"An unexpected error occurred while reading the graph: {e}")
        raise # Re-raise

    return graph


def generate_random_solution():
    """Generates a balanced random initial solution (list of 0s and 1s)."""
    half = NUM_VERTICES // 2
    # Create a list with exactly half 0s and half 1s (approx for odd NUM_VERTICES)
    solution = [0] * half + [1] * (NUM_VERTICES - half)
    # Shuffle the list randomly
    random.shuffle(solution)
    return solution


def mutate(solution, mutation_size):
    """
    Perturbs the solution by swapping 'mutation_size' pairs of 0s and 1s.
    This preserves the balance of the partition.
    """
    mutated = solution.copy() # Work on a copy
    n = len(mutated)
    if n == 0: return []

    # Find indices of nodes currently in partition 0 and 1
    zeros_indices = [i for i, bit in enumerate(mutated) if bit == 0]
    ones_indices = [i for i, bit in enumerate(mutated) if bit == 1]

    # Determine the actual number of swaps possible (limited by the smaller partition size)
    num_swaps = min(mutation_size, len(zeros_indices), len(ones_indices))

    if num_swaps > 0:
        # Randomly select 'num_swaps' indices from each partition
        zeros_to_swap = random.sample(zeros_indices, num_swaps)
        ones_to_swap = random.sample(ones_indices, num_swaps)

        # Perform the swaps
        for i in range(num_swaps):
            zero_idx = zeros_to_swap[i]
            one_idx = ones_to_swap[i]
            mutated[zero_idx] = 1
            mutated[one_idx] = 0

    return mutated


def get_hamming_distance(parent1, parent2):
    """Compute the Hamming distance between two solutions (lists/arrays)."""
    dist = 0
    # Ensure lengths match, otherwise the concept is ill-defined
    if len(parent1) != len(parent2):
        # warnings.warn("Hamming distance called on solutions of different lengths.")
        return max(len(parent1), len(parent2)) # Or return an error indicator like -1

    for i in range(len(parent1)):
        if parent1[i] != parent2[i]:
            dist += 1
    return dist


def crossover(parent1, parent2):
    """
    Performs uniform crossover, adapted to maintain partition balance.
    Uses the Hamming distance heuristic from the practical description.
    Returns a new child solution (list of ints).
    """
    n = len(parent1)
    child = [0] * n # Initialize child

    # Basic validation
    if len(parent2) != n:
        # warnings.warn("Crossover called with parents of different lengths. Generating random.")
        return generate_random_solution() # Fallback

    p1_eff = parent1 # Effective parent 1
    # Calculate Hamming distance
    hd = get_hamming_distance(parent1, parent2)

    # If distance is large, invert one parent (heuristic to increase agreement)
    if hd > n / 2:
        p1_eff = [1 - bit for bit in parent1] # Invert bits

    disagree_indices = [] # Indices where p1_eff and parent2 differ
    ones_needed = n // 2 # Target number of 1s for balance
    zeros_needed = n - ones_needed # Target number of 0s

    ones_count = 0 # Count of 1s inherited from agreeing positions
    zeros_count = 0 # Count of 0s inherited from agreeing positions

    # First pass: Inherit agreeing bits
    for i in range(n):
        if p1_eff[i] == parent2[i]:
            # Ensure the inherited bit is 0 or 1
            bit = int(p1_eff[i]) # Convert just in case
            if bit not in [0, 1]:
                # Handle unexpected values, e.g., default to 0 or raise error
                # warnings.warn(f"Non-binary value {bit} encountered during crossover agreement.")
                bit = 0 # Default to 0
            child[i] = bit
            if child[i] == 1: ones_count += 1
            else: zeros_count += 1
        else:
            disagree_indices.append(i) # Store index for later assignment

    # Second pass: Fill disagreeing positions to achieve balance
    ones_to_add = ones_needed - ones_count
    zeros_to_add = zeros_needed - zeros_count

    # Check if the required number of flips matches the number of disagreeing positions
    if ones_to_add < 0 or zeros_to_add < 0 or (ones_to_add + zeros_to_add != len(disagree_indices)):
        # This indicates an issue, likely imbalance in parents or the HD heuristic interaction
        # Fallback to balancing the partially created child
        # print(f"Warning: Crossover balance issue (Needed {ones_to_add} ones, {zeros_to_add} zeros for {len(disagree_indices)} slots). Rebalancing child.")
        return balance_child(child) # Attempt to fix balance

    else:
        # Shuffle the disagreeing indices to randomly assign remaining 0s and 1s
        random.shuffle(disagree_indices)
        # Assign the required number of 1s
        for i in range(ones_to_add):
            child[disagree_indices[i]] = 1
        # Assign the required number of 0s (the rest)
        for i in range(ones_to_add, len(disagree_indices)):
            child[disagree_indices[i]] = 0

    # Final check (optional)
    # if child.count(1) != ones_needed:
    #    print("Error: Crossover resulted in unbalanced child even after assignment.")
    #    return balance_child(child) # Try balancing again

    return child # Returns list of ints


# --- FM HEURISTIC ---
def fm_heuristic(initial_solution, graph, max_passes=10):
    """
    Fiduccia-Mattheyses heuristic for graph partitioning.
    Uses gain buckets (linked lists) for efficient selection of nodes to move.
    Aims to improve the cut size while maintaining partition balance.

    Args:
        initial_solution (list): The starting partition (list of 0s and 1s).
        graph (list): Adjacency list/set representation of the graph.
        max_passes (int): Maximum number of outer passes (iterations) allowed.
                          Each pass attempts to move all nodes once.

    Returns:
        tuple: (best_balanced_solution_found, passes_performed_this_call)
               Returns the best *balanced* solution encountered during the passes.
               If no balanced solution is found or improved, returns the initial balanced one.
    """
    n = len(initial_solution)
    if n == 0: return initial_solution.copy(), 0
    target_part_size = n // 2 # Target size for partition 0 (and 1, approx)

    # --- Initialization ---
    # Ensure the starting solution is balanced
    working_solution = initial_solution.copy()
    if working_solution.count(0) != target_part_size:
        # print("Warning: Initial solution for FM is unbalanced. Balancing...")
        working_solution = balance_child(working_solution)
        if working_solution.count(0) != target_part_size:
            print("CRITICAL ERROR: Failed to balance initial solution for FM.")
            return initial_solution.copy(), 0 # Return original if balancing failed

    # Determine max degree for gain bucket sizing
    max_degree = 0
    for i in range(n):
        if i < len(graph) and hasattr(graph[i], '__iter__'):
            max_degree = max(max_degree, len(graph[i]))
    # Max possible gain is +max_degree, min is -max_degree. Add 1 for the range.
    # Use max(1, max_degree) to handle disconnected graphs or single nodes.
    max_degree_adj = max(1, max_degree)
    gain_list_size = 2 * max_degree_adj + 1 # Size needed for gains from -max_degree to +max_degree
    min_gain = -max_degree_adj

    # Initialize Node objects for all vertices
    nodes = [Node() for _ in range(n)]

    # Track the best balanced solution found across all passes
    best_solution_overall = working_solution.copy()
    best_overall_cut_size = fitness_function(best_solution_overall, graph)

    improved_in_cycle = True # Flag to control the outer loop
    passes_done = 0

    # --- Outer FM Loop (Passes) ---
    while improved_in_cycle and passes_done < max_passes:
        improved_in_cycle = False # Reset for this pass
        passes_done += 1
        # Store the solution at the start of the pass for potential rollback
        solution_at_pass_start = working_solution.copy()
        current_cut_at_pass_start = fitness_function(solution_at_pass_start, graph)

        # --- Initialize for the Pass ---
        # Gain buckets: gain_lists[partition][gain + max_degree_adj] = head_node_index
        gain_lists = [[-1] * gain_list_size for _ in range(2)] # -1 indicates empty list
        # Max gain trackers for each partition's bucket list
        max_gain = [min_gain - 1] * 2 # Initialize below the minimum possible gain

        # Calculate initial gains and populate gain buckets
        indices = list(range(n))
        # random.shuffle(indices) # Shuffling order can sometimes help escape local optima
        for i in indices:
            nodes[i].moved = False # Reset moved status for the new pass
            nodes[i].partition = working_solution[i]
            # Ensure neighbors are correctly assigned (handle potential graph inconsistencies)
            nodes[i].neighbors = graph[i] if i < len(graph) and graph[i] else set()
            nodes[i].gain = calculate_gain(i, working_solution, graph)
            # Insert the node into the appropriate gain bucket if gain is within valid range
            if min_gain <= nodes[i].gain <= max_degree_adj:
                insert_node(nodes, gain_lists, max_gain, i, max_degree_adj)
            # else:
                # This might happen if max_degree calculation was off or graph changed
                # warnings.warn(f"Node {i} calculated gain {nodes[i].gain} outside expected range [{-max_degree_adj}, {max_degree_adj}].")


        moves_sequence = [] # Stores the sequence of nodes moved in this pass
        cumulative_gains = [0.0] # Stores cumulative gain after each move (starts with 0)
        # Track the solution state *during* the inner loop moves
        current_solution_in_pass = working_solution.copy()

        # --- Inner FM Loop (Moves within a pass) ---
        for k in range(n): # Try to move n nodes
            best_node_to_move = -1
            selected_gain = -float('inf')
            move_from_part = -1

            # --- Find Best Valid Node to Move ---
            # Check partition sizes based on the *current* state within the pass
            current_part0_count_k = current_solution_in_pass.count(0)
            current_part1_count_k = n - current_part0_count_k

            # Determine which partition to potentially move *from* to maintain balance
            # Prioritize moving from the larger partition if imbalance occurs
            # Or, if balanced, consider moving the node with highest gain from either side.
            allowed_to_move_from = []
            # Check if moving from partition 0 is allowed (doesn't make it too small)
            # The target size is target_part_size. Allow moving if current size > target_part_size - tolerance
            # Simple balance: Allow move only if size > 1 (or target_part_size/2 for strictness)
            # Relaxed balance constraint during pass (Kernighan-Lin style):
            # Allow move if the partition size is > 0 (or some minimum threshold)
            # Let's use a simple check: allow move if partition has > 1 node (prevents emptying)
            # OR use the balance constraint from the paper: allow if size > min_size (n/2 - tolerance)
            # Let's stick to the idea of moving from the side with the highest gain node,
            # but only if the move doesn't violate balance *too much*.
            # The original FM/KL allows temporary imbalance during the pass.

            potential_candidates = [] # Store (gain, node_idx, from_part)

            # Find best candidate from partition 0
            if current_part0_count_k > 0: # Can move from partition 0
                gain0 = max_gain[0]
                if gain0 > min_gain -1: # Check if there are any nodes in buckets for part 0
                    gain0_idx = gain0 + max_degree_adj
                    if 0 <= gain0_idx < gain_list_size:
                        node_idx_in_bucket = gain_lists[0][gain0_idx]
                        # Find the first *unmoved* node in the highest gain bucket
                        while node_idx_in_bucket != -1:
                            if 0 <= node_idx_in_bucket < len(nodes) and not nodes[node_idx_in_bucket].moved:
                                potential_candidates.append((gain0, node_idx_in_bucket, 0))
                                break
                            # Check bounds before accessing next node
                            if 0 <= node_idx_in_bucket < len(nodes):
                                node_idx_in_bucket = nodes[node_idx_in_bucket].next
                            else: break # Invalid index

            # Find best candidate from partition 1
            if current_part1_count_k > 0: # Can move from partition 1
                gain1 = max_gain[1]
                if gain1 > min_gain -1: # Check if there are any nodes in buckets for part 1
                    gain1_idx = gain1 + max_degree_adj
                    if 0 <= gain1_idx < gain_list_size:
                        node_idx_in_bucket = gain_lists[1][gain1_idx]
                        # Find the first *unmoved* node in the highest gain bucket
                        while node_idx_in_bucket != -1:
                             if 0 <= node_idx_in_bucket < len(nodes) and not nodes[node_idx_in_bucket].moved:
                                potential_candidates.append((gain1, node_idx_in_bucket, 1))
                                break
                             # Check bounds before accessing next node
                             if 0 <= node_idx_in_bucket < len(nodes):
                                node_idx_in_bucket = nodes[node_idx_in_bucket].next
                             else: break # Invalid index

            # Select the best overall candidate based on gain
            if potential_candidates:
                 potential_candidates.sort(key=lambda x: x[0], reverse=True) # Sort by gain descending
                 selected_gain, best_node_to_move, move_from_part = potential_candidates[0]
            else:
                 # No movable nodes found (all nodes moved or buckets empty)
                 break # Exit the inner loop (moves for this pass)


            # --- Perform the Move ---
            if best_node_to_move == -1:
                # Should have been caught by the 'break' above, but as a safeguard
                break # Exit inner loop if no node can be moved

            node_to_move_idx = best_node_to_move
            original_partition = nodes[node_to_move_idx].partition # Should be == move_from_part

            # 1. Lock the node (mark as moved) and remove from gain lists
            nodes[node_to_move_idx].moved = True
            remove_node(nodes, gain_lists, node_to_move_idx, max_degree_adj)

            # 2. Update the current solution state for this pass
            current_solution_in_pass[node_to_move_idx] = 1 - original_partition
            nodes[node_to_move_idx].partition = current_solution_in_pass[node_to_move_idx] # Update node's partition info

            # 3. Record the move and cumulative gain
            moves_sequence.append(node_to_move_idx)
            cumulative_gains.append(cumulative_gains[-1] + selected_gain)

            # 4. Update gains of neighbors
            for neighbor_idx in nodes[node_to_move_idx].neighbors:
                if 0 <= neighbor_idx < n and not nodes[neighbor_idx].moved:
                    # Calculate gain delta: +2 if neighbor was in the same partition, -2 if different
                    gain_delta = 2 if current_solution_in_pass[neighbor_idx] == original_partition else -2

                    # Remove neighbor from its current gain bucket
                    neighbor_gain_before = nodes[neighbor_idx].gain
                    neighbor_gain_idx_before = neighbor_gain_before + max_degree_adj
                    if 0 <= neighbor_gain_idx_before < gain_list_size:
                         remove_node(nodes, gain_lists, neighbor_idx, max_degree_adj)

                    # Update neighbor's gain
                    nodes[neighbor_idx].gain += gain_delta

                    # Insert neighbor into its new gain bucket
                    neighbor_gain_idx_after = nodes[neighbor_idx].gain + max_degree_adj
                    if 0 <= neighbor_gain_idx_after < gain_list_size:
                         insert_node(nodes, gain_lists, max_gain, neighbor_idx, max_degree_adj)
                    # else:
                         # warnings.warn(f"Neighbor {neighbor_idx} updated gain {nodes[neighbor_idx].gain} outside range during move.")


            # 5. Update max_gain trackers after potential changes
            update_max_gain(gain_lists, max_gain, max_degree_adj)
        # --- End of Inner FM Loop (Moves) ---

        if not moves_sequence:
            # No moves were made in this pass, unlikely to improve further
            # print("   FM Pass ended: No moves made.")
            break # Exit the outer loop

        # --- Find Best Prefix of Moves ---
        # Find the step 'k' in the sequence that yielded the maximum cumulative gain
        best_k = np.argmax(cumulative_gains) # Index of max cumulative gain
        max_cumulative_gain = cumulative_gains[best_k]
        best_num_moves = best_k # The number of moves corresponding to the best gain

        # --- Rollback/Apply Best Move Sequence ---
        # Start from the solution *before* this pass began
        solution_after_rollback = solution_at_pass_start.copy()
        # Apply only the first 'best_num_moves' from the sequence
        for i in range(best_num_moves):
            # Add bounds check for moves_sequence access
            if i < len(moves_sequence):
                node_idx = moves_sequence[i]
                if 0 <= node_idx < n: # Check node index validity
                    # Flip the partition for the node in the best prefix
                    solution_after_rollback[node_idx] = 1 - solution_after_rollback[node_idx]
                else:
                    # This indicates a serious error in the moves_sequence generation
                    print(f"CRITICAL ERROR: Invalid node index {node_idx} in moves list during rollback.")
                    # Revert fully to the state before the pass on error
                    solution_after_rollback = solution_at_pass_start.copy()
                    best_num_moves = 0 # Mark that no valid improvement was found this pass
                    break
            else:
                # This indicates an error if best_num_moves > len(moves_sequence)
                print(f"CRITICAL ERROR: Index {i} out of bounds for moves_sequence during rollback.")
                solution_after_rollback = solution_at_pass_start.copy()
                best_num_moves = 0
                break

        # The final solution for this pass is the one after applying the best prefix
        final_solution_this_pass = solution_after_rollback
        current_cut_size = fitness_function(final_solution_this_pass, graph)

        # --- Check for Improvement and Balance ---
        # IMPORTANT: Only accept the result of the pass if it's balanced
        if final_solution_this_pass.count(0) != target_part_size:
             # print(f"   FM Pass {passes_done} resulted in imbalance ({final_solution_this_pass.count(0)} zeros). Discarding pass.")
             # Keep working_solution as it was at the start of this pass (solution_at_pass_start)
             working_solution = solution_at_pass_start.copy()
             # Don't set improved_in_cycle = True, as the improvement wasn't valid
        else:
             # Accept the balanced state resulting from the best prefix
             working_solution = final_solution_this_pass.copy()
             # Check if this valid pass improved the *overall* best cut size found so far
             if current_cut_size < best_overall_cut_size:
                 # print(f"   FM Pass {passes_done}: Improved overall best cut from {best_overall_cut_size} to {current_cut_size} (Gain: {max_cumulative_gain:.1f} at k={best_k}).")
                 best_overall_cut_size = current_cut_size
                 best_solution_overall = final_solution_this_pass.copy()
                 improved_in_cycle = True # Mark that improvement occurred in this cycle
             # elif max_cumulative_gain > 0:
                 # print(f"   FM Pass {passes_done}: Positive gain ({max_cumulative_gain:.1f} at k={best_k}), but cut size {current_cut_size} not better than overall best {best_overall_cut_size}.")
                 # Pass accepted, but didn't improve global best. Continue if allowed.
             # else:
                 # print(f"   FM Pass {passes_done}: No improvement found (Max Gain: {max_cumulative_gain:.1f} at k={best_k}). Cut size {current_cut_size}.")
                 # Pass accepted (it's balanced), but no improvement. Loop might stop.
                 pass

    # --- End of Outer FM Loop ---

    # Final check: Ensure the returned solution is balanced
    if best_solution_overall.count(0) != target_part_size:
        print(f"CRITICAL ERROR: fm_heuristic final best solution is not balanced! Reverting to initial balanced.")
        # Attempt to re-balance the initial solution as a last resort
        initial_balanced = initial_solution.copy()
        if initial_balanced.count(0) != target_part_size:
            initial_balanced = balance_child(initial_balanced)
        # If still not balanced, something is fundamentally wrong
        if initial_balanced.count(0) != target_part_size:
             print("FATAL: Could not produce a balanced solution in FM.")
             # Return a newly generated random balanced solution
             return generate_random_solution(), passes_done
        return initial_balanced, passes_done


    # print(f"FM finished after {passes_done} passes. Best cut size: {best_overall_cut_size}")
    return best_solution_overall, passes_done


# --- Revised Metaheuristics (Handling Pass Limits and Time Limits) ---
# These functions now accept max_total_passes OR time_limit.
# They pass down the remaining allowance to fm_heuristic.

def MLS(graph, num_starts, max_total_passes=None, time_limit=None):
    """ Multi-Start Local Search using FM Heuristic. """
    best_solution = None
    best_fit = float("inf")
    start_time = time.perf_counter()
    total_passes_consumed = 0
    # Determine if we must run until pass limit is hit (used for timing)
    force_run_to_pass_limit = (max_total_passes is not None and time_limit is None)
    # If forcing pass limit, ignore num_starts; otherwise, use it.
    effective_num_starts = float('inf') if force_run_to_pass_limit else num_starts

    i = 0
    while i < effective_num_starts:
        current_time = time.perf_counter()
        # Check stopping conditions
        if time_limit is not None and current_time - start_time >= time_limit:
            # print(f"MLS: Time limit ({time_limit:.2f}s) reached.")
            break
        if max_total_passes is not None and total_passes_consumed >= max_total_passes:
            # print(f"MLS: Pass limit ({max_total_passes}) reached.")
            break

        # Generate initial solution
        sol = generate_random_solution()

        # Calculate remaining passes allowed for this FM call
        remaining_passes = float('inf') # Default if no pass limit
        if max_total_passes is not None:
            remaining_passes = max(0, max_total_passes - total_passes_consumed)
            if remaining_passes == 0: break # No more passes allowed
            remaining_passes = int(remaining_passes) # FM expects int

        # Run FM
        try:
            local_opt, passes_this_call = fm_heuristic(sol, graph, max_passes=remaining_passes)
            total_passes_consumed += passes_this_call
            if local_opt is not None:
                fit = fitness_function(local_opt, graph)
                if fit < best_fit:
                    best_fit = fit
                    best_solution = local_opt.copy() # Store a copy
        except Exception as e:
            print(f"Error in MLS fm_heuristic call: {e}")
            # Decide whether to continue or break on error
            continue # Continue to next start

        finally:
             # Increment start counter regardless of success/failure, unless forcing pass limit
             if not force_run_to_pass_limit:
                 i += 1
             elif max_total_passes is None: # If forcing pass limit but none set, increment anyway
                 i += 1


    # Return None if no solution was ever found
    if best_solution is None:
        # print("MLS finished without finding any valid solution.")
        best_solution = generate_random_solution() # Return random balanced as fallback

    return best_solution, total_passes_consumed


def ILS(graph, initial_solution, mutation_size, max_total_passes=None, time_limit=None):
    """ Iterated Local Search using FM Heuristic. """
    # Ensure initial solution is balanced
    if initial_solution.count(0) != len(initial_solution) // 2:
        best_solution = balance_child(initial_solution.copy())
    else:
        best_solution = initial_solution.copy()

    best_fit = fitness_function(best_solution, graph)
    start_time = time.perf_counter()
    total_passes_consumed = 0
    no_improvement = 0
    max_no_improvement = 10 # Stopping criterion if not time/pass limited
    unchanged_count = 0 # Tracks consecutive times the fitness didn't improve
    iteration = 0

    # Determine if we must run until pass limit is hit
    force_run_to_pass_limit = (max_total_passes is not None and time_limit is None)

    while True:
        iteration += 1
        current_time = time.perf_counter()

        # Check stopping conditions
        if time_limit is not None and current_time - start_time >= time_limit:
            # print(f"ILS: Time limit ({time_limit:.2f}s) reached.")
            break
        if max_total_passes is not None and total_passes_consumed >= max_total_passes:
            # print(f"ILS: Pass limit ({max_total_passes}) reached.")
            break
        # Only use no_improvement stop if *not* forcing pass/time limit
        if not force_run_to_pass_limit and no_improvement >= max_no_improvement:
            # print(f"ILS: Stopping after {max_no_improvement} iterations without improvement.")
            break

        # Mutate the current best solution
        mutated = mutate(best_solution, mutation_size)
        # Ensure mutation result is balanced (mutate should preserve, but double-check)
        if mutated.count(0) != len(mutated) // 2:
            mutated = balance_child(mutated)

        # Calculate remaining passes allowed for this FM call
        remaining_passes = float('inf')
        if max_total_passes is not None:
            remaining_passes = max(0, max_total_passes - total_passes_consumed)
            if remaining_passes == 0: break
            remaining_passes = int(remaining_passes)

        # Run FM on the mutated solution
        try:
            local_opt, passes_this_call = fm_heuristic(mutated, graph, max_passes=remaining_passes)
            total_passes_consumed += passes_this_call

            if local_opt is not None:
                fit = fitness_function(local_opt, graph)

                # Acceptance Criterion (only accept better solutions)
                if fit == best_fit:
                    unchanged_count += 1 # Fitness stayed the same
                    no_improvement += 1
                elif fit < best_fit:
                    best_solution = local_opt.copy() # Found a new best
                    best_fit = fit
                    no_improvement = 0 # Reset counter
                    unchanged_count = 0
                else:
                    no_improvement += 1 # Worse solution, increment counter
                    # unchanged_count doesn't reset here
            else:
                # FM failed or returned None
                no_improvement += 1

        except Exception as e:
            print(f"Error in ILS fm_heuristic call: {e}")
            no_improvement += 1 # Count as no improvement if FM fails
            continue # Continue to next iteration

    return best_solution, unchanged_count, total_passes_consumed


def ILS_annealing(graph, initial_solution, mutation_size, max_total_passes=None, time_limit=None):
    """ ILS with Simulated Annealing acceptance criterion. """
    # Ensure initial solution is balanced
    if initial_solution.count(0) != len(initial_solution) // 2:
        current_solution = balance_child(initial_solution.copy())
    else:
        current_solution = initial_solution.copy()

    best_solution = current_solution.copy()
    current_fit = fitness_function(current_solution, graph)
    best_fit = current_fit

    start_time = time.perf_counter()
    total_passes_consumed = 0
    no_improvement_on_best = 0 # Tracks iterations since *best* was improved
    max_no_improvement = 20 # Stop if best hasn't improved for this many iterations
    unchanged_count = 0 # Tracks consecutive times the *current* fitness didn't change

    # Annealing parameters
    temperature = 10.0 # Initial temperature (adjust based on typical fitness delta)
    cooling_rate = 0.98 # Multiplicative cooling rate
    min_temperature = 0.1 # Stop if temperature gets too low

    iteration = 0
    force_run_to_pass_limit = (max_total_passes is not None and time_limit is None)

    while True:
        iteration += 1
        current_time = time.perf_counter()

        # Check stopping conditions
        if time_limit is not None and current_time - start_time >= time_limit: break
        if max_total_passes is not None and total_passes_consumed >= max_total_passes: break
        if not force_run_to_pass_limit:
            if no_improvement_on_best >= max_no_improvement: break
            if temperature <= min_temperature: break

        # Mutate the *current* solution
        mutated = mutate(current_solution, mutation_size)
        if mutated.count(0) != len(mutated) // 2:
            mutated = balance_child(mutated)

        # Calculate remaining passes
        remaining_passes = float('inf')
        if max_total_passes is not None:
            remaining_passes = max(0, max_total_passes - total_passes_consumed)
            if remaining_passes == 0: break
            remaining_passes = int(remaining_passes)

        # Run FM
        try:
            local_opt, passes_this_call = fm_heuristic(mutated, graph, max_passes=remaining_passes)
            total_passes_consumed += passes_this_call

            if local_opt is not None:
                fit = fitness_function(local_opt, graph)

                if fit == current_fit: unchanged_count += 1
                else: unchanged_count = 0 # Reset if fitness changed

                # Simulated Annealing Acceptance Criterion
                delta_e = fit - current_fit # Change in fitness (energy)
                accepted_move = False

                if delta_e < 0: # Always accept improving moves
                    accepted_move = True
                elif temperature > 1e-9: # Avoid division by zero or near-zero T
                    acceptance_prob = math.exp(-delta_e / temperature)
                    if random.random() < acceptance_prob:
                        accepted_move = True # Accept worsening move with probability

                if accepted_move:
                    current_solution = local_opt.copy()
                    current_fit = fit
                    # Update best solution if current is better
                    if fit < best_fit:
                        best_solution = local_opt.copy()
                        best_fit = fit
                        no_improvement_on_best = 0 # Reset best improvement counter
                    else:
                        no_improvement_on_best += 1 # Best didn't improve
                else:
                    # Move was rejected
                    no_improvement_on_best += 1

            else: # FM returned None
                no_improvement_on_best += 1

        except Exception as e:
            print(f"Error in ILS_A fm_heuristic call: {e}")
            no_improvement_on_best += 1
            continue

        # Cool down temperature (only if not forcing pass/time limit)
        if not force_run_to_pass_limit:
            temperature *= cooling_rate

    return best_solution, unchanged_count, total_passes_consumed


def ILS_adaptive(graph, initial_solution, max_total_passes=None, time_limit=None):
    """ ILS with Adaptive Mutation Strength. """
    # Parameters for adaptive mutation
    initial_mutation_mean = 15.0 # Starting average mutation size
    mutation_std_dev = 2.0    # Starting standard deviation for mutation size
    decay_factor = 0.98       # Factor to reduce mutation strength on improvement
    increase_factor = 1.1     # Factor to increase mutation strength on stagnation
    stagnation_threshold = 50 # Iterations without improvement to trigger increase
    max_iterations = 100000   # Default limit if no other limits active
    min_mutation_size = 1
    max_mutation_size = NUM_VERTICES // 4 # Avoid excessive mutation
    stagnation_window = 15    # Window size for checking fitness variance during stagnation
    stagnation_tolerance = 0.0001 # Minimum variance to consider not stuck
    # restart_reset = True # Option to reset mutation strength on restart (not implemented here)

    # Ensure initial solution is balanced
    if initial_solution.count(0) != len(initial_solution) // 2:
        current_solution = balance_child(initial_solution.copy())
    else:
        current_solution = initial_solution.copy()

    best_solution = current_solution.copy()
    current_fit = fitness_function(current_solution, graph)
    best_fit = current_fit

    current_mutation_mean = float(initial_mutation_mean)
    current_mutation_std_dev = float(mutation_std_dev)
    mutation_history = [] # Track mutation parameters over time (optional)
    stagnation_count = 0
    iteration = 0
    fitness_history = [current_fit] # Track recent fitness values for stagnation check

    start_time = time.perf_counter()
    total_passes_consumed = 0
    force_run_to_pass_limit = (max_total_passes is not None and time_limit is None)
    # Use max_iterations only if not forcing pass/time limit
    effective_max_iterations = float('inf') if force_run_to_pass_limit else max_iterations

    while iteration < effective_max_iterations:
        iteration += 1
        current_time = time.perf_counter()

        # Check stopping conditions
        if time_limit is not None and current_time - start_time >= time_limit: break
        if max_total_passes is not None and total_passes_consumed >= max_total_passes: break
        # No stagnation stop needed if forcing pass/time limit

        # --- Adapt Mutation Size ---
        # Sample mutation size from a normal distribution
        mutation_size = int(np.random.normal(current_mutation_mean, current_mutation_std_dev))
        # Clamp mutation size to valid range
        mutation_size = max(min_mutation_size, min(mutation_size, max_mutation_size))

        # Mutate and ensure balance
        mutated = mutate(current_solution, mutation_size)
        if mutated.count(0) != len(mutated) // 2:
            mutated = balance_child(mutated)

        # Calculate remaining passes
        remaining_passes = float('inf')
        if max_total_passes is not None:
            remaining_passes = max(0, max_total_passes - total_passes_consumed)
            if remaining_passes == 0: break
            remaining_passes = int(remaining_passes)

        # Run FM
        try:
            local_opt, passes_this_call = fm_heuristic(mutated, graph, max_passes=remaining_passes)
            total_passes_consumed += passes_this_call
        except Exception as e:
            print(f"Error in ILS_Ad fm_heuristic call: {e}")
            stagnation_count += 1 # Count error as stagnation
            continue # Skip adaptation logic for this iteration

        if local_opt is None:
            stagnation_count += 1 # Count FM failure as stagnation
            continue # Skip adaptation logic

        # Evaluate the result
        local_opt_fit = fitness_function(local_opt, graph)
        fitness_history.append(local_opt_fit)
        if len(fitness_history) > stagnation_window * 2: # Keep history size manageable
            fitness_history.pop(0)


        # --- Update Strategy Based on Outcome ---
        if local_opt_fit < current_fit: # Improvement found
            current_solution = local_opt.copy()
            current_fit = local_opt_fit
            stagnation_count = 0 # Reset stagnation counter

            # Decrease mutation strength (become more conservative)
            current_mutation_mean = max(min_mutation_size, current_mutation_mean * decay_factor)
            current_mutation_std_dev = max(1.0, current_mutation_std_dev * decay_factor) # Ensure std dev doesn't go below 1

            # Update overall best if needed
            if current_fit < best_fit:
                best_solution = current_solution.copy()
                best_fit = current_fit
        else: # No improvement
            stagnation_count += 1

        # Record mutation parameters (optional)
        mutation_history.append((iteration, mutation_size, current_mutation_mean, current_mutation_std_dev, current_fit))

        # --- Check for Stagnation and Increase Mutation (only if not forcing limits) ---
        if not force_run_to_pass_limit and stagnation_count >= stagnation_threshold:
            is_stuck = True
            # Check if fitness has actually plateaued recently
            if len(fitness_history) >= stagnation_window:
                # Calculate variance or range of recent fitness values
                recent_fitness = fitness_history[-stagnation_window:]
                if np.ptp(recent_fitness) > stagnation_tolerance: # Use peak-to-peak range
                # if np.std(recent_fitness) > stagnation_tolerance: # Or use standard deviation
                    is_stuck = False # Still fluctuating enough

            if is_stuck:
                # Increase mutation strength (become more explorative)
                current_mutation_mean = min(max_mutation_size, current_mutation_mean * increase_factor)
                # Increase std dev, but keep it relative to the mean
                current_mutation_std_dev = min(current_mutation_mean / 2, current_mutation_std_dev * increase_factor)
                current_mutation_std_dev = max(1.0, current_mutation_std_dev) # Ensure minimum std dev
                # print(f"   Stagnation detected at iter {iteration}. Increasing mutation mean to {current_mutation_mean:.2f}, std dev to {current_mutation_std_dev:.2f}")
                stagnation_count = 0 # Reset counter after adapting

    # Return relevant information
    return (best_solution, best_fit, mutation_history, iteration,
            stagnation_count, int(current_mutation_mean), total_passes_consumed)


def GLS(graph, population_size, stopping_crit=None, max_total_passes=None, time_limit=None):
    """ Genetic Local Search (GLS) using FM Heuristic. """
    start_time = time.perf_counter()
    total_passes_consumed = 0
    population = []
    fitness_values = []
    force_run_to_pass_limit = (max_total_passes is not None and time_limit is None)

    # --- Population Initialization ---
    # print("GLS: Initializing population...")
    for i in range(population_size):
        current_time = time.perf_counter()
        # Check limits during initialization
        if time_limit is not None and current_time - start_time >= time_limit: break
        if max_total_passes is not None and total_passes_consumed >= max_total_passes: break

        initial_sol = generate_random_solution()

        # Calculate remaining passes for this FM call
        remaining_passes = float('inf')
        if max_total_passes is not None:
            remaining_passes = max(0, max_total_passes - total_passes_consumed)
            if remaining_passes == 0: break
            remaining_passes = int(remaining_passes)

        try:
            # Apply local search (FM) to each initial random solution
            sol_opt, passes_this_call = fm_heuristic(initial_sol, graph, max_passes=remaining_passes)
            total_passes_consumed += passes_this_call
            if sol_opt is not None:
                population.append(sol_opt)
                fitness_values.append(fitness_function(sol_opt, graph))
            # else: print(f"   Warning: FM failed during GLS initialization {i+1}.")
        except Exception as e:
            print(f"  Error during GLS init fm_heuristic call {i+1}: {e}")
            continue # Skip this individual if FM fails

    if not population:
        print("GLS Error: Failed to initialize any population members.")
        return generate_random_solution(), total_passes_consumed # Return fallback

    # Find initial best
    best_fit_idx = np.argmin(fitness_values) if fitness_values else -1
    if best_fit_idx == -1 :
        print("GLS Error: No valid fitness values after initialization.")
        return generate_random_solution(), total_passes_consumed # Fallback
    best_fit = fitness_values[best_fit_idx]
    best_solution = population[best_fit_idx].copy() # Store copy
    generation_without_improvement = 0
    gen = 0
    # print(f"GLS Initial Population Size: {len(population)}, Initial Best Fitness: {best_fit}")


    # --- Evolutionary Loop ---
    # Use a high generation limit only if not forcing pass/time limit
    effective_max_generations = float('inf') if force_run_to_pass_limit else 100000

    while gen < effective_max_generations:
        gen += 1
        current_time = time.perf_counter()

        # Check stopping conditions
        if time_limit is not None and current_time - start_time >= time_limit: break
        if max_total_passes is not None and total_passes_consumed >= max_total_passes: break
        # Use generation stopping criterion only if provided and not forcing limits
        if not force_run_to_pass_limit and stopping_crit is not None and generation_without_improvement >= stopping_crit:
            # print(f"GLS: Stopping after {stopping_crit} generations without improvement.")
            break
        if len(population) < 2:
            # print("GLS Warning: Population size dropped below 2. Stopping.")
            break # Need at least two parents for crossover


        # --- Parent Selection (Random) ---
        # Could implement more sophisticated selection (e.g., tournament, roulette)
        parent1_idx = random.randrange(len(population))
        parent2_idx = random.randrange(len(population))
        # Ensure parents are different (optional, but common)
        # while parent1_idx == parent2_idx and len(population) > 1:
        #     parent2_idx = random.randrange(len(population))
        parent1 = population[parent1_idx]
        parent2 = population[parent2_idx]

        # --- Crossover ---
        child = crossover(parent1, parent2)
        # Ensure child is balanced (crossover should handle, but check)
        if child.count(0) != len(child)//2:
            child = balance_child(child)


        # --- Local Search on Child ---
        remaining_passes = float('inf')
        if max_total_passes is not None:
            remaining_passes = max(0, max_total_passes - total_passes_consumed)
            if remaining_passes == 0: break
            remaining_passes = int(remaining_passes)

        try:
            child_opt, passes_this_call = fm_heuristic(child, graph, max_passes=remaining_passes)
            total_passes_consumed += passes_this_call
        except Exception as e:
            print(f"  Error during GLS evolve fm_heuristic gen {gen}: {e}")
            # Decide how to handle: skip generation, replace random, etc.
            generation_without_improvement += 1 # Count as no improvement
            continue # Skip replacement for this generation

        if child_opt is None:
            # print(f"  Warning: FM failed for child in GLS gen {gen}.")
            generation_without_improvement += 1
            continue # Skip replacement


        # --- Selection/Replacement (Replace Worst) ---
        fit_child = fitness_function(child_opt, graph)
        worst_fit_idx = np.argmax(fitness_values) # Find index of worst individual
        worst_fit = fitness_values[worst_fit_idx]

        improvement_found_this_gen = False
        # Replace worst only if child is better or equal (or just better)
        if fit_child < worst_fit: # Use < for strict improvement, <= to allow equal fitness
            # print(f"   GLS Gen {gen}: Replacing worst (fit {worst_fit}) with child (fit {fit_child})")
            population[worst_fit_idx] = child_opt.copy() # Replace with copy
            fitness_values[worst_fit_idx] = fit_child

            # Update overall best if child is the new best
            if fit_child < best_fit:
                # print(f"   GLS Gen {gen}: New best fitness found: {fit_child}")
                best_fit = fit_child
                best_solution = child_opt.copy() # Store copy
                improvement_found_this_gen = True
        # else: print(f"   GLS Gen {gen}: Child (fit {fit_child}) not better than worst (fit {worst_fit}).")


        # Update stagnation counter
        if improvement_found_this_gen:
            generation_without_improvement = 0
        else:
            generation_without_improvement += 1

    # print(f"GLS finished after {gen} generations. Best fitness: {best_fit}")
    return best_solution, total_passes_consumed


# --- Helper Functions for Parallel Execution ---
# These wrap the single-run logic for each experiment type

def _run_single_mls_pass_limited(args):
    """ Helper for running one MLS pass-limited trial. """
    graph, run_idx, fm_passes_limit = args
    start_time = time.perf_counter()
    # Use a high num_starts as it's limited by passes
    best_sol, passes_consumed = MLS(graph, num_starts=100000, max_total_passes=fm_passes_limit, time_limit=None)
    end_time = time.perf_counter()
    comp_time = end_time - start_time
    if best_sol is None:
        fit = float('inf'); solution_str = ""
    else:
        fit = fitness_function(best_sol, graph); solution_str = "".join(map(str, best_sol))
    # Return tuple matching expected format for aggregation
    return ("MLS_PassBased", run_idx, "", "", fit, comp_time, passes_consumed, "", solution_str)

def _run_single_simple_ils_pass_limited(args):
    """ Helper for running one Simple ILS pass-limited trial. """
    graph, run_idx, mutation_size, fm_passes_limit = args
    initial_sol = generate_random_solution()
    start_time = time.perf_counter()
    best_sol, unchanged_count, passes_consumed = ILS(
        graph, initial_sol, mutation_size, max_total_passes=fm_passes_limit, time_limit=None
    )
    end_time = time.perf_counter()
    comp_time = end_time - start_time
    if best_sol is None:
        fit = float('inf'); solution_str = ""; unchanged_count = -1
    else:
        fit = fitness_function(best_sol, graph); solution_str = "".join(map(str, best_sol))
    return ("Simple_ILS_PassBased", run_idx, mutation_size, "", fit, comp_time, passes_consumed, unchanged_count, solution_str)

def _run_single_gls_pass_limited(args):
    """ Helper for running one GLS pass-limited trial. """
    graph, run_idx, stopping_crit, population_size, fm_passes_limit = args
    start_time = time.perf_counter()
    best_sol, passes_consumed = GLS(
        graph, population_size, stopping_crit, max_total_passes=fm_passes_limit, time_limit=None
    )
    end_time = time.perf_counter()
    comp_time = end_time - start_time
    if best_sol is None:
        fit = float('inf'); solution_str = ""
    else:
        fit = fitness_function(best_sol, graph); solution_str = "".join(map(str, best_sol))
    return ("GLS_PassBased", run_idx, "", stopping_crit, fit, comp_time, passes_consumed, "", solution_str)

def _run_single_ils_adaptive_pass_limited(args):
    """ Helper for running one Adaptive ILS pass-limited trial. """
    graph, run_idx, fm_passes_limit = args
    initial_sol = generate_random_solution()
    start_time = time.perf_counter()
    res_tuple = ILS_adaptive(
        graph, initial_sol, max_total_passes=fm_passes_limit, time_limit=None
    )
    end_time = time.perf_counter()
    comp_time = end_time - start_time
    # Unpack results carefully
    best_sol, best_fit_val, _, _, final_stag_count, final_mut_mean, passes_consumed = res_tuple
    if best_sol is None:
        fit = float('inf'); solution_str = ""; final_stag_count = -1; final_mut_mean = -1
    else:
        fit = best_fit_val; solution_str = "".join(map(str, best_sol))
    # Use final_mut_mean for 'Mutation_Size', final_stag_count for 'Unchanged_Count'
    return ("ILS_Adaptive_PassBased", run_idx, final_mut_mean, "", fit, comp_time, passes_consumed, final_stag_count, solution_str)

def _run_single_ils_annealing_pass_limited(args):
    """ Helper for running one ILS Annealing pass-limited trial. """
    graph, run_idx, mutation_size, fm_passes_limit = args
    initial_sol = generate_random_solution()
    start_time = time.perf_counter()
    best_sol, unchanged_count, passes_consumed = ILS_annealing(
        graph, initial_sol, mutation_size, max_total_passes=fm_passes_limit, time_limit=None
    )
    end_time = time.perf_counter()
    comp_time = end_time - start_time
    if best_sol is None:
        fit = float('inf'); solution_str = ""; unchanged_count = -1
    else:
        fit = fitness_function(best_sol, graph); solution_str = "".join(map(str, best_sol))
    return ("ILS_Annealing_PassBased", run_idx, mutation_size, "", fit, comp_time, passes_consumed, unchanged_count, solution_str)

# --- Runtime Experiment Helpers ---
def _run_single_mls_runtime(args):
    """ Helper for one MLS runtime trial. """
    graph, run_idx, time_limit = args
    best_sol, passes_consumed = MLS(graph, num_starts=1000000, time_limit=time_limit, max_total_passes=None)
    if best_sol is None: fit = float('inf'); solution_str = ""
    else: fit = fitness_function(best_sol, graph); solution_str = "".join(map(str, best_sol))
    return ("MLS_Runtime", run_idx, "", "", fit, time_limit, passes_consumed, "", solution_str)

def _run_single_ils_annealing_runtime(args):
    """ Helper for one ILS Annealing runtime trial. """
    graph, run_idx, time_limit, mutation_size = args
    initial_sol = generate_random_solution()
    best_sol, unchanged_count, passes_consumed = ILS_annealing(
        graph, initial_sol, mutation_size, time_limit=time_limit, max_total_passes=None
    )
    if best_sol is None: fit = float('inf'); solution_str = ""; unchanged_count = -1
    else: fit = fitness_function(best_sol, graph); solution_str = "".join(map(str, best_sol))
    return ("ILS_Annealing_Runtime", run_idx, mutation_size, "", fit, time_limit, passes_consumed, unchanged_count, solution_str)

def _run_single_ils_simple_runtime(args):
    """ Helper for one Simple ILS runtime trial. """
    graph, run_idx, time_limit, mutation_size = args
    initial_sol = generate_random_solution()
    best_sol, unchanged_count, passes_consumed = ILS(
        graph, initial_sol, mutation_size, time_limit=time_limit, max_total_passes=None
    )
    if best_sol is None: fit = float('inf'); solution_str = ""; unchanged_count = -1
    else: fit = fitness_function(best_sol, graph); solution_str = "".join(map(str, best_sol))
    return ("ILS_Simple_Runtime", run_idx, mutation_size, "", fit, time_limit, passes_consumed, unchanged_count, solution_str)

def _run_single_gls_runtime(args):
    """ Helper for one GLS runtime trial. """
    graph, run_idx, time_limit, stopping_crit, population_size = args
    best_sol, passes_consumed = GLS(
        graph, population_size, stopping_crit, time_limit=time_limit, max_total_passes=None
    )
    if best_sol is None: fit = float('inf'); solution_str = ""
    else: fit = fitness_function(best_sol, graph); solution_str = "".join(map(str, best_sol))
    return ("GLS_Runtime", run_idx, "", stopping_crit, fit, time_limit, passes_consumed, "", solution_str)

def _run_single_ils_adaptive_runtime(args):
    """ Helper for one Adaptive ILS runtime trial. """
    graph, run_idx, time_limit = args
    initial_sol = generate_random_solution()
    res_tuple = ILS_adaptive(
        graph, initial_sol, time_limit=time_limit, max_total_passes=None
    )
    best_sol, best_fit_val, _, _, final_stag_count, final_mut_mean, passes_consumed = res_tuple
    if best_sol is None: fit = float('inf'); solution_str = ""; final_stag_count = -1; final_mut_mean = -1
    else: fit = best_fit_val; solution_str = "".join(map(str, best_sol))
    return ("ILS_Adaptive_Runtime", run_idx, final_mut_mean, "", fit, time_limit, passes_consumed, final_stag_count, solution_str)


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Setup ---
    try:
        # Assume script is in the same directory as the graph file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        graph_file_path = os.path.join(script_dir, "Graph500.txt")
        if not os.path.exists(graph_file_path):
            # Try looking in the current working directory as a fallback
            graph_file_path = "Graph500.txt"
            if not os.path.exists(graph_file_path):
                 raise FileNotFoundError(f"Graph file not found in script directory or current directory: Graph500.txt")
        graph = read_graph_data(graph_file_path)
        print("Graph data loaded successfully.")
    except Exception as e:
        print(f"Error during setup: {e}")
        exit() # Exit if graph loading fails

    # --- Determine Time Limit ---
    # Run this block ONCE to find your DYNAMIC_TIME_LIMIT
    # Set CALCULATE_TIME_LIMIT = True to run timing
    CALCULATE_TIME_LIMIT = False # <<< SET TO True TO CALCULATE, THEN False AND HARDCODE VALUE
    DYNAMIC_TIME_LIMIT = 62.0  # <<< HARDCODE MEDIAN TIME HERE AFTER CALCULATING

    if CALCULATE_TIME_LIMIT:
        print("\n--- Measuring Time for 10000 FM Passes ---")
        NUM_TIMING_RUNS = 10
        PASSES_TO_TIME = FM_PASSES_LIMIT # Use the same limit as experiments
        fm_times = []
        print(f"(Running MLS {NUM_TIMING_RUNS} times with cumulative pass limit of {PASSES_TO_TIME})")
        # Run timing sequentially to avoid interference
        for i in range(NUM_TIMING_RUNS):
            print(f"  Timing Run {i + 1}/{NUM_TIMING_RUNS}...")
            initial_sol = generate_random_solution()
            start_time = time.perf_counter()
            try:
                # Use MLS configured to stop *exactly* after 10k passes
                # Provide a huge num_starts so it's limited by passes
                _, passes_executed = MLS(graph, num_starts=100000, max_total_passes=PASSES_TO_TIME)
                end_time = time.perf_counter()
                # Use time only if pass limit was actually reached (or exceeded slightly)
                # Allow for slight overshoot due to passes_this_call granularity
                if passes_executed >= PASSES_TO_TIME * 0.95: # Allow 5% tolerance
                    duration = end_time - start_time
                    fm_times.append(duration)
                    print(f"    Time for {passes_executed} passes: {duration:.6f} seconds")
                else:
                     print(f"    Warning: MLS timing run finished early ({passes_executed}/{PASSES_TO_TIME} passes). Discarding time.")
            except Exception as e:
                print(f"    Error during MLS timing run: {e}")

        if fm_times:
            median_fm_time = np.median(fm_times)
            mean_fm_time = np.mean(fm_times)
            std_fm_time = np.std(fm_times)
            print(f"\n--- Timing Results ---")
            print(f"  Times recorded: {fm_times}")
            print(f"  Median Time for ~{PASSES_TO_TIME} passes: {median_fm_time:.6f} seconds")
            print(f"  Mean Time for ~{PASSES_TO_TIME} passes:   {mean_fm_time:.6f} seconds")
            print(f"  Std Dev Time for ~{PASSES_TO_TIME} passes: {std_fm_time:.6f} seconds")
            print(f"\n  >>> Set DYNAMIC_TIME_LIMIT = {median_fm_time:.6f} (median) in the script <<<")
        else:
            print("\n  No valid FM timing runs completed. Cannot determine DYNAMIC_TIME_LIMIT.")
            print("  >>> Please ensure MLS runs correctly and set DYNAMIC_TIME_LIMIT manually. <<<")
        print("--- End of FM Timing Measurement ---")
        exit() # Stop script after timing measurement

    # --- Use the determined (or hardcoded) time limit ---
    print(f"Using Dynamic Time Limit: {DYNAMIC_TIME_LIMIT:.6f} seconds for runtime experiments.")

    # --- List to store all results ---
    all_results_list = []
    # Use a lock for thread-safe appending to the list
    results_lock = threading.Lock()

    # --- ThreadPoolExecutor ---
    # Use 'with' statement for automatic shutdown
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:

        # -----------------------------
        # Pass-Limited Experiments (Comparison a) - Run in Parallel
        # -----------------------------
        print(f"\n--- Running Pass-Limited Experiments ({EXPERIMENT_RUNS} runs each, Limit: {FM_PASSES_LIMIT} passes, Parallel: {NUM_WORKERS} threads) ---")
        futures = [] # Store Future objects

        # --- Submit MLS Pass-Limited Runs ---
        print("  Submitting MLS Pass-Limited runs...")
        for i in range(EXPERIMENT_RUNS):
            futures.append(executor.submit(_run_single_mls_pass_limited, (graph, i, FM_PASSES_LIMIT)))

        # --- Submit Simple ILS Pass-Limited Runs ---
        print("  Submitting Simple ILS Pass-Limited runs...")
        mutation_sizes_ils = [5, 25, 50, 75, 100, 125, 150]  # Example list
        for ms in mutation_sizes_ils:
            print(f"    Simple ILS Mutation Size: {ms}")
            for i in range(EXPERIMENT_RUNS):
                futures.append(executor.submit(_run_single_simple_ils_pass_limited, (graph, i, ms, FM_PASSES_LIMIT)))

        # --- Submit GLS Pass-Limited Runs ---
        print("  Submitting GLS Pass-Limited runs...")
        stopping_crits_gls = [1, 2] # Example criteria
        for sc in stopping_crits_gls:
             print(f"    GLS Stopping Criterion: {sc}")
             for i in range(EXPERIMENT_RUNS):
                 futures.append(executor.submit(_run_single_gls_pass_limited, (graph, i, sc, POPULATION_SIZE, FM_PASSES_LIMIT)))

        # --- Submit ILS Adaptive Pass-Limited Runs ---
        print("  Submitting ILS Adaptive Pass-Limited runs...")
        for i in range(EXPERIMENT_RUNS):
             futures.append(executor.submit(_run_single_ils_adaptive_pass_limited, (graph, i, FM_PASSES_LIMIT)))

        # --- Submit ILS Annealing Pass-Limited Runs ---
        print("  Submitting ILS Annealing Pass-Limited runs...")
        mutation_sizes_ils_a = [5, 25, 50, 75, 100, 125, 150]  # Example list
        for ms in mutation_sizes_ils_a:
             print(f"    ILS Annealing Mutation Size: {ms}")
             for i in range(EXPERIMENT_RUNS):
                 futures.append(executor.submit(_run_single_ils_annealing_pass_limited, (graph, i, ms, FM_PASSES_LIMIT)))

        # --- Collect Pass-Limited Results ---
        print(f"\n  Collecting results from {len(futures)} pass-limited tasks...")
        pass_limited_results_raw = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result_tuple = future.result()
                pass_limited_results_raw.append(result_tuple)
                # Optional: print progress
                # print(f"    Finished: {result_tuple[0]} Run {result_tuple[1]} (Params: M={result_tuple[2]}, S={result_tuple[3]}) -> Fit={result_tuple[4]}")
            except Exception as exc:
                print(f'    Task generated an exception: {exc}')

        # Convert raw tuples to dictionaries
        for r in pass_limited_results_raw:
            all_results_list.append({
                "Experiment": r[0], "Run": r[1], "Mutation_Size": r[2], "Stopping_Crit": r[3],
                "Fitness": r[4], "Comp_Time": r[5], "Actual_Passes": r[6],
                "Unchanged_Count": r[7], "Solution": r[8]
            })
        print(f"  Collected {len(pass_limited_results_raw)} pass-limited results.")


        # --- Determine best parameters for Runtime tests (using collected results) ---
        print("\n--- Determining Best Parameters for Runtime Tests ---")
        best_ils_mutation_size = 75 # Default
        best_ils_annealing_mutation_size = 75 # Default
        best_gls_stopping_crit = 2 # Default

        df_pass_results = pd.DataFrame(all_results_list) # Use all results collected so far

        try:
            df_simple_ils = df_pass_results[df_pass_results['Experiment'] == 'Simple_ILS_PassBased'].copy()
            if not df_simple_ils.empty:
                df_simple_ils['Mutation_Size'] = pd.to_numeric(df_simple_ils['Mutation_Size'], errors='coerce')
                df_simple_ils['Fitness'] = pd.to_numeric(df_simple_ils['Fitness'], errors='coerce')
                df_simple_ils.dropna(subset=['Mutation_Size', 'Fitness'], inplace=True)
                if not df_simple_ils.empty:
                    median_fits = df_simple_ils.groupby('Mutation_Size')['Fitness'].median()
                    if not median_fits.empty:
                        best_ils_mutation_size = int(median_fits.idxmin())
        except Exception as e: print(f"Could not determine best Simple ILS param: {e}")

        try:
            df_ils_anneal = df_pass_results[df_pass_results['Experiment'] == 'ILS_Annealing_PassBased'].copy()
            if not df_ils_anneal.empty:
                df_ils_anneal['Mutation_Size'] = pd.to_numeric(df_ils_anneal['Mutation_Size'], errors='coerce')
                df_ils_anneal['Fitness'] = pd.to_numeric(df_ils_anneal['Fitness'], errors='coerce')
                df_ils_anneal.dropna(subset=['Mutation_Size', 'Fitness'], inplace=True)
                if not df_ils_anneal.empty:
                    median_fits = df_ils_anneal.groupby('Mutation_Size')['Fitness'].median()
                    if not median_fits.empty:
                        best_ils_annealing_mutation_size = int(median_fits.idxmin())
        except Exception as e: print(f"Could not determine best ILS Annealing param: {e}")

        try:
            df_gls = df_pass_results[df_pass_results['Experiment'] == 'GLS_PassBased'].copy()
            if not df_gls.empty and 'Stopping_Crit' in df_gls.columns:
                 df_gls['Stopping_Crit'] = pd.to_numeric(df_gls['Stopping_Crit'], errors='coerce')
                 df_gls['Fitness'] = pd.to_numeric(df_gls['Fitness'], errors='coerce')
                 df_gls.dropna(subset=['Stopping_Crit', 'Fitness'], inplace=True)
                 if not df_gls.empty:
                     median_fits = df_gls.groupby('Stopping_Crit')['Fitness'].median()
                     if not median_fits.empty:
                         best_gls_stopping_crit = int(median_fits.idxmin())
            # else: print("Warning: Stopping_Crit column missing or empty GLS_PassBased results.")
        except Exception as e: print(f"Could not determine best GLS param: {e}")

        print(f"Using parameters for Runtime: ILS Mut={best_ils_mutation_size}, ILS_A Mut={best_ils_annealing_mutation_size}, GLS Stop={best_gls_stopping_crit}")


        # -----------------------------
        # Runtime experiments (Comparison b - Fixed Time Limit) - Run in Parallel
        # -----------------------------
        print(f"\n--- Running Runtime Experiments ({RUNTIME_RUNS} repetitions, Limit: {DYNAMIC_TIME_LIMIT:.6f}s, Parallel: {NUM_WORKERS} threads) ---")
        futures = [] # Reset futures list for runtime tasks

        for i in range(RUNTIME_RUNS):
             print(f"  Submitting Runtime Repetition {i + 1}/{RUNTIME_RUNS}...")
             # Submit all 5 algorithm runs for this repetition
             futures.append(executor.submit(_run_single_mls_runtime, (graph, i, DYNAMIC_TIME_LIMIT)))
             futures.append(executor.submit(_run_single_ils_annealing_runtime, (graph, i, DYNAMIC_TIME_LIMIT, best_ils_annealing_mutation_size)))
             futures.append(executor.submit(_run_single_ils_simple_runtime, (graph, i, DYNAMIC_TIME_LIMIT, best_ils_mutation_size)))
             futures.append(executor.submit(_run_single_gls_runtime, (graph, i, DYNAMIC_TIME_LIMIT, best_gls_stopping_crit, POPULATION_SIZE)))
             futures.append(executor.submit(_run_single_ils_adaptive_runtime, (graph, i, DYNAMIC_TIME_LIMIT)))

        # --- Collect Runtime Results ---
        print(f"\n  Collecting results from {len(futures)} runtime tasks...")
        runtime_results_raw = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result_tuple = future.result()
                runtime_results_raw.append(result_tuple)
                # Optional: print progress
                # print(f"    Finished: {result_tuple[0]} Rep {result_tuple[1]} -> Fit={result_tuple[4]}, Passes={result_tuple[6]}")
            except Exception as exc:
                print(f'    Task generated an exception: {exc}')

        # Append runtime results to the main list
        for r in runtime_results_raw:
             all_results_list.append({
                 "Experiment": r[0], "Run": r[1], "Mutation_Size": r[2], "Stopping_Crit": r[3],
                 "Fitness": r[4], "Comp_Time": r[5], "Actual_Passes": r[6],
                 "Unchanged_Count": r[7], "Solution": r[8]
             })
        print(f"  Collected {len(runtime_results_raw)} runtime results.")

    # --- End of ThreadPoolExecutor block ---

    # -----------------------------
    # Final DataFrame Creation and Saving
    # -----------------------------
    print("\n--- Aggregating All Results ---")
    if not all_results_list:
        print("Warning: No results were collected. Cannot create DataFrame.")
    else:
        df_experiments = pd.DataFrame(all_results_list)

        # Define final columns and fill missing values AFTER creating DataFrame
        final_columns = [
            "Experiment", "Run", "Mutation_Size", "Stopping_Crit",
            "Fitness", "Comp_Time", "Actual_Passes", "Unchanged_Count", "Solution"
        ]
        # Reindex to ensure all columns exist and are in order
        df_experiments = df_experiments.reindex(columns=final_columns)

        # Fill NaNs with appropriate placeholders or values
        df_experiments.fillna({
            "Mutation_Size": "",       # Use empty string for non-applicable params
            "Stopping_Crit": "",       # Use empty string
            "Unchanged_Count": "",     # Use empty string or maybe -1 if 0 is meaningful
            "Actual_Passes": -1,       # Use -1 to indicate N/A or error
            "Fitness": float('inf'),   # Use infinity for failed runs
            "Comp_Time": -1.0,         # Use -1 for time if run failed early
            "Solution": ""             # Empty string for solution if none found
             }, inplace=True)

        # Optional: Convert types for better analysis later
        # df_experiments['Fitness'] = pd.to_numeric(df_experiments['Fitness'], errors='coerce')
        # df_experiments['Actual_Passes'] = pd.to_numeric(df_experiments['Actual_Passes'], errors='coerce').astype('Int64') # Use nullable Int

        output_csv_path = os.path.join(script_dir, "experiment_results_combined_multithreaded.csv")
        try:
            df_experiments.to_csv(output_csv_path, index=False, float_format='%.6f')
            print(f"\nAll combined experiment results saved in '{output_csv_path}'.")
        except Exception as e:
            print(f"\nError saving results to CSV: {e}")

    print("\n--- Experiment Script Finished ---")
