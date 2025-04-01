import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import os
import warnings
import sys # Import sys for exit

# Attempt to import networkx, provide instructions if missing
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: networkx library not found. Graph visualization and fitness verification will be skipped.")
    print("Install it using: pip install networkx")


# --- Configuration ---
CSV_FILENAME = "experiment_results_combined.csv" # Input CSV file name
GRAPH_FILENAME = "Graph500.txt" # Graph file needed for verification and visualization
OUTPUT_DIR = "analysis_results" # Directory to save plots and tables
ALPHA = 0.05 # Significance level for statistical tests
NUM_VERTICES = 500 # Define the expected number of vertices for verification
EXPECTED_HALF_COUNT = NUM_VERTICES // 2
RUNTIME_RUNS = 25 # Expected number of runs for runtime experiments (used later)

# --- Global List for Output Files ---
generated_files = []

# --- Helper Functions ---

def calculate_summary_stats(data, group_col=None):
    """
    Calculates summary statistics for Fitness, Comp_Time, and Actual_Passes.

    Args:
        data (pd.DataFrame): DataFrame containing the experiment results.
        group_col (str, optional): Column name to group by. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with summary statistics for each group.
                      Includes Count, Min/Median/Mean/Std Dev/Max for Fitness,
                      Mean/Std Dev for Comp_Time and Actual_Passes, and Best Fitness Count.
    """
    if group_col:
        if group_col not in data.columns:
            print(f"Warning: Grouping column '{group_col}' not found in data. Calculating overall stats.")
            grouped_indices = {'all_data': data.index}
        else:
            # Group by the specified column, dropping NA values in that column
            grouped_indices = data.dropna(subset=[group_col]).groupby(group_col).groups
    else:
        grouped_indices = {'all_data': data.index}

    stats_list = []
    for name, idx in grouped_indices.items():
        group_data = data.loc[idx]
        if group_data.empty:
            continue

        stats = {'Group': name} # Initialize stats dict with group name

        # Calculate Fitness Stats
        if 'Fitness' in group_data.columns:
            fitness_data = pd.to_numeric(group_data['Fitness'], errors='coerce').dropna()
            if fitness_data.empty:
                stats['Count'] = 0 # Indicate no valid fitness data for count
            else:
                best_fitness = fitness_data.min()
                count_total = fitness_data.count()
                count_best = (fitness_data == best_fitness).sum()
                stats.update({
                    'Count': count_total,
                    'Fitness_Min': best_fitness,
                    'Fitness_Median': fitness_data.median(),
                    'Fitness_Mean': fitness_data.mean(),
                    'Fitness_StdDev': fitness_data.std(),
                    'Fitness_Max': fitness_data.max(),
                    'Best_Fitness_Count': f"{count_best}/{count_total}" if count_total > 0 else "0/0"
                })
        else:
             stats['Count'] = len(group_data) # Use group size if Fitness is missing
             print(f"Warning: 'Fitness' column missing for group '{name}'. Cannot calculate fitness stats.")


        # Calculate Time Stats
        if 'Comp_Time' in group_data.columns:
            time_data = pd.to_numeric(group_data['Comp_Time'], errors='coerce').dropna()
            stats['Time_Mean'] = time_data.mean() if not time_data.empty else np.nan
            stats['Time_StdDev'] = time_data.std() if not time_data.empty else np.nan
        else:
            stats['Time_Mean'] = np.nan
            stats['Time_StdDev'] = np.nan

        # Calculate Pass Stats
        if 'Actual_Passes' in group_data.columns:
            # Filter out placeholder -1 values before calculating stats
            passes_data = pd.to_numeric(group_data['Actual_Passes'], errors='coerce')
            passes_data = passes_data[passes_data != -1].dropna()
            stats['Passes_Mean'] = passes_data.mean() if not passes_data.empty else np.nan
            stats['Passes_StdDev'] = passes_data.std() if not passes_data.empty else np.nan
        else:
            stats['Passes_Mean'] = np.nan
            stats['Passes_StdDev'] = np.nan


        # Calculate Unchanged Count Stats (for relevant experiments)
        if 'Unchanged_Count' in group_data.columns:
            unchanged_data = group_data['Unchanged_Count']
            # Handle potential strings before converting to numeric
            if pd.api.types.is_string_dtype(unchanged_data):
                 unchanged_data = unchanged_data.replace("", np.nan) # Replace empty strings with NaN
            unchanged_data = pd.to_numeric(unchanged_data, errors='coerce')
            # Filter out placeholder -1 values
            unchanged_data = unchanged_data[unchanged_data != -1].dropna()
            stats['Unchanged_Mean'] = unchanged_data.mean() if not unchanged_data.empty else np.nan
            stats['Unchanged_StdDev'] = unchanged_data.std() if not unchanged_data.empty else np.nan
        else:
            stats['Unchanged_Mean'] = np.nan
            stats['Unchanged_StdDev'] = np.nan

        stats_list.append(stats)

    if not stats_list: return pd.DataFrame()

    stats_df = pd.DataFrame(stats_list)
    # Define desired column order
    cols_order = [
        'Group', 'Count', 'Fitness_Min', 'Fitness_Median', 'Fitness_Mean', 'Fitness_StdDev',
        'Fitness_Max', 'Best_Fitness_Count', 'Time_Mean', 'Time_StdDev',
        'Passes_Mean', 'Passes_StdDev', 'Unchanged_Mean', 'Unchanged_StdDev'
    ]
    # Filter out columns that don't exist in the DataFrame and reorder
    cols_present = [col for col in cols_order if col in stats_df.columns]
    stats_df = stats_df[cols_present]

    # Set index or clean up Group column
    if group_col and 'Group' in stats_df.columns:
        stats_df = stats_df.set_index('Group')
    elif 'Group' in stats_df.columns and not group_col and len(stats_df) == 1 and stats_df['Group'].iloc[0] == 'all_data':
        stats_df = stats_df.drop(columns=['Group'])

    return stats_df


def run_mannwhitneyu_test(data1, data2, name1, name2, alpha=0.05):
    """
    Performs and interprets Mann-Whitney U test on the 'Fitness' column.
    Returns p-value and a result string. Handles potential errors and empty data.
    """
    # Ensure data is pandas Series for dropna/empty checks
    data1 = pd.Series(data1); data2 = pd.Series(data2)
    data1_clean = pd.to_numeric(data1, errors='coerce').dropna()
    data2_clean = pd.to_numeric(data2, errors='coerce').dropna()

    if data1_clean.empty or data2_clean.empty:
        print(f"  Skipping test between {name1} and {name2} due to empty/non-numeric data.")
        return None, "Skipped (Empty/Non-Numeric Data)"

    try:
        # alternative='less': H1 is data1 < data2 (median fitness is lower/better)
        stat_less, p_value_less = mannwhitneyu(data1_clean, data2_clean, alternative='less')
        # alternative='greater': H1 is data1 > data2 (median fitness is higher/worse)
        stat_greater, p_value_greater = mannwhitneyu(data1_clean, data2_clean, alternative='greater')

        print(f"  Mann-Whitney U Test: {name1} vs {name2}")
        print(f"    U-statistic: {stat_less:.4f}") # U stat is same for less/greater
        print(f"    P-value ({name1} < {name2}): {p_value_less:.4f}") # Test if name1 is better
        print(f"    P-value ({name1} > {name2}): {p_value_greater:.4f}") # Test if name1 is worse

        if p_value_less < alpha:
            result = f"Significant ({name1} better than {name2}, p={p_value_less:.4f})"
            print(f"    Result: {result}")
            return p_value_less, result
        elif p_value_greater < alpha:
            result = f"Significant ({name2} better than {name1}, p={p_value_greater:.4f})"
            print(f"    Result: {result}")
            return p_value_greater, result # Return the significant p-value (for name2 being better)
        else:
            # If neither one-sided test is significant, report two-sided p-value
            stat_two_sided, p_value_two_sided = mannwhitneyu(data1_clean, data2_clean, alternative='two-sided')
            result = f"Not Significant (p={p_value_two_sided:.4f})"
            print(f"    Result: {result}")
            return p_value_two_sided, result
    except ValueError as e:
        # Handle cases like identical data or insufficient data
        print(f"  Skipping test between {name1} and {name2} due to error: {e}")
        if "identical" in str(e).lower() or data1_clean.equals(data2_clean):
             return 1.0, "Not Significant (Identical/Constant Data)"
        else: return None, f"Skipped (Error: {e})"
    except Exception as e:
        print(f"  An unexpected error occurred during Mann-Whitney U test for {name1} vs {name2}: {e}")
        return None, f"Skipped (Unexpected Error: {e})"


def save_output(df_or_fig, filename, is_plot=False, float_format='%.4f'):
    """Saves DataFrame to CSV or figure to PNG in the output directory."""
    global generated_files
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        try: os.makedirs(OUTPUT_DIR)
        except OSError as e: print(f"Error creating output directory {OUTPUT_DIR}: {e}"); return

    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        if is_plot:
            df_or_fig.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close(df_or_fig) # Close the figure to free memory
            print(f"Plot saved: {filepath}")
        else:
            # Ensure DataFrame index is saved if it's meaningful (like parameters)
            save_index = not isinstance(df_or_fig.index, pd.RangeIndex)
            df_or_fig.to_csv(filepath, float_format=float_format, index=save_index)
            print(f"Table saved: {filepath}")
        generated_files.append(filepath) # Add to list of generated files
    except Exception as e:
        print(f"Error saving output {filepath}: {e}")


def verify_solution_balance(solution_str, expected_len, expected_ones):
    """Checks if a solution string has the expected length and number of 1s."""
    if not isinstance(solution_str, str) or len(solution_str) != expected_len:
        return False
    ones = solution_str.count('1')
    zeros = solution_str.count('0')
    # Check if count of 0s and 1s adds up to total length (handles non-binary chars)
    if (ones + zeros) != expected_len:
        return False
    return ones == expected_ones

def read_graph_data(filename, num_vertices):
    """
    Reads graph data from the specified file format.

    Args:
        filename (str): Path to the graph file.
        num_vertices (int): Expected number of vertices.

    Returns:
        networkx.Graph or None: A NetworkX graph object if successful, None otherwise.
        dict or None: A dictionary of node positions {node_id: (x, y)} if successful, None otherwise.
    """
    if not NETWORKX_AVAILABLE:
        print("Error: NetworkX not available. Cannot read graph data.")
        return None, None

    adj = {i: set() for i in range(num_vertices)}
    pos = {} # Dictionary to store node positions {node_id: (x, y)}
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
                    vertex_id = int(parts[0]) - 1 # Convert to 0-based index
                    if not (0 <= vertex_id < num_vertices):
                        print(f"Warning: Skipping line {ln}. Vertex ID {parts[0]} out of range [1, {num_vertices}].")
                        continue

                    # --- Extract coordinates (optional, for plotting) ---
                    coord_part_index = -1
                    if len(parts) > 1 and parts[1].startswith('(') and parts[1].endswith(')'):
                        coord_part_index = 1
                        coord_str = parts[1]
                        try:
                            x_str, y_str = coord_str[1:-1].split(',')
                            pos[vertex_id] = (float(x_str), float(y_str))
                        except ValueError:
                            print(f"Warning: Could not parse coordinates '{coord_str}' on line {ln}. Using random position.")
                            pos[vertex_id] = (np.random.rand(), np.random.rand())
                    else:
                        # Assign random position if coordinates are missing or malformed
                         pos[vertex_id] = (np.random.rand(), np.random.rand())

                    # --- Find neighbors ---
                    # Find the number of neighbors (should be the first integer after ID or coordinates)
                    num_neighbors_idx = -1
                    start_search_idx = 1 if coord_part_index == -1 else coord_part_index + 1
                    for idx in range(start_search_idx, len(parts)):
                        if parts[idx].isdigit():
                            num_neighbors_idx = idx
                            break

                    if num_neighbors_idx == -1:
                        print(f"Warning: Skipping line {ln}. Could not find number of neighbors.")
                        continue

                    num_neighbors = int(parts[num_neighbors_idx])
                    neighbor_ids_str = parts[num_neighbors_idx + 1 : num_neighbors_idx + 1 + num_neighbors]

                    if len(neighbor_ids_str) != num_neighbors:
                         print(f"Warning: Skipping line {ln}. Expected {num_neighbors} neighbors, found {len(neighbor_ids_str)}.")
                         continue

                    connected_vertices = [int(n) - 1 for n in neighbor_ids_str] # Convert to 0-based
                    valid_neighbors = {nb for nb in connected_vertices if 0 <= nb < num_vertices}
                    if len(valid_neighbors) != len(connected_vertices):
                        print(f"Warning: Line {ln} contains neighbor IDs outside range [1, {num_vertices}].")
                    adj[vertex_id].update(valid_neighbors)

                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping line {ln} due to parsing error: {e}")
                    continue

        # --- Ensure symmetry and build NetworkX graph ---
        G = nx.Graph()
        for i in range(num_vertices): G.add_node(i) # Add all nodes first

        edges_added = set()
        for i in range(num_vertices):
            for neighbor in list(adj[i]):
                if 0 <= neighbor < num_vertices:
                    # Add edge if not already added (handles symmetry)
                    edge = tuple(sorted((i, neighbor)))
                    if edge not in edges_added:
                        G.add_edge(i, neighbor)
                        edges_added.add(edge)
                    # Ensure symmetry in adjacency list (optional, G handles it)
                    # if i not in adj.get(neighbor, set()):
                    #     adj.setdefault(neighbor, set()).add(i)
                else:
                    # This case should ideally not happen if filtering worked
                    adj[i].discard(neighbor)

        if G.number_of_nodes() != num_vertices:
             print(f"Warning: Graph created with {G.number_of_nodes()} nodes, expected {num_vertices}.")
        if len(pos) != num_vertices and len(pos) > 0:
             print(f"Warning: Read positions for {len(pos)} nodes, expected {num_vertices}.")


        print(f"Graph data loaded successfully: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        return G, pos

    except FileNotFoundError:
        print(f"Error: Graph file not found at {filename}.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during graph reading: {e}")
        return None, None


def calculate_actual_fitness(solution_str, graph):
    """
    Calculates the actual fitness (cut size) of a solution string for a given graph.

    Args:
        solution_str (str): The binary string representing the partition (e.g., '0110...').
        graph (networkx.Graph): The graph object.

    Returns:
        int or None: The calculated cut size, or None if input is invalid.
    """
    if not isinstance(solution_str, str) or not NETWORKX_AVAILABLE or graph is None:
        return None
    if len(solution_str) != graph.number_of_nodes():
        print(f"Error calculating fitness: Solution length {len(solution_str)} != graph size {graph.number_of_nodes()}")
        return None
    if not all(c in '01' for c in solution_str):
        print(f"Error calculating fitness: Solution contains non-binary characters.")
        return None

    cut_size = 0
    try:
        for u, v in graph.edges():
            # Nodes u and v are 0-based indices from networkx
            if 0 <= u < len(solution_str) and 0 <= v < len(solution_str):
                if solution_str[u] != solution_str[v]:
                    cut_size += 1
            else:
                # This should not happen if graph nodes match solution length check
                print(f"Warning: Edge ({u},{v}) involves node index out of solution string bounds.")
                return None # Indicate error
        return cut_size
    except IndexError:
         print(f"Error calculating fitness: Index out of bounds for solution string of length {len(solution_str)}.")
         return None
    except Exception as e:
         print(f"Error calculating fitness: {e}")
         return None


def verify_row_fitness(row, graph):
    """Helper function for df.apply to verify fitness of a single row."""
    reported_fitness_raw = row.get('Fitness')
    solution_str = row.get('Solution')

    # --- Input Validation ---
    if pd.isna(reported_fitness_raw) or solution_str is None or not isinstance(solution_str, str):
        return False # Cannot verify if data is missing/invalid type

    # Convert reported fitness to numeric (integer expected)
    try:
        # Attempt direct int conversion first, fallback to float then int
        try:
            reported_fitness = int(reported_fitness_raw)
        except ValueError:
            reported_fitness = int(float(reported_fitness_raw))
    except (ValueError, TypeError):
         print(f"Warning: Could not convert reported fitness '{reported_fitness_raw}' to integer for comparison.")
         return False # Cannot compare if reported fitness is not numeric

    # --- Calculate Actual Fitness ---
    calculated_fitness = calculate_actual_fitness(solution_str, graph)

    # --- Compare ---
    if calculated_fitness is None:
        return False # Calculation failed

    return calculated_fitness == reported_fitness


def plot_best_partition(df, graph, graph_pos):
    """Finds the best overall solution and plots the graph partition."""
    if not NETWORKX_AVAILABLE:
        print("Skipping graph visualization: networkx not available.")
        return
    if graph is None:
         print("Skipping graph visualization: Graph data not loaded.")
         return
    if 'Fitness' not in df.columns or 'Solution' not in df.columns:
        print("Warning: Cannot plot partition. 'Fitness' or 'Solution' columns missing.")
        return

    try:
        df_numeric_fitness = df.copy()
        # Convert Fitness to numeric, coercing errors, then drop rows where conversion failed
        df_numeric_fitness['Fitness'] = pd.to_numeric(df_numeric_fitness['Fitness'], errors='coerce')
        df_numeric_fitness.dropna(subset=['Fitness'], inplace=True)

        if df_numeric_fitness.empty:
            print("No valid numeric fitness values found to determine best solution for plotting.")
            return

        # Find the row with the minimum fitness
        best_row = df_numeric_fitness.loc[df_numeric_fitness['Fitness'].idxmin()]
        best_fitness = best_row['Fitness']
        best_solution_str = str(best_row['Solution']) # Ensure string
        best_experiment = best_row.get('Experiment', 'N/A') # Use .get for safety

        print(f"\n--- Visualizing Best Overall Partition ---")
        print(f"Best solution found by '{best_experiment}' with reported fitness {best_fitness}")

        # Validate the best solution string before plotting
        if len(best_solution_str) != graph.number_of_nodes():
            print(f"Error: Best solution string length ({len(best_solution_str)}) mismatch with graph size ({graph.number_of_nodes()}). Cannot plot.")
            return
        if not all(c in '01' for c in best_solution_str):
            print("Error: Best solution contains non-binary characters. Cannot plot.")
            return

        # Use pre-calculated positions if available and matching, otherwise default layout
        plot_pos = graph_pos if graph_pos and len(graph_pos) == graph.number_of_nodes() else nx.spring_layout(graph, seed=42)

        colors = ['skyblue' if bit == '0' else 'lightcoral' for bit in best_solution_str]
        fig, ax = plt.subplots(figsize=(12, 12))
        nx.draw_networkx_nodes(graph, plot_pos, node_color=colors, node_size=50, ax=ax)
        nx.draw_networkx_edges(graph, plot_pos, alpha=0.3, width=0.5, ax=ax)
        ax.set_title(f"Best Graph Partition Found (Reported Fitness = {best_fitness})\nAlgorithm: {best_experiment}")
        ax.set_xticks([]); ax.set_yticks([]); plt.axis('off')
        save_output(fig, 'plot_best_partition_visualization.png', is_plot=True)
    except KeyError as e:
         print(f"Error during graph visualization: Missing expected column - {e}")
    except Exception as e:
        print(f"Error during graph visualization: {e}")


# --- Main Analysis Script ---
if __name__ == "__main__":
    print("--- Starting Analysis Script ---")

    # --- Setup Output Directory ---
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
            print(f"Created output directory: {OUTPUT_DIR}")
        except OSError as e:
            print(f"Error creating output directory {OUTPUT_DIR}: {e}")
            sys.exit(1) # Exit if cannot create output dir

    # --- Determine Script Directory ---
    try:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback for environments where __file__ is not defined (e.g., interactive)
        script_dir = os.getcwd()
        print(f"Warning: __file__ not defined. Using current working directory: {script_dir}")

    # --- Load Data ---
    try:
        csv_path = os.path.join(script_dir, CSV_FILENAME)
        if not os.path.exists(csv_path):
             # If not found in script dir, try current working directory
             print(f"Info: '{CSV_FILENAME}' not found in script directory. Trying current directory...")
             csv_path = os.path.join(os.getcwd(), CSV_FILENAME)

        if not os.path.exists(csv_path):
             raise FileNotFoundError(f"Error: CSV file '{CSV_FILENAME}' not found in script directory or current directory.")

        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data from: {csv_path} ({len(df)} rows)")
        print(f"Columns: {df.columns.tolist()}")

        # Check for essential columns early
        required_cols = ['Experiment', 'Run', 'Fitness', 'Solution']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing essential columns in CSV: {missing_cols}. Cannot proceed with full analysis.")
            # Decide whether to exit or continue with limited analysis
            # For now, we'll print warnings later when specific columns are needed
            # sys.exit(1)

        if 'Experiment' in df.columns:
             print(f"Experiment types: {df['Experiment'].unique()}")
        else:
             print("Warning: 'Experiment' column not found.")

    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or performing initial checks on CSV: {e}")
        sys.exit(1)

    # --- Load Graph Data ---
    graph_file_path = os.path.join(script_dir, GRAPH_FILENAME)
    if not os.path.exists(graph_file_path):
         print(f"Info: '{GRAPH_FILENAME}' not found in script directory. Trying current directory...")
         graph_file_path = os.path.join(os.getcwd(), GRAPH_FILENAME)

    # Read graph data (needed for verification and plotting)
    graph, graph_pos = None, None
    if NETWORKX_AVAILABLE:
        if os.path.exists(graph_file_path):
             graph, graph_pos = read_graph_data(graph_file_path, NUM_VERTICES)
             if graph is None:
                 print("Warning: Failed to load graph data. Fitness verification and plotting will be skipped.")
        else:
             print(f"Warning: Graph file '{GRAPH_FILENAME}' not found. Fitness verification and plotting will be skipped.")
    else:
         print("Warning: NetworkX not installed. Fitness verification and plotting will be skipped.")


    # --- Data Verification 1: Solution Balance ---
    print("\n--- Verifying Solution Validity (Length & Balance) ---")
    if 'Solution' not in df.columns:
        print("Warning: 'Solution' column not found. Skipping balance verification.")
    else:
        # Ensure Solution column is string type for verification
        df['Solution'] = df['Solution'].astype(str)
        is_balanced = df['Solution'].apply(verify_solution_balance, args=(NUM_VERTICES, EXPECTED_HALF_COUNT))
        if is_balanced.all():
            print("Verification PASSED: All solutions have correct length and 50/50 balance.")
        else:
            invalid_solutions = df.loc[~is_balanced, ['Experiment', 'Run', 'Solution']].copy()
            print(f"Verification FAILED: Found {len(invalid_solutions)} invalid solutions (wrong length or balance)!")
            print("First 10 invalid solutions:")
            print(invalid_solutions.head(10).to_string())
            save_output(invalid_solutions, 'table_invalid_solutions_balance.csv')
            print("WARNING: Proceeding with analysis, but results for invalid solutions may be unreliable.")

    # --- Data Verification 2: Fitness Genuineness ---
    print("\n--- Verifying Solution Fitness (Genuineness Check) ---")
    if not NETWORKX_AVAILABLE or graph is None:
        print("Skipping fitness verification because NetworkX is not available or graph data failed to load.")
    elif 'Fitness' not in df.columns or 'Solution' not in df.columns:
         print("Skipping fitness verification because 'Fitness' or 'Solution' column is missing.")
    else:
        # Apply the verification function
        # Note: df['Solution'] was already converted to string above
        df['Fitness_Matches'] = df.apply(verify_row_fitness, axis=1, args=(graph,))

        if df['Fitness_Matches'].all():
            print("Verification PASSED: All reported 'Fitness' values match calculated cut sizes.")
        else:
            mismatched_fitness_rows = df[~df['Fitness_Matches']].copy()
            print(f"Verification FAILED: Found {len(mismatched_fitness_rows)} solutions where reported 'Fitness' does NOT match calculated cut size!")

            # Calculate actual fitness for the mismatched rows to include in the report
            mismatched_fitness_rows['Calculated_Fitness'] = mismatched_fitness_rows['Solution'].apply(lambda sol: calculate_actual_fitness(sol, graph))

            # Select relevant columns for the report
            report_cols = ['Experiment', 'Run', 'Fitness', 'Calculated_Fitness', 'Solution']
            # Add parameter columns if they exist, for context
            param_cols = ['Mutation_Size', 'Stopping_Crit']
            for pcol in param_cols:
                if pcol in mismatched_fitness_rows.columns and pcol not in report_cols:
                    report_cols.insert(2, pcol) # Insert params before Fitness

            mismatched_report = mismatched_fitness_rows[report_cols]

            print("First 10 mismatches:")
            print(mismatched_report.head(10).to_string())
            save_output(mismatched_report, 'table_fitness_mismatches.csv')
            print("WARNING: Proceeding with analysis, but discrepancies exist between reported and calculated fitness.")

        # Optionally drop the temporary verification column
        # df = df.drop(columns=['Fitness_Matches'])


    # --- Visualize Best Partition ---
    # Plotting uses the 'graph' and 'graph_pos' loaded earlier
    plot_best_partition(df, graph, graph_pos)


    # --- Suppress FutureWarnings from Seaborn/Pandas ---
    warnings.simplefilter(action='ignore', category=FutureWarning)


    # --- Analysis for Comparison (a): Pass-Based ---
    print("\n--- Analyzing Comparison (a): Pass-Based Experiments (e.g., 10 Runs) ---")
    if 'Experiment' not in df.columns:
         print("Cannot perform Pass-Based analysis: 'Experiment' column missing.")
    else:
        df_pass = df[df['Experiment'].str.contains('_PassBased', na=False)].copy()
        if df_pass.empty:
            print("No Pass-Based data found (rows with '_PassBased' in Experiment name). Skipping Comparison (a).")
        else:
            pass_algos = sorted(df_pass['Experiment'].unique())
            print(f"Pass-Based Algorithms Found: {pass_algos}")
            all_pass_stats = {}
            best_configs_pass = {} # Store best param value for each type based on median fitness

            for algo in pass_algos:
                data_algo = df_pass[df_pass['Experiment'] == algo].copy()
                if data_algo.empty: continue

                group_col = None
                param_col_name = None # To store the actual parameter column name

                # Identify parameter column based on algorithm name
                if algo in ['Simple_ILS_PassBased', 'ILS_Annealing_PassBased', 'ILS_Adaptive_PassBased']:
                    param_col_name = 'Mutation_Size'
                elif algo == 'GLS_PassBased':
                    param_col_name = 'Stopping_Crit'
                # Add other algorithms with parameters here if needed

                if param_col_name and param_col_name in data_algo.columns:
                    group_col = param_col_name
                    # Convert parameter column to numeric for grouping and plotting
                    data_algo[group_col] = pd.to_numeric(data_algo[group_col], errors='coerce')
                    data_algo.dropna(subset=[group_col], inplace=True)
                    # Convert to int if appropriate (e.g., mutation size, stopping crit)
                    try:
                         # Check if all values are integers before converting
                         if data_algo[group_col].apply(lambda x: x == int(x)).all():
                              data_algo[group_col] = data_algo[group_col].astype(int)
                    except Exception:
                         pass # Keep as float if conversion fails or not appropriate
                elif param_col_name:
                     print(f"Warning: Parameter column '{param_col_name}' not found for {algo}. Calculating overall stats.")


                stats = calculate_summary_stats(data_algo, group_col=group_col)
                print(f"\n--- {algo} Summary Statistics ---")
                if not stats.empty:
                    print(stats.to_string(float_format="%.4f"))
                    save_output(stats, f'table_stats_{algo}.csv', float_format='%.4f')
                    all_pass_stats[algo] = stats

                    # Determine best parameter based on lowest median fitness
                    if group_col and 'Fitness_Median' in stats.columns and not stats['Fitness_Median'].isna().all():
                        best_param = stats['Fitness_Median'].idxmin()
                        best_configs_pass[algo] = best_param
                        print(f"Best param for {algo} (by median fitness): {group_col}={best_param}")
                    elif not group_col:
                        best_configs_pass[algo] = None # No parameter to select
                        print(f"{algo} has no parameter column ('{param_col_name}') or data for parameter selection.")
                    else:
                         print(f"Could not determine best param for {algo} (no valid median fitness data).")

                else:
                    print(f"No valid data found for {algo} to calculate statistics.")
                    all_pass_stats[algo] = pd.DataFrame() # Store empty df


            # --- Generate Plots for Parameter Sweeps ---
            # Plot only if stats and relevant columns exist
            if 'Simple_ILS_PassBased' in all_pass_stats and not all_pass_stats['Simple_ILS_PassBased'].empty:
                stats = all_pass_stats['Simple_ILS_PassBased']
                data_plot = df_pass[df_pass['Experiment'] == 'Simple_ILS_PassBased'].copy()
                if 'Mutation_Size' in data_plot.columns and 'Fitness' in data_plot.columns:
                    data_plot['Mutation_Size'] = pd.to_numeric(data_plot['Mutation_Size'], errors='coerce')
                    data_plot.dropna(subset=['Mutation_Size', 'Fitness'], inplace=True)
                    # Ensure integer for plotting if possible
                    try: data_plot['Mutation_Size'] = data_plot['Mutation_Size'].astype(int)
                    except Exception: pass

                    if not data_plot.empty:
                         fig, ax = plt.subplots(figsize=(12, 7))
                         sns.boxplot(data=data_plot, x='Mutation_Size', y='Fitness', ax=ax)
                         ax.set_title('Simple ILS Fitness vs. Mutation Size (Pass-Based)')
                         save_output(fig, 'plot_boxplot_Simple_ILS_PassBased.png', is_plot=True)

                         if 'Fitness_Median' in stats.columns and not stats['Fitness_Median'].isna().all():
                             fig, ax = plt.subplots(figsize=(10, 6))
                             stats['Fitness_Median'].sort_index().plot(kind='line', marker='o', ax=ax)
                             ax.set_title('Simple ILS Median Fitness vs. Mutation Size')
                             ax.set_xlabel('Mutation Size'); ax.set_ylabel('Median Fitness')
                             ax.grid(True); save_output(fig, 'plot_line_Simple_ILS_MedianFit_PassBased.png', is_plot=True)

                         if 'Unchanged_Mean' in stats.columns and not stats['Unchanged_Mean'].isna().all():
                             fig, ax = plt.subplots(figsize=(10, 6))
                             stats['Unchanged_Mean'].sort_index().plot(kind='line', marker='o', ax=ax)
                             ax.set_title('Simple ILS Mean Unchanged Count vs. Mutation Size')
                             ax.set_xlabel('Mutation Size'); ax.set_ylabel('Mean Unchanged Count')
                             ax.grid(True); save_output(fig, 'plot_line_Simple_ILS_Unchanged_PassBased.png', is_plot=True)


            if 'ILS_Annealing_PassBased' in all_pass_stats and not all_pass_stats['ILS_Annealing_PassBased'].empty:
                stats = all_pass_stats['ILS_Annealing_PassBased']
                data_plot = df_pass[df_pass['Experiment'] == 'ILS_Annealing_PassBased'].copy()
                if 'Mutation_Size' in data_plot.columns and 'Fitness' in data_plot.columns:
                    data_plot['Mutation_Size'] = pd.to_numeric(data_plot['Mutation_Size'], errors='coerce')
                    data_plot.dropna(subset=['Mutation_Size', 'Fitness'], inplace=True)
                    try: data_plot['Mutation_Size'] = data_plot['Mutation_Size'].astype(int)
                    except Exception: pass

                    if not data_plot.empty:
                         fig, ax = plt.subplots(figsize=(12, 7))
                         sns.boxplot(data=data_plot, x='Mutation_Size', y='Fitness', ax=ax)
                         ax.set_title('ILS Annealing Fitness vs. Mutation Size (Pass-Based)')
                         save_output(fig, 'plot_boxplot_ILS_Annealing_PassBased.png', is_plot=True)
                         # Could add median/unchanged plots similar to Simple ILS if desired

            if 'GLS_PassBased' in all_pass_stats and not all_pass_stats['GLS_PassBased'].empty:
                stats = all_pass_stats['GLS_PassBased']
                data_plot = df_pass[df_pass['Experiment'] == 'GLS_PassBased'].copy()
                if 'Stopping_Crit' in data_plot.columns and 'Fitness' in data_plot.columns:
                    # Convert Stopping_Crit to string for categorical plotting if it's numeric
                    data_plot['Stopping_Crit'] = pd.to_numeric(data_plot['Stopping_Crit'], errors='coerce')
                    data_plot.dropna(subset=['Stopping_Crit', 'Fitness'], inplace=True)
                    try: data_plot['Stopping_Crit'] = data_plot['Stopping_Crit'].astype(int).astype(str)
                    except Exception: data_plot['Stopping_Crit'] = data_plot['Stopping_Crit'].astype(str) # Fallback to string

                    if not data_plot.empty:
                         fig, ax = plt.subplots(figsize=(8, 6))
                         sns.boxplot(data=data_plot, x='Stopping_Crit', y='Fitness', ax=ax, order=sorted(data_plot['Stopping_Crit'].unique()))
                         ax.set_title('GLS Fitness vs. Stopping Criterion (Pass-Based)')
                         save_output(fig, 'plot_boxplot_GLS_PassBased.png', is_plot=True)


            # --- Overall Comparison Plot and Stats (Pass-Based Best Configs) ---
            print("\n--- Comparing Best Configurations from Pass-Based Runs ---")
            best_configs_data_list = []
            best_configs_names = []

            for algo_exp, best_param in best_configs_pass.items():
                algo_name_short = algo_exp.replace('_PassBased','')
                data_algo_all_runs = df_pass[df_pass['Experiment'] == algo_exp].copy() # Get all runs for this algo

                display_name = algo_name_short
                filtered_data = data_algo_all_runs # Default to all data if no param or param invalid

                if best_param is not None:
                    param_col = 'Mutation_Size' if 'ILS' in algo_exp else ('Stopping_Crit' if 'GLS' in algo_exp else None)

                    if param_col and param_col in data_algo_all_runs.columns:
                        # Ensure data types match for filtering (numeric comparison)
                        data_algo_all_runs[param_col] = pd.to_numeric(data_algo_all_runs[param_col], errors='coerce')
                        # Filter for the best parameter value
                        # Use tolerance for float comparison if necessary, but these params are likely int
                        filtered_data = data_algo_all_runs[data_algo_all_runs[param_col] == best_param]

                        # Format display name with parameter
                        try:
                             param_val_str = str(int(best_param)) # Use int for display if possible
                        except ValueError:
                             param_val_str = f"{best_param:.2f}" # Fallback for float

                        param_abbr = param_col.split('_')[0][0:3].lower() # e.g., 'mut', 'sto'
                        display_name = f"{algo_name_short} ({param_abbr}={param_val_str})"
                    else:
                         print(f"Warning: Could not filter {algo_exp} by best param '{best_param}' - column '{param_col}' missing or invalid.")
                         # Keep all data for this algo in the comparison if filtering fails

                if not filtered_data.empty:
                    # Add an 'Algorithm' column with the display name for plotting/grouping
                    best_configs_data_list.append(filtered_data.assign(Algorithm=display_name))
                    if display_name not in best_configs_names: # Avoid duplicates if filtering failed
                        best_configs_names.append(display_name)
                else:
                    print(f"Warning: No data found for {algo_exp} with best param {best_param}. Excluding from comparison.")


            if len(best_configs_data_list) > 1:
                df_best_pass = pd.concat(best_configs_data_list, ignore_index=True)

                # Plotting
                fig, ax = plt.subplots(figsize=(max(10, 1.5*len(best_configs_names)), 7))
                # Sort names for consistent plot order
                plot_order = sorted(best_configs_names)
                sns.boxplot(data=df_best_pass, x='Algorithm', y='Fitness', order=plot_order, ax=ax, palette="tab10")
                sns.stripplot(data=df_best_pass, x='Algorithm', y='Fitness', order=plot_order, ax=ax, color='black', size=4, jitter=0.1)
                ax.set_title('Comparison of Best Algorithm Configurations (Pass-Based)')
                ax.set_xlabel('Algorithm Configuration'); ax.set_ylabel('Fitness (Cut Size)')
                plt.xticks(rotation=25, ha='right'); plt.tight_layout(); # Adjust layout
                save_output(fig, 'plot_boxplot_BestConfigs_PassBased.png', is_plot=True)

                # Pairwise Statistical Tests
                print("\nPairwise Statistical Tests for Best Pass-Based Configurations (Mann-Whitney U):")
                test_results_best_pass = []
                pairs_tested = set()
                for i in range(len(plot_order)):
                    for j in range(i + 1, len(plot_order)):
                        name1 = plot_order[i]; name2 = plot_order[j]
                        pair = tuple(sorted((name1, name2))) # Ensure consistent pair naming
                        if pair in pairs_tested: continue
                        pairs_tested.add(pair)

                        data1 = df_best_pass[df_best_pass['Algorithm'] == name1]['Fitness']
                        data2 = df_best_pass[df_best_pass['Algorithm'] == name2]['Fitness']

                        p_val, result = run_mannwhitneyu_test(data1, data2, name1, name2, ALPHA)
                        test_results_best_pass.append({'Comparison': f"{name1} vs {name2}", 'P-Value': p_val, 'Result': result})

                if test_results_best_pass:
                    tests_df_pass = pd.DataFrame(test_results_best_pass)
                    print(tests_df_pass.to_string(index=False))
                    save_output(tests_df_pass, 'table_tests_BestConfigs_PassBased.csv', float_format='%.4f')
            elif len(best_configs_data_list) == 1:
                 print("Only one valid algorithm configuration found. Cannot perform pairwise comparison.")
            else:
                print("Not enough data/configurations for overall pass-based comparison plot/tests.")


    # --- Analysis for Comparison (b): Time-Based ---
    print("\n\n--- Analyzing Comparison (b): Runtime Experiments (e.g., 25 Runs) ---")
    if 'Experiment' not in df.columns:
         print("Cannot perform Runtime analysis: 'Experiment' column missing.")
    else:
        df_runtime = df[df['Experiment'].str.contains('_Runtime', na=False)].copy()

        if df_runtime.empty:
            print("No Runtime data found (rows with '_Runtime' in Experiment name). Skipping Comparison (b).")
        else:
            # Check run counts
            run_counts = df_runtime.groupby('Experiment')['Run'].nunique()
            print(f"\nRuntime experiment runs per algorithm:")
            print(run_counts.to_string())
            expected_runtime_runs = RUNTIME_RUNS # Use config variable
            if (run_counts != expected_runtime_runs).any():
                print(f"Warning: Not all runtime experiments have exactly {expected_runtime_runs} runs!")

            # --- Q5: Runtime Comparison (All Algos) ---
            print("\n--- Runtime Comparison (All Algorithms) ---")

            # Use parameters determined from pass-based runs (best_configs_pass) for detailed names
            # Provide defaults if pass-based analysis didn't run or find a best param
            best_params_runtime_map = {
                'Simple_ILS_PassBased': best_configs_pass.get('Simple_ILS_PassBased', None),
                'ILS_Annealing_PassBased': best_configs_pass.get('ILS_Annealing_PassBased', None),
                'GLS_PassBased': best_configs_pass.get('GLS_PassBased', None),
                'ILS_Adaptive_PassBased': best_configs_pass.get('ILS_Adaptive_PassBased', None), # Added Adaptive
                # Add other parameterized algos here if needed
            }

            def format_runtime_algo_name_detailed(row):
                """ Creates a detailed name for runtime algorithms using best params found earlier. """
                base_name_pass = row['Experiment'].replace('_Runtime', '_PassBased')
                name_short = row['Experiment'].replace('_Runtime', '')
                best_param = best_params_runtime_map.get(base_name_pass, None)

                if best_param is not None:
                    param_col = None
                    if 'ILS' in name_short: param_col = 'Mutation_Size'
                    elif 'GLS' in name_short: param_col = 'Stopping_Crit'

                    if param_col:
                         try: param_val_str = str(int(best_param))
                         except ValueError: param_val_str = f"{best_param:.1f}" # Float fallback
                         param_abbr = param_col.split('_')[0][0:3].lower()
                         return f"{name_short} ({param_abbr}={param_val_str})"
                    else:
                         # Handle cases like Adaptive ILS if param name logic needs adjustment
                         return f"{name_short} (param={best_param})" # Generic if col name unknown
                else:
                    # If no best param was found or algo wasn't in pass-based
                    return name_short # MLS, or others without params/results

            # Apply the detailed naming function
            df_runtime['Algorithm_Detailed'] = df_runtime.apply(format_runtime_algo_name_detailed, axis=1)

            # Calculate summary stats grouped by the new detailed names
            runtime_stats = calculate_summary_stats(df_runtime, group_col='Algorithm_Detailed')
            print("\nSummary Statistics for Runtime Experiments:")
            if not runtime_stats.empty:
                print(runtime_stats.to_string(float_format="%.4f"))
                save_output(runtime_stats, 'table_stats_Runtime.csv', float_format='%.4f')

                # --- Comparative Box Plot for Runtime ---
                fig, ax = plt.subplots(figsize=(max(12, 1.5*len(runtime_stats)), 8))
                algo_order_runtime = sorted(df_runtime['Algorithm_Detailed'].unique()) # Use detailed names
                sns.boxplot(data=df_runtime, x='Algorithm_Detailed', y='Fitness', order=algo_order_runtime, ax=ax, palette="Set2")
                sns.stripplot(data=df_runtime, x='Algorithm_Detailed', y='Fitness', order=algo_order_runtime, ax=ax, color='black', size=3, jitter=0.1)

                # Attempt to get median time for title, handle potential errors
                median_time_limit = np.nan
                if 'Comp_Time' in df_runtime.columns:
                     median_time_limit = pd.to_numeric(df_runtime['Comp_Time'], errors='coerce').median()

                title = f'Algorithm Fitness Comparison (Fixed Runtime ≈ {median_time_limit:.1f}s, {expected_runtime_runs} Runs)' if pd.notna(median_time_limit) else \
                        f'Algorithm Fitness Comparison (Fixed Runtime, {expected_runtime_runs} Runs)'
                ax.set_title(title)
                ax.set_xlabel('Algorithm Configuration'); ax.set_ylabel('Fitness (Cut Size)')
                plt.xticks(rotation=30, ha='right'); plt.tight_layout()
                save_output(fig, 'plot_boxplot_Comparison_Runtime.png', is_plot=True)

                # --- Statistical Tests for Runtime ---
                print("\nRuntime Statistical Tests (Mann-Whitney U):")
                runtime_algos_detailed = algo_order_runtime # Use sorted order from plot
                test_results_runtime = []
                pairs_tested_runtime = set()
                for i in range(len(runtime_algos_detailed)):
                    for j in range(i + 1, len(runtime_algos_detailed)):
                        name1 = runtime_algos_detailed[i]; name2 = runtime_algos_detailed[j]
                        pair = tuple(sorted((name1, name2)))
                        if pair in pairs_tested_runtime: continue
                        pairs_tested_runtime.add(pair)

                        data1 = df_runtime[df_runtime['Algorithm_Detailed'] == name1]['Fitness']
                        data2 = df_runtime[df_runtime['Algorithm_Detailed'] == name2]['Fitness']

                        p_val, result = run_mannwhitneyu_test(data1, data2, name1, name2, ALPHA)
                        test_results_runtime.append({'Comparison': f"{name1} vs {name2}", 'P-Value': p_val, 'Result': result})

                if test_results_runtime:
                    tests_df_runtime = pd.DataFrame(test_results_runtime)
                    print(tests_df_runtime.to_string(index=False))
                    save_output(tests_df_runtime, 'table_tests_Comparison_Runtime.csv', float_format='%.4f')
            else:
                 print("No statistics calculated for runtime experiments.")


    # --- List Generated Files ---
    print("\n--- Generated Output Files ---")
    if generated_files:
        print(f"Files saved in directory: '{os.path.abspath(OUTPUT_DIR)}'")
        for f in sorted(generated_files):
            print(f"- {os.path.basename(f)}") # Print only filename
    else:
        print("No output files were generated (or saving failed).")

    print("\n--- Analysis Script Finished ---")
