import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import os
import warnings

# Attempt to import networkx, provide instructions if missing
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: networkx library not found. Graph visualization will be skipped.")
    print("Install it using: pip install networkx")


# --- Configuration ---
CSV_FILENAME = "experiment_results_combined.csv" # Input CSV file name
GRAPH_FILENAME = "Graph500.txt" # Graph file needed for visualization
OUTPUT_DIR = "analysis_results" # Directory to save plots and tables
ALPHA = 0.05 # Significance level for statistical tests
NUM_VERTICES = 500 # Define the expected number of vertices for verification
EXPECTED_HALF_COUNT = NUM_VERTICES // 2

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
            grouped_indices = data.groupby(group_col).groups
    else:
        grouped_indices = {'all_data': data.index}

    stats_list = []
    for name, idx in grouped_indices.items():
        group_data = data.loc[idx]
        if group_data.empty:
            continue

        fitness_data = pd.to_numeric(group_data['Fitness'], errors='coerce').dropna()
        if fitness_data.empty:
            stats = {'Group': name, 'Count': 0}
        else:
            best_fitness = fitness_data.min()
            count_total = fitness_data.count()
            count_best = (fitness_data == best_fitness).sum()
            stats = {
                'Group': name,
                'Count': count_total,
                'Fitness_Min': best_fitness,
                'Fitness_Median': fitness_data.median(),
                'Fitness_Mean': fitness_data.mean(),
                'Fitness_StdDev': fitness_data.std(),
                'Fitness_Max': fitness_data.max(),
                'Best_Fitness_Count': f"{count_best}/{count_total}" if count_total > 0 else "0/0"
            }

        if 'Comp_Time' in group_data.columns:
            time_data = pd.to_numeric(group_data['Comp_Time'], errors='coerce').dropna()
            stats['Time_Mean'] = time_data.mean() if not time_data.empty else np.nan
            stats['Time_StdDev'] = time_data.std() if not time_data.empty else np.nan

        if 'Actual_Passes' in group_data.columns:
            passes_data = pd.to_numeric(group_data['Actual_Passes'], errors='coerce').dropna()
            stats['Passes_Mean'] = passes_data.mean() if not passes_data.empty else np.nan
            stats['Passes_StdDev'] = passes_data.std() if not passes_data.empty else np.nan

        stats_list.append(stats)

    if not stats_list: return pd.DataFrame()

    stats_df = pd.DataFrame(stats_list)
    cols_order = [
        'Group', 'Count', 'Fitness_Min', 'Fitness_Median', 'Fitness_Mean', 'Fitness_StdDev',
        'Fitness_Max', 'Best_Fitness_Count', 'Time_Mean', 'Time_StdDev', 'Passes_Mean', 'Passes_StdDev'
    ]
    cols_order = [col for col in cols_order if col in stats_df.columns]
    stats_df = stats_df[cols_order]

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
    data1_clean = pd.to_numeric(data1, errors='coerce').dropna()
    data2_clean = pd.to_numeric(data2, errors='coerce').dropna()

    if data1_clean.empty or data2_clean.empty:
        print(f"  Skipping test between {name1} and {name2} due to empty/non-numeric data.")
        return None, "Skipped (Empty/Non-Numeric Data)"

    try:
        stat_less, p_value_less = mannwhitneyu(data1_clean, data2_clean, alternative='less')
        stat_greater, p_value_greater = mannwhitneyu(data1_clean, data2_clean, alternative='greater')

        print(f"  Mann-Whitney U Test: {name1} vs {name2}")
        print(f"    U-statistic: {stat_less:.4f}")
        print(f"    P-value ({name1} < {name2}): {p_value_less:.4f}")
        print(f"    P-value ({name1} > {name2}): {p_value_greater:.4f}")

        if p_value_less < alpha:
            result = f"Significant ({name1} better than {name2}, p={p_value_less:.4f})"
            print(f"    Result: {result}")
            return p_value_less, result
        elif p_value_greater < alpha:
            result = f"Significant ({name2} better than {name1}, p={p_value_greater:.4f})"
            print(f"    Result: {result}")
            return p_value_greater, result
        else:
            stat_two_sided, p_value_two_sided = mannwhitneyu(data1_clean, data2_clean, alternative='two-sided')
            result = f"Not Significant (p={p_value_two_sided:.4f})"
            print(f"    Result: {result}")
            return p_value_two_sided, result
    except ValueError as e:
        print(f"  Skipping test between {name1} and {name2} due to error: {e}")
        if data1_clean.equals(data2_clean): return 1.0, "Not Significant (Identical Data)"
        else: return None, f"Skipped (Error: {e})"
    except Exception as e:
        print(f"  An unexpected error occurred during Mann-Whitney U test for {name1} vs {name2}: {e}")
        return None, f"Skipped (Unexpected Error: {e})"


def save_output(df_or_fig, filename, is_plot=False, float_format='%.4f'):
    """Saves DataFrame to CSV or figure to PNG in the output directory."""
    global generated_files
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        if is_plot:
            df_or_fig.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close(df_or_fig) # Close the figure
            print(f"Plot saved: {filepath}")
        else:
            df_or_fig.to_csv(filepath, float_format=float_format)
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

def read_graph_data_for_plotting(filename):
    """Reads graph data, extracting adjacency list and coordinates for plotting."""
    adj = {i: set() for i in range(NUM_VERTICES)}
    pos = {} # Dictionary to store node positions {node_id: (x, y)}
    print(f"Reading graph data for plotting from: {filename}")
    try:
        with open(filename, "r") as f:
            ln = 0
            for line in f:
                ln += 1
                line = line.strip()
                if not line: continue
                parts = line.split()
                if not parts: continue

                try:
                    vertex_id = int(parts[0]) - 1 # 0-based index
                    if not (0 <= vertex_id < NUM_VERTICES): continue

                    # Extract coordinates (assuming format like '(x,y)')
                    coord_str = parts[1]
                    if coord_str.startswith('(') and coord_str.endswith(')'):
                        try:
                            x_str, y_str = coord_str[1:-1].split(',')
                            pos[vertex_id] = (float(x_str), float(y_str))
                        except ValueError:
                             print(f"Warning: Could not parse coordinates '{coord_str}' on line {ln}. Skipping position.")
                             pos[vertex_id] = (np.random.rand(), np.random.rand()) # Assign random position as fallback


                    # Find neighbors (using same logic as experiment script)
                    num_neighbors_idx = -1
                    for idx, part in enumerate(parts):
                         if idx > 0 and ')' in parts[idx-1] and part.isdigit(): num_neighbors_idx = idx; break
                    if num_neighbors_idx == -1:
                         for idx, part in enumerate(parts):
                              if idx > 0 and part.isdigit(): num_neighbors_idx = idx; break
                    if num_neighbors_idx == -1 or num_neighbors_idx + 1 >= len(parts): continue

                    neighbor_ids_str = parts[num_neighbors_idx + 1:]
                    connected_vertices = [int(n) - 1 for n in neighbor_ids_str]
                    valid_neighbors = {nb for nb in connected_vertices if 0 <= nb < NUM_VERTICES}
                    adj[vertex_id].update(valid_neighbors)

                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping line {ln} during plot data reading due to parsing error: {e}.")
                    continue

        # Ensure symmetry
        for i in range(NUM_VERTICES):
            for neighbor in list(adj[i]):
                if 0 <= neighbor < NUM_VERTICES:
                    if i not in adj[neighbor]: adj[neighbor].add(i)
                else: adj[i].discard(neighbor) # Remove invalid neighbor index

        # Create NetworkX graph
        G = nx.Graph()
        for i in range(NUM_VERTICES):
             G.add_node(i, pos=pos.get(i)) # Add node with position attribute if available
             for neighbor in adj[i]:
                  if i < neighbor: # Avoid adding edges twice
                       G.add_edge(i, neighbor)

        print("Graph data loaded for plotting.")
        return G, pos

    except FileNotFoundError:
        print(f"Error: Graph file not found at {filename} for plotting.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during graph reading for plotting: {e}")
        return None, None

def plot_best_partition(df, graph_file):
    """Finds the best overall solution and plots the graph partition."""
    if not NETWORKX_AVAILABLE:
        print("Skipping graph visualization as networkx is not available.")
        return

    if 'Fitness' not in df.columns or 'Solution' not in df.columns:
         print("Warning: Cannot plot partition. 'Fitness' or 'Solution' column missing.")
         return

    try:
        # Find the row with the absolute minimum fitness
        best_row = df.loc[pd.to_numeric(df['Fitness'], errors='coerce').idxmin()]
        best_fitness = best_row['Fitness']
        best_solution_str = best_row['Solution']
        best_experiment = best_row['Experiment']
        print(f"\n--- Visualizing Best Overall Partition ---")
        print(f"Best solution found by '{best_experiment}' with fitness {best_fitness}")

        if not isinstance(best_solution_str, str) or len(best_solution_str) != NUM_VERTICES:
            print("Error: Best solution string is invalid. Cannot visualize.")
            return

        # Read graph data including positions
        G, pos = read_graph_data_for_plotting(graph_file)
        if G is None:
             print("Error reading graph data for visualization.")
             return

        # Assign colors based on the partition
        colors = ['skyblue' if bit == '0' else 'lightcoral' for bit in best_solution_str]

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 12))
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=50, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax)

        ax.set_title(f"Best Graph Partition Found (Fitness = {best_fitness})\nAlgorithm: {best_experiment}")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis('off')
        save_output(fig, 'plot_best_partition_visualization.png', is_plot=True)

    except Exception as e:
        print(f"Error during graph visualization: {e}")


# --- Main Analysis Script ---
if __name__ == "__main__":
    print("--- Starting Analysis Script ---")

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        try: os.makedirs(OUTPUT_DIR); print(f"Created output directory: {OUTPUT_DIR}")
        except OSError as e: print(f"Error creating output directory {OUTPUT_DIR}: {e}"); exit()

    # Load data
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, CSV_FILENAME)
        if not os.path.exists(csv_path):
            print(f"Warning: CSV not found in script directory, trying CWD: {os.getcwd()}")
            csv_path = CSV_FILENAME
            if not os.path.exists(csv_path): raise FileNotFoundError(f"Error: CSV file '{CSV_FILENAME}' not found.")
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data from: {csv_path} ({len(df)} rows)")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Experiment types: {df['Experiment'].unique()}")
    except Exception as e: print(f"Error loading CSV: {e}"); exit()

    # --- Data Verification ---
    print("\n--- Verifying Solution Validity (Length & Balance) ---")
    if 'Solution' not in df.columns:
        print("Warning: 'Solution' column not found. Cannot verify.")
    else:
        df['Solution'] = df['Solution'].astype(str)
        is_valid = df['Solution'].apply(verify_solution_balance, args=(NUM_VERTICES, EXPECTED_HALF_COUNT))
        if is_valid.all():
            print("Verification PASSED: All solutions have correct length and 50/50 balance.")
        else:
            invalid_solutions = df[~is_valid]
            print(f"Verification FAILED: Found {len(invalid_solutions)} invalid solutions!")
            print(invalid_solutions[['Experiment', 'Run', 'Solution']].head(10).to_string())
            save_output(invalid_solutions, 'table_invalid_solutions.csv')
            print("WARNING: Proceeding with analysis, results may be affected.")
            # exit() # Optional: Stop if invalid solutions found

    # --- Visualize Best Partition ---
    graph_file_path = os.path.join(script_dir, GRAPH_FILENAME)
    if not os.path.exists(graph_file_path): graph_file_path = GRAPH_FILENAME # Fallback to CWD
    if os.path.exists(graph_file_path):
         plot_best_partition(df, graph_file_path)
    else:
         print(f"Warning: Graph file '{GRAPH_FILENAME}' not found. Skipping partition visualization.")

    # Suppress future warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # --- Analysis for Comparison (a): Pass-Based ---
    print("\n--- Analyzing Comparison (a): Pass-Based Experiments (10 Runs) ---")
    df_pass = df[df['Experiment'].str.contains('_PassBased', na=False)].copy()

    if df_pass.empty:
        print("No Pass-Based data found. Skipping Comparison (a).")
    else:
        pass_algos = sorted(df_pass['Experiment'].unique())
        print(f"Pass-Based Algorithms: {pass_algos}")

        # --- Calculate Stats for Each Algorithm/Parameter ---
        all_pass_stats = {}
        for algo in pass_algos:
            data = df_pass[df_pass['Experiment'] == algo]
            if algo in ['Simple_ILS_PassBased', 'ILS_Annealing_PassBased']:
                data['Mutation_Size'] = pd.to_numeric(data['Mutation_Size'], errors='coerce')
                data.dropna(subset=['Mutation_Size'], inplace=True)
                data['Mutation_Size'] = data['Mutation_Size'].astype(int)
                stats = calculate_summary_stats(data, group_col='Mutation_Size')
                print(f"\n--- {algo} Summary Statistics by Mutation Size ---")
                print(stats.to_string(float_format="%.4f"))
                save_output(stats, f'table_stats_{algo}.csv', float_format='%.4f')
                all_pass_stats[algo] = stats # Store grouped stats
            elif algo == 'GLS_PassBased':
                data['Stopping_Crit_Str'] = data['Stopping_Crit'].fillna('None').astype(str)
                stats = calculate_summary_stats(data, group_col='Stopping_Crit_Str')
                print(f"\n--- {algo} Summary Statistics by Stopping Criterion ---")
                print(stats.to_string(float_format="%.4f"))
                save_output(stats, f'table_stats_{algo}.csv', float_format='%.4f')
                all_pass_stats[algo] = stats # Store grouped stats
            else: # MLS, ILS_Adaptive
                stats = calculate_summary_stats(data)
                print(f"\n--- {algo} Summary Statistics ---")
                print(stats.to_string(float_format="%.4f"))
                save_output(stats, f'table_stats_{algo}.csv', float_format='%.4f')
                all_pass_stats[algo] = stats # Store overall stats


        # --- Identify Best Configurations ---
        best_configs_details = {} # Store name, data, and specific param value

        # MLS
        if 'MLS_PassBased' in all_pass_stats:
            best_configs_details['MLS'] = {'data': df_pass[df_pass['Experiment'] == 'MLS_PassBased']['Fitness'], 'param': None}

        # Simple ILS
        if 'Simple_ILS_PassBased' in all_pass_stats:
            stats = all_pass_stats['Simple_ILS_PassBased']
            if not stats.empty and 'Fitness_Median' in stats.columns:
                best_param = stats['Fitness_Median'].idxmin()
                print(f"Best Simple ILS Mutation Size (by median): {best_param}")
                data = df_pass[(df_pass['Experiment'] == 'Simple_ILS_PassBased') & (df_pass['Mutation_Size'] == best_param)]['Fitness']
                best_configs_details[f'ILS Simple (mut={best_param})'] = {'data': data, 'param': best_param}

        # GLS
        if 'GLS_PassBased' in all_pass_stats:
             stats = all_pass_stats['GLS_PassBased']
             if not stats.empty and 'Fitness_Median' in stats.columns:
                  best_param_str = stats['Fitness_Median'].idxmin()
                  print(f"Best GLS Stopping Crit (by median): {best_param_str}")
                  data = df_pass[(df_pass['Experiment'] == 'GLS_PassBased') & (df_pass['Stopping_Crit_Str'] == best_param_str)]['Fitness']
                  # Find original crit value (might be None)
                  original_crit = df_pass.loc[data.index, 'Stopping_Crit'].iloc[0]
                  best_configs_details[f'GLS (crit={best_param_str})'] = {'data': data, 'param': original_crit}


        # ILS Annealing
        if 'ILS_Annealing_PassBased' in all_pass_stats:
            stats = all_pass_stats['ILS_Annealing_PassBased']
            if not stats.empty and 'Fitness_Median' in stats.columns:
                best_param = stats['Fitness_Median'].idxmin()
                print(f"Best ILS Annealing Mutation Size (by median): {best_param}")
                data = df_pass[(df_pass['Experiment'] == 'ILS_Annealing_PassBased') & (df_pass['Mutation_Size'] == best_param)]['Fitness']
                best_configs_details[f'ILS Annealing (mut={best_param})'] = {'data': data, 'param': best_param}

        # ILS Adaptive
        if 'ILS_Adaptive_PassBased' in all_pass_stats:
            best_configs_details['ILS Adaptive'] = {'data': df_pass[df_pass['Experiment'] == 'ILS_Adaptive_PassBased']['Fitness'], 'param': None}

        # --- Generate Plots and Comparisons for Best Configs ---
        if len(best_configs_details) > 1:
            print("\n--- Overall Comparison (a): Best Configurations (Pass-Based) ---")
            best_configs_data_list = []
            best_configs_names = list(best_configs_details.keys())
            for name, details in best_configs_details.items():
                 # Re-fetch the full data rows for the best config to use calculate_summary_stats
                 exp_name = name.split(' ')[0] + '_PassBased' # Reconstruct experiment name
                 param_col = None
                 param_val = details['param']
                 if 'mut=' in name: param_col = 'Mutation_Size'
                 elif 'crit=' in name: param_col = 'Stopping_Crit_Str'; param_val=str(param_val) if param_val is not None else 'None' # Use string version for filtering

                 if param_col:
                      full_data = df_pass[(df_pass['Experiment'] == exp_name) & (df_pass[param_col] == param_val)]
                 else: # For MLS, ILS_Adaptive
                      full_data = df_pass[df_pass['Experiment'] == exp_name]

                 best_configs_data_list.append(full_data.assign(Algorithm=name)) # Assign the descriptive name

            df_best_pass = pd.concat(best_configs_data_list, ignore_index=True)

            # Combined Stats Table
            best_pass_stats = calculate_summary_stats(df_best_pass, group_col='Algorithm')
            best_pass_stats = best_pass_stats.reindex(best_configs_names) # Ensure consistent order
            print("\nSummary Statistics for Best Pass-Based Configurations:")
            print(best_pass_stats.to_string(float_format="%.4f"))
            save_output(best_pass_stats, 'table_stats_BestConfigs_PassBased.csv', float_format='%.4f')

            # Combined Box Plot
            fig, ax = plt.subplots(figsize=(max(10, 1.5*len(best_configs_names)), 7))
            sns.boxplot(data=df_best_pass, x='Algorithm', y='Fitness', order=best_configs_names, ax=ax, palette="tab10")
            sns.stripplot(data=df_best_pass, x='Algorithm', y='Fitness', order=best_configs_names, ax=ax, color='black', size=4, jitter=0.1)
            ax.set_title('Comparison of Best Algorithm Configurations (Pass-Based, 10 Runs)')
            ax.set_xlabel('Algorithm Configuration')
            ax.set_ylabel('Fitness (Cut Size)')
            plt.xticks(rotation=20, ha='right')
            save_output(fig, 'plot_boxplot_BestConfigs_PassBased.png', is_plot=True)

            # Pairwise Statistical Tests
            print("\nPairwise Statistical Tests for Best Pass-Based Configurations (Mann-Whitney U):")
            test_results_best_pass = []
            pairs_tested = set()
            for i in range(len(best_configs_names)):
                for j in range(i + 1, len(best_configs_names)):
                    name1 = best_configs_names[i]
                    name2 = best_configs_names[j]
                    pair = tuple(sorted((name1, name2)))
                    if pair in pairs_tested: continue
                    pairs_tested.add(pair)
                    data1 = best_configs_details[name1]['data']
                    data2 = best_configs_details[name2]['data']
                    p_val, result = run_mannwhitneyu_test(data1, data2, name1, name2, ALPHA)
                    test_results_best_pass.append({'Comparison': f"{name1} vs {name2}", 'P-Value': p_val, 'Result': result})
            if test_results_best_pass:
                save_output(pd.DataFrame(test_results_best_pass), 'table_tests_BestConfigs_PassBased.csv', float_format='%.4f')
        else:
            print("Not enough data for overall pass-based comparison.")


    # --- Analysis for Comparison (b): Time-Based ---
    print("\n\n--- Analyzing Comparison (b): Runtime Experiments (25 Runs) ---")
    df_runtime = df[df['Experiment'].str.contains('_Runtime', na=False)].copy()

    if df_runtime.empty:
        print("No Runtime data found. Skipping Comparison (b).")
    else:
        run_counts = df_runtime.groupby('Experiment')['Run'].nunique()
        print(f"\nRuntime experiment runs per algorithm (expected 25): \n{run_counts.to_string()}")
        if (run_counts != 25).any(): print("Warning: Not all runtime experiments have exactly 25 runs!")

        # --- Q5: Runtime Comparison (All Algos) ---
        print("\n--- Q5: Runtime Comparison (All Algorithms) ---")
        df_runtime['Algorithm'] = df_runtime['Experiment'].str.replace('_Runtime', '', regex=False) # Clean names for plot/table
        # Add specific parameters to names for clarity
        def format_runtime_algo_name(row):
            name = row['Algorithm']
            if name == 'Simple_ILS' and 'Mutation_Size' in row and pd.notna(row['Mutation_Size']): return f"ILS Simple (mut={int(row['Mutation_Size'])})"
            if name == 'ILS_Annealing' and 'Mutation_Size' in row and pd.notna(row['Mutation_Size']): return f"ILS Annealing (mut={int(row['Mutation_Size'])})"
            if name == 'GLS' and 'Stopping_Crit' in row: return f"GLS (crit={str(row['Stopping_Crit']) if pd.notna(row['Stopping_Crit']) else 'None'})"
            return name # MLS, ILS_Adaptive
        df_runtime['Algorithm_Detailed'] = df_runtime.apply(format_runtime_algo_name, axis=1)


        runtime_stats = calculate_summary_stats(df_runtime, group_col='Algorithm_Detailed')
        print("\nSummary Statistics for Runtime Experiments:")
        print(runtime_stats.to_string(float_format="%.4f"))
        save_output(runtime_stats, 'table_stats_Runtime.csv', float_format='%.4f')

        # Comparative Box Plot for Runtime
        fig, ax = plt.subplots(figsize=(max(12, 1.5*len(runtime_stats)), 8))
        algo_order_runtime = sorted(df_runtime['Algorithm_Detailed'].unique())
        sns.boxplot(data=df_runtime, x='Algorithm_Detailed', y='Fitness', order=algo_order_runtime, ax=ax, palette="Set2")
        sns.stripplot(data=df_runtime, x='Algorithm_Detailed', y='Fitness', order=algo_order_runtime, ax=ax, color='black', size=3, jitter=0.1)
        ax.set_title(f'Algorithm Fitness Comparison (Fixed Runtime ≈ {df_runtime["Comp_Time"].median():.1f}s, 25 Runs)')
        ax.set_xlabel('Algorithm Configuration')
        ax.set_ylabel('Fitness (Cut Size)')
        plt.xticks(rotation=30, ha='right')
        save_output(fig, 'plot_boxplot_Q5_Comparison_Runtime.png', is_plot=True)

        # Statistical Tests for Runtime
        print("\nRuntime Statistical Tests (Mann-Whitney U):")
        runtime_algos_detailed = df_runtime['Algorithm_Detailed'].unique()
        test_results_runtime = []
        pairs_tested_runtime = set()
        for i in range(len(runtime_algos_detailed)):
            for j in range(i + 1, len(runtime_algos_detailed)):
                name1 = runtime_algos_detailed[i]
                name2 = runtime_algos_detailed[j]
                pair = tuple(sorted((name1, name2)))
                if pair in pairs_tested_runtime: continue
                pairs_tested_runtime.add(pair)
                data1 = df_runtime[df_runtime['Algorithm_Detailed'] == name1]['Fitness']
                data2 = df_runtime[df_runtime['Algorithm_Detailed'] == name2]['Fitness']
                p_val, result = run_mannwhitneyu_test(data1, data2, name1, name2, ALPHA)
                test_results_runtime.append({'Comparison': f"{name1} vs {name2}", 'P-Value': p_val, 'Result': result})
        if test_results_runtime:
            save_output(pd.DataFrame(test_results_runtime), 'table_tests_Q5_Runtime.csv', float_format='%.4f')

    # --- List Generated Files ---
    print("\n--- Generated Output Files ---")
    if generated_files:
        for f in sorted(generated_files):
            print(f"- {f}")
    else:
        print("No output files were generated (or saving failed).")

    print("\n--- Analysis Script Finished ---")
