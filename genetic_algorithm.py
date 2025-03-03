"""
geneticAlgorithm.py

This module contains a genetic algorithm implementation with detailed tracing for the
'Counting Ones problem'.
It also includes functions to run experiments on various fitness functions and crossover operators,
and to find the minimum population size.

Author: Daan Westland and Julius Bijkerk
Date: March 3rd 2025

Usage:
    - Run the main script to execute all experiments and save results in the 'results' folder.
    - The module can also be imported and functions can be called directly.
    - The module contains functions for running experiments, summarizing results,
    and finding the minimum population size.
    - The module also contains functions for running the genetic algorithm with and without tracing, 
    and for computing fitness functions and crossover operators.
"""

import os
import random
import time
import statistics
import matplotlib.pyplot as plt

# ----------------------------
# Global Parameters
# ----------------------------
STRING_LENGTH = 40  # bit-string length
K = 4  # subfunction length (always 4)
# For trap functions, d is set via lambda:
# 1 = deceptive, 2.5 = non-deceptive


# ----------------------------
# Fitness Functions
# ----------------------------
def counting_ones(solution):
    """Counting Ones: returns the sum of ones in the binary string."""
    return sum(int(bit) for bit in solution)


def trap_function_tight(solution, d):
    """
    Tightly linked trap function.
    For each block of K bits:
      - if block is all ones, fitness = K;
      - otherwise, fitness = K - d - ((K - d)/(K - 1)) * (#n of ones in block)

    Args:
    - solution: a binary string (e.g., '0101010101').
    - d: the deception value (1 for deceptive, 2.5 for non-deceptive).

    Returns:
    - The fitness of the solution.
    """
    total = 0
    for i in range(0, STRING_LENGTH, K):
        block = solution[i : i + K]
        ones = counting_ones(block)
        total += K if ones == K else K - d - ((K - d) / (K - 1)) * ones
    return total


def trap_function_spread(solution, d):
    """
    Not tightly linked trap function.
    For each of 10 subfunctions, use positions:
    j, j+10, j+20, j+30 (for j=0,...,9).

    Args:
    - solution: a binary string (e.g., '0101010101').
    - d: the deception value (1 for deceptive, 2.5 for non-deceptive).

    Returns:
    - The fitness of the solution.
    """
    total = 0
    for j in range(STRING_LENGTH // K):
        block = solution[j] + solution[j + 10] + solution[j + 20] + solution[j + 30]
        ones = counting_ones(block)
        total += K if ones == K else K - d - ((K - d) / (K - 1)) * ones
    return total


# ----------------------------
# Crossover Operators
# ----------------------------
def uniform_crossover(parent1, parent2):
    """
    Uniform crossover (UX): each bit is chosen randomly from one of the parents

    Args:
    - parent1: a binary string (e.g., '0101010101').
    - parent2: a binary string (e.g., '1010101010').

    Returns:
    - A list of four strings: [parent1, parent2, child1, child
    """

    # Child 1
    child1 = "".join(
        parent1[i] if random.random() < 0.5 else parent2[i]
        for i in range(STRING_LENGTH)
    )

    # Child 2
    child2 = "".join(
        parent1[i] if random.random() < 0.5 else parent2[i]
        for i in range(STRING_LENGTH)
    )

    return [parent1, parent2, child1, child2]


def two_point_crossover(parent1, parent2):
    """
    Two-point crossover (2X): two crossover points are chosen to swap segments.

    Args:
    - parent1: a binary string (e.g., '0101010101').
    - parent2: a binary string (e.g., '1010101010').

    Returns:
    - A list of four strings: [parent1, parent2, child1, child
    """

    # Child 1
    p1, p2 = random.randint(0, STRING_LENGTH - 1), random.randint(0, STRING_LENGTH - 1)
    start, end = min(p1, p2), max(p1, p2)
    child1 = parent1[:start] + parent2[start:end] + parent1[end:]

    # Child 2 (with new random points)
    p1, p2 = random.randint(0, STRING_LENGTH - 1), random.randint(0, STRING_LENGTH - 1)
    start, end = min(p1, p2), max(p1, p2)
    child2 = parent1[:start] + parent2[start:end] + parent1[end:]

    return [parent1, parent2, child1, child2]


# ----------------------------
# Population Utilities
# ----------------------------
def generate_population(pop_size):
    """Generates a list of random bit-string individuals."""
    return [
        "".join(str(random.randint(0, 1)) for _ in range(STRING_LENGTH))
        for _ in range(pop_size)
    ]


def compute_population_fitness(population, eval_fitness):
    """Returns list of tuples: (individual, fitness); computes fitness once per individual."""
    return [(ind, eval_fitness(ind)) for ind in population]


# ----------------------------
# Family Competition with Selection Tracking
# ----------------------------
def family_competition(crossover_operator, eval_fitness, parent1, parent2):
    """
    Given two parents, perform crossover to create two offspring.
    Evaluate all four individuals and select the best two.
    Also, for bit positions where parents differ, count:
      - selection errors (both winners have '0')
      - correct decisions (both winners have '1').
    """
    family = crossover_operator(parent1, parent2)
    members = [
        [eval_fitness(family[0]), family[0], True],
        [eval_fitness(family[1]), family[1], True],
        [eval_fitness(family[2]), family[2], False],
        [eval_fitness(family[3]), family[3], False],
    ]
    members.sort(key=lambda x: x[0], reverse=True)
    winners = members[:2]
    # improvement is marked if at least one winner is a new offspring.
    improvement = not (winners[0][2] and winners[1][2])

    sel_error = sel_correct = 0
    for i in range(STRING_LENGTH):
        if parent1[i] != parent2[i]:
            if winners[0][1][i] == winners[1][1][i]:
                if winners[0][1][i] == "0":
                    sel_error += 1
                elif winners[0][1][i] == "1":
                    sel_correct += 1
    return {
        "nextGen": winners,
        "improvement": improvement,
        "highest": winners[0],  # [fitness, individual, is_parent]
        "sel_error": sel_error,
        "sel_correct": sel_correct,
    }


# ----------------------------
# Core Genetic Algorithm (without tracing)
# ----------------------------
def genetic_algorithm(
    crossover_operator, fitness_function, pop_size, max_no_improve=20
):
    """
    GA stops if:
      1) The global optimum (fitness == STRING_LENGTH) is present in the population.
      2) No offspring with higher fitness than parents is produced,
      for max_no_improve consecutive generations.

    Args:
    - crossover_operator: the crossover operator to use.
    - fitness_function: the fitness function to optimize.
    - pop_size: the population size.
    - max_no_improve: the maximum number of generations without improvement to allow.

    Returns a dictionary with the following
    keys: generations, foundOptimal, noImprovement, fitness_evals.
    """
    fitness_evals = [0]

    def eval_fitness(ind):
        fitness_evals[0] += 1
        return fitness_function(ind)

    population = generate_population(pop_size)
    generation = 0
    no_improvement_count = 0

    # Track best overall for reporting, not for stopping
    best = [0, None]

    while True:
        # Compute population fitness once (for logging / best tracking)
        pop_fit = compute_population_fitness(population, eval_fitness)
        cur_best = max(pop_fit, key=lambda x: x[1])
        if cur_best[1] > best[0]:
            best = [cur_best[1], cur_best[0]]

        # --- Modified stopping criterion ---
        # If the best individual in the population is the global optimum, stop.
        if best[0] == STRING_LENGTH:
            break
        # ------------------------------------

        # If we've already had max_no_improve gens with no improvement, stop
        if no_improvement_count >= max_no_improve:
            break

        improvement_this_gen = False
        new_population = []
        random.shuffle(population)

        for parent1, parent2 in zip(population[::2], population[1::2]):
            result = family_competition(
                crossover_operator, eval_fitness, parent1, parent2
            )
            # If any offspring is strictly fitter than its parents, mark improvement.
            if result["improvement"]:
                improvement_this_gen = True

            new_population.extend([result["nextGen"][0][1], result["nextGen"][1][1]])

        if improvement_this_gen:
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        population = new_population
        generation += 1

    return {
        "generations": generation,
        "foundOptimal": (best[0] == STRING_LENGTH),
        "noImprovement": no_improvement_count,
        "fitness_evals": fitness_evals[0],
    }


# ----------------------------
# Genetic Algorithm with Detailed Tracing
# ----------------------------
def genetic_algorithm_trace(
    crossover_operator, fitness_function, pop_size, max_no_improve=20
):
    """
    Runs the GA for Counting Ones with detailed per-generation tracing.
    Records:
      - Proportion of ones in the population.
      - Selection errors and correct decisions.
      - Schema analysis (based on the first bit).
      - Fitness evaluations and CPU time.

    Stops if:
        1) The global optimum (fitness == STRING_LENGTH) is present in the population.
        2) No offspring
        with higher fitness than its parents is produced for max_no_improve consecutive generations.

    Args:
    - crossover_operator: the crossover operator to use.
    - fitness_function: the fitness function to optimize.
    - pop_size: the population size.
    - max_no_improve: the maximum number of generations without improvement to allow.

    Returns a dictionary with the following
    keys: generations, foundOptimal, noImprovement, fitness_evals, cpu_time, trace.
    """
    fitness_evals = [0]

    def eval_fitness(ind):
        fitness_evals[0] += 1
        return fitness_function(ind)

    population = generate_population(pop_size)
    generation = 0
    no_improvement_count = 0
    best = [0, None]
    trace_data = []
    start_time = time.time()

    while True:
        pop_fit = compute_population_fitness(population, eval_fitness)
        total_ones = sum(counting_ones(ind) for ind, _ in pop_fit)
        prop_ones = total_ones / (pop_size * STRING_LENGTH)

        # Schema stats: based on first bit
        schema1 = [fit for (ind, fit) in pop_fit if ind[0] == "1"]
        schema0 = [fit for (ind, fit) in pop_fit if ind[0] == "0"]

        def schema_stats(schema):
            if not schema:
                return (0, 0, 0)
            return (
                len(schema),
                statistics.mean(schema),
                statistics.pstdev(schema) if len(schema) > 1 else 0.0,
            )

        count1, avg1, stdev1 = schema_stats(schema1)
        count0, avg0, stdev0 = schema_stats(schema0)

        cur_best = max(pop_fit, key=lambda x: x[1])
        if cur_best[1] > best[0]:
            best = [cur_best[1], cur_best[0]]
        converged = best[0] == STRING_LENGTH

        trace_data.append(
            {
                "generation": generation,
                "prop_ones": prop_ones,
                "sel_error": 0,  # will accumulate below
                "sel_correct": 0,  # will accumulate below
                "schema1_count": count1,
                "schema1_avg": avg1,
                "schema1_stdev": stdev1,
                "schema0_count": count0,
                "schema0_avg": avg0,
                "schema0_stdev": stdev0,
            }
        )

        # --- Modified stopping criterion ---
        # Stop if the population contains the global optimum.
        if converged:
            break
        # ------------------------------------

        improvement_this_gen = False
        sel_error_total = 0
        sel_correct_total = 0
        new_population = []
        random.shuffle(population)

        for parent1, parent2 in zip(population[::2], population[1::2]):
            result = family_competition(
                crossover_operator, eval_fitness, parent1, parent2
            )
            sel_error_total += result["sel_error"]
            sel_correct_total += result["sel_correct"]

            if result["improvement"]:
                improvement_this_gen = True

            new_population.extend([result["nextGen"][0][1], result["nextGen"][1][1]])

        trace_data[-1]["sel_error"] = sel_error_total
        trace_data[-1]["sel_correct"] = sel_correct_total

        if improvement_this_gen:
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= max_no_improve:
            break

        population = new_population
        generation += 1

    cpu_time = time.time() - start_time
    return {
        "generations": generation,
        "foundOptimal": converged,
        "noImprovement": no_improvement_count,
        "fitness_evals": fitness_evals[0],
        "cpu_time": cpu_time,
        "trace": trace_data,
    }


# ----------------------------
# Reliability and Metrics Helpers
# ----------------------------
def run_experiment_metrics(
    crossover_operator, fitness_function, pop_size, runs=10, max_no_improve=20
):
    """Runs GA 'runs' times with the given parameters and returns metrics."""
    results = []
    for _ in range(runs):
        start = time.time()
        result = genetic_algorithm(
            crossover_operator, fitness_function, pop_size, max_no_improve
        )
        cpu_time = time.time() - start
        results.append(
            (
                result["generations"],
                result["fitness_evals"],
                cpu_time,
                result["foundOptimal"],
            )
        )
    return results


def summarize_metrics(results):
    """Computes averages and standard deviations for generations, evaluations, and CPU time."""
    successes = [r for r in results if r[3]]
    num_success = len(successes)
    gens = [r[0] for r in successes]
    evals = [r[1] for r in successes]
    times = [r[2] for r in successes]
    summary = {
        "num_success": num_success,
        "avg_generations": statistics.mean(gens) if gens else None,
        "std_generations": statistics.pstdev(gens) if len(gens) > 1 else 0,
        "avg_evals": statistics.mean(evals) if evals else None,
        "std_evals": statistics.pstdev(evals) if len(evals) > 1 else 0,
        "avg_cpu": statistics.mean(times) if times else None,
        "std_cpu": statistics.pstdev(times) if len(times) > 1 else 0,
    }
    return summary


# ----------------------------
# Population Size Search (Doubling + Bisection)
# ----------------------------
def run_10_times(crossover_operator, fitness_function, pop_size):
    """Returns True if at least 9 out of 10 independent runs find the global optimum."""
    successes = sum(
        1
        for _ in range(10)
        if genetic_algorithm(crossover_operator, fitness_function, pop_size)[
            "foundOptimal"
        ]
    )
    return successes >= 9


def find_minimum_population_size(crossover_operator, fitness_function):
    """
    Finds the minimal population size (multiple of 10) using doubling then bisection.
    """
    min_pop, max_pop = 10, 1280
    if run_10_times(crossover_operator, fitness_function, min_pop):
        return min_pop

    found = False
    while min_pop < max_pop and not found:
        mid = min(min_pop * 2, max_pop)
        if run_10_times(crossover_operator, fitness_function, mid):
            max_pop = mid
            found = True
        else:
            min_pop = mid
        print(f"Doubling phase: min_pop={min_pop}, mid_pop={mid}, max_pop={max_pop}")

    if not found:
        return "FAIL"

    while max_pop - min_pop > 10:
        mid = int(round((min_pop + max_pop) / 20.0)) * 10  # ensure multiple of 10
        mid = max(mid, 10)  # Make sure it is at least 10
        if run_10_times(crossover_operator, fitness_function, mid):
            max_pop = mid
        else:
            min_pop = mid
        print(f"Bisection phase: min_pop={min_pop}, mid_pop={mid}, max_pop={max_pop}")
    return max_pop


# ----------------------------
# Main Execution: Run All Experiments and Save Results
# ----------------------------
if __name__ == "__main__":
    # Create a results folder if it doesn't exist
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Define crossover operators to test
    crossover_methods = {"2X": two_point_crossover, "UX": uniform_crossover}
    # Define experiments: each experiment is a dictionary with a fitness function and name.
    experiments = [
        {
            "name": "Counting_Ones",
            "fitness": counting_ones,
            "description": "Counting Ones function",
        },
        {
            "name": "Deceptive_Trap_Tight",
            "fitness": lambda sol: trap_function_tight(sol, 1),
            "description": "Deceptive Trap Function (Tightly Linked)",
        },
        {
            "name": "Nondeceptive_Trap_Tight",
            "fitness": lambda sol: trap_function_tight(sol, 2.5),
            "description": "Non-deceptive Trap Function (Tightly Linked)",
        },
        {
            "name": "Deceptive_Trap_Spread",
            "fitness": lambda sol: trap_function_spread(sol, 1),
            "description": "Deceptive Trap Function (Not Tightly Linked)",
        },
        {
            "name": "Nondeceptive_Trap_Spread",
            "fitness": lambda sol: trap_function_spread(sol, 2.5),
            "description": "Non-deceptive Trap Function (Not Tightly Linked)",
        },
    ]

    summary_results = {}
    for exp in experiments:
        EXP_NAME = exp["name"]
        summary_results[EXP_NAME] = {}
        for cross_name, cross_func in crossover_methods.items():
            min_pop = find_minimum_population_size(cross_func, exp["fitness"])
            result_filename = os.path.join(
                RESULTS_DIR, f"{EXP_NAME}_{cross_name}_results.txt"
            )
            with open(result_filename, "w", encoding="utf-8") as f:
                f.write(f"Experiment: {exp['description']}\n")
                f.write(f"Crossover Operator: {cross_name}\n")
                f.write(f"Minimal Population Size: {min_pop}\n")
            if min_pop != "FAIL":
                results = run_experiment_metrics(cross_func, exp["fitness"], min_pop)
                summary = summarize_metrics(results)
                summary_results[EXP_NAME][cross_name] = {
                    "min_pop": min_pop,
                    "metrics": summary,
                }
                with open(result_filename, "a", encoding="utf-8") as f:
                    f.write("Performance Metrics (over 10 runs on minimal pop):\n")
                    f.write(f"  Successful runs: {summary['num_success']}/10\n")
                    f.write(
                        f"  Avg Generations: {summary['avg_generations']} "
                        f"(Std: {summary['std_generations']})\n"
                    )
                    f.write(
                        f"  Avg Fitness Evaluations: {summary['avg_evals']}"
                        f"(Std: {summary['std_evals']})\n"
                    )
                    f.write(
                        f"  Avg CPU Time (s): {summary['avg_cpu']} (Std: {summary['std_cpu']})\n"
                    )
            else:
                summary_results[EXP_NAME][cross_name] = "FAIL"

    with open(os.path.join(RESULTS_DIR, "Master_Summary.txt"), "w", encoding="utf-8") as f:
        for EXP_NAME, data in summary_results.items():
            f.write(f"Experiment: {EXP_NAME}\n")
            for cross_name, metrics in data.items():
                f.write(f"  Crossover: {cross_name} -> {metrics}\n")
            f.write("\n")

    if "Counting_Ones" in [e["name"] for e in experiments]:
        trace_result = genetic_algorithm_trace(uniform_crossover, counting_ones, 200)
        trace_summary_file = os.path.join(RESULTS_DIR, "Counting_Ones_UX_Tracing.txt")
        with open(trace_summary_file, "w", encoding="utf-8") as f:
            f.write("Counting Ones (Uniform Crossover, N=200) Trace Summary:\n")
            f.write(f"Generations: {trace_result['generations']}\n")
            f.write(f"Fitness Evaluations: {trace_result['fitness_evals']}\n")
            f.write(f"CPU Time (s): {trace_result['cpu_time']}\n")
            f.write(f"Converged: {trace_result['foundOptimal']}\n")

        trace = trace_result["trace"]
        generations = [d["generation"] for d in trace]
        prop_ones = [d["prop_ones"] for d in trace]
        sel_errors = [d["sel_error"] for d in trace]
        sel_corrects = [d["sel_correct"] for d in trace]
        schema1_counts = [d["schema1_count"] for d in trace]
        schema0_counts = [d["schema0_count"] for d in trace]
        schema1_avg = [d["schema1_avg"] for d in trace]
        schema1_stdev = [d["schema1_stdev"] for d in trace]
        schema0_avg = [d["schema0_avg"] for d in trace]
        schema0_stdev = [d["schema0_stdev"] for d in trace]

        plt.figure(figsize=(10, 6))
        plt.plot(generations, prop_ones, marker="o")
        plt.title("Proportion of Ones vs. Generation")
        plt.xlabel("Generation")
        plt.ylabel("Proportion of Ones")
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, "Experiment1_ProportionOfOnes.png"))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(generations, sel_errors, label="Selection Errors", marker="x")
        plt.plot(generations, sel_corrects, label="Correct Selections", marker="o")
        plt.title("Selection Decisions vs. Generation")
        plt.xlabel("Generation")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, "Experiment1_SelectionDecisions.png"))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(
            generations, schema1_counts, label="Schema 1* (first bit = 1)", marker="o"
        )
        plt.plot(
            generations, schema0_counts, label="Schema 0* (first bit = 0)", marker="x"
        )
        plt.title("Schema Counts vs. Generation")
        plt.xlabel("Generation")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, "Experiment1_SchemaCounts.png"))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.errorbar(
            generations,
            schema1_avg,
            yerr=schema1_stdev,
            label="Schema 1* Fitness",
            marker="o",
            capsize=5,
        )
        plt.errorbar(
            generations,
            schema0_avg,
            yerr=schema0_stdev,
            label="Schema 0* Fitness",
            marker="x",
            capsize=5,
        )
        plt.title("Schema Fitness vs. Generation")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, "Experiment1_SchemaFitness.png"))
        plt.close()

    print("All experiments complete. Results stored in the 'results' folder.")
