# Evolutionary Computation (INFOEA) – Practical Assignment 2 (2024–2025)

**Authors**:  
Daan Westland (1675877)  
Julius Bijkerk (1987011)

**Course**: Evolutionary Computation – Utrecht University  
**Instructor**: dr. Dirk Thierens (d.thierens@uu.nl)

---

## Overview

This repository contains our final implementation and analysis for Practical Assignment 2 of the Evolutionary Computation course. The project focuses on solving the **Graph Bipartitioning Problem (GBP)** using multiple metaheuristic strategies.

All experiments and figures in the report were generated from a single Python script: `project2_gridsearch.py`

---

## Methods Implemented

- **MLS** – Multi-start Local Search  
- **ILS** – Iterated Local Search (with greedy, SA, and adaptive variants)  
- **GLS** – Genetic Local Search  
- All methods use the **Fiduccia-Mattheyses (FM)** algorithm as local optimizer.

---

## Experiment Design

Two main evaluation settings:
1. **Fixed FM Passes** (10,000 passes)
2. **Fixed Runtime** (based on average MLS runtime)

The variants of ILS tested include:
- Simulated Annealing acceptance (ILS-SA)
- Adaptive mutation size (ILS-A)

---

## 🛠️ Running the Code

```bash
python project2_gridsearch.py
