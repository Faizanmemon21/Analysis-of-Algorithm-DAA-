"""
Evaluation harness for the IFBL Route Optimizer.
Generates REAL measured data (not estimates) for the CCP report's
Testing & Evaluation section:
  1. Scalability: how Held-Karp's runtime grows vs. n (exponential blow-up).
  2. Crossover point: where Held-Karp becomes impractical.
  3. Accuracy: how close the Nearest-Neighbor heuristic gets to the
     Held-Karp optimum, measured as % gap, across multiple random instances.
  4. Efficiency: % distance reduction of the optimized route vs. random
     "maverick" routes, averaged over multiple trials for statistical validity.
"""

import random
import time
import statistics
from ifbl_route_optimizer import (
    held_karp, nearest_neighbor, build_distance_matrix, random_maverick_route
)


def random_locations(n, seed):
    rng = random.Random(seed)
    locs = {0: ("Depot", 0, 0)}
    for i in range(1, n + 1):
        locs[i] = (f"Addr{i}", rng.uniform(-15, 15), rng.uniform(-15, 15))
    return locs


def test_scalability():
    print("=" * 70)
    print("TEST 1: SCALABILITY OF HELD-KARP (EXACT) ALGORITHM")
    print("=" * 70)
    print(f"{'n':>4} {'Held-Karp time (s)':>20} {'states explored (approx)':>26}")
    results = []
    for n in [4, 6, 8, 10, 12, 14]:
        locs = random_locations(n, seed=42)
        dist = build_distance_matrix(locs)
        start = time.perf_counter()
        cost, route = held_karp(dist)
        elapsed = time.perf_counter() - start
        states = (n) * (2 ** (n - 1))  # O(n * 2^n) states, n delivery nodes
        results.append((n, elapsed, states))
        print(f"{n:>4} {elapsed:>20.4f} {states:>26,}")
    print("\nObservation: runtime grows exponentially with n, confirming the")
    print("O(n^2 * 2^n) theoretical complexity. This is why the CCP design")
    print("caps the exact solver at a practical threshold and switches to")
    print("the heuristic for larger batches.\n")
    return results


def test_heuristic_accuracy():
    print("=" * 70)
    print("TEST 2: NEAREST-NEIGHBOR HEURISTIC ACCURACY vs. HELD-KARP OPTIMUM")
    print("=" * 70)
    print(f"{'n':>4} {'Optimal (km)':>14} {'Heuristic (km)':>16} {'Gap %':>8}")
    gaps = []
    for trial in range(10):
        n = random.Random(trial).choice([6, 7, 8, 9, 10])
        locs = random_locations(n, seed=1000 + trial)
        dist = build_distance_matrix(locs)
        opt_cost, _ = held_karp(dist)
        heur_cost, _ = nearest_neighbor(dist)
        gap = (heur_cost - opt_cost) / opt_cost * 100 if opt_cost > 0 else 0
        gaps.append(gap)
        print(f"{n:>4} {opt_cost:>14.2f} {heur_cost:>16.2f} {gap:>7.1f}%")
    print(f"\nAverage gap over {len(gaps)} trials: {statistics.mean(gaps):.1f}%")
    print(f"Worst-case gap observed: {max(gaps):.1f}%")
    print("\nObservation: the heuristic does not guarantee optimality (as")
    print("expected, since it is greedy), but stays within a moderate margin")
    print("of the true optimum while running in polynomial time.\n")
    return gaps


def test_efficiency_vs_maverick():
    print("=" * 70)
    print("TEST 3: OPTIMIZED ROUTE vs. RANDOM 'MAVERICK' ROUTE")
    print("=" * 70)
    print(f"{'n':>4} {'Optimal (km)':>14} {'Maverick avg (km)':>18} {'Reduction %':>13}")
    reductions = []
    for n in [6, 8, 10]:
        locs = random_locations(n, seed=7)
        dist = build_distance_matrix(locs)
        opt_cost, _ = held_karp(dist)
        maverick_costs = []
        for trial in range(20):
            random.seed(trial)
            mcost, _ = random_maverick_route(dist)
            maverick_costs.append(mcost)
        avg_maverick = statistics.mean(maverick_costs)
        reduction = (avg_maverick - opt_cost) / avg_maverick * 100
        reductions.append(reduction)
        print(f"{n:>4} {opt_cost:>14.2f} {avg_maverick:>18.2f} {reduction:>12.1f}%")
    print(f"\nAverage reduction across tested n: {statistics.mean(reductions):.1f}%")
    print("\nObservation: optimized routing consistently outperforms random")
    print("rider-chosen ('maverick') routes, supporting the CEO's economic")
    print("goal of reduced fuel mileage stated in WP6.\n")
    return reductions


if __name__ == "__main__":
    test_scalability()
    test_heuristic_accuracy()
    test_efficiency_vs_maverick()
