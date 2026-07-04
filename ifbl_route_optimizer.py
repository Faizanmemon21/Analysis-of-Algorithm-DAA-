"""
IFBL Route Optimizer
=====================
Complex Computing Problem (CCP) - Design and Analysis of Algorithm
Project: Minimum Length Trip Planning for Food Delivery Optimization (IFBL)

Implements:
  1. Exact TSP solver using the Held-Karp dynamic programming algorithm
     (guarantees the minimum-cost route; practical for n <= ~15-18 addresses).
  2. Heuristic TSP solver using Nearest-Neighbor construction
     (fast, near-optimal route for larger delivery batches).

The program is menu-driven, as required by the CCP "Implementation" deliverable.
"""

import itertools
import math
import time

# ---------------------------------------------------------------------------
# Real IFBL Karachi Data
# ---------------------------------------------------------------------------
# Node 0 is always the IFBL Head Office (the depot).
# Coordinates are REAL GPS positions (latitude, longitude) for Karachi.
# Distance is calculated using the Haversine formula (real km on Earth).

LOCATIONS = {
    0:  ("IFBL Head Office (Depot)", 24.8607, 67.0011),
    1:  ("Gulshan-e-Iqbal",          24.9215, 67.0944),
    2:  ("DHA Phase 5",              24.8123, 67.0775),
    3:  ("Clifton Block 5",          24.8133, 67.0298),
    4:  ("North Nazimabad",          24.9474, 67.0304),
    5:  ("Saddar",                   24.8553, 67.0104),
    6:  ("Korangi",                  24.8305, 67.1284),
    7:  ("Malir",                    24.8930, 67.1927),
    8:  ("Tariq Road",               24.8615, 67.0317),
    9:  ("Nazimabad",                24.9158, 67.0285),
    10: ("Liaquatabad",              24.9040, 67.0423),
    11: ("FB Area",                  24.9370, 67.0650),
    12: ("Orangi Town",              24.9480, 66.9942),
    13: ("Landhi",                   24.8527, 67.2052),
    14: ("Shah Faisal Colony",       24.8741, 67.1562),
}


def haversine(a, b):
    """
    Real-world distance between two GPS coordinates (lat, lng) in km.
    Uses the Haversine formula for spherical Earth distance.
    Much more accurate than Euclidean for real map coordinates.
    """
    R = 6371  # Earth radius in km
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    s = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(s), math.sqrt(1 - s))


def build_distance_matrix(locations):
    """Build an n x n real-distance matrix using Haversine formula."""
    ids  = sorted(locations.keys())
    n    = len(ids)
    dist = [[0.0] * n for _ in range(n)]
    for i in ids:
        for j in ids:
            if i != j:
                _, lati, loni = locations[i]
                _, latj, lonj = locations[j]
                dist[i][j] = haversine((lati, loni), (latj, lonj))
    return dist


# ---------------------------------------------------------------------------
# 1. Exact Algorithm: Held-Karp Dynamic Programming
# ---------------------------------------------------------------------------
def held_karp(dist):
    """
    Solves TSP exactly using the Held-Karp algorithm.

    dist : n x n distance matrix, node 0 is the depot.
    Returns (min_cost, route) where route is a list of node indices
    starting and ending at the depot (0).

    Complexity: O(n^2 * 2^n) time, O(n * 2^n) space.
    Practical only for small n (recommended n <= 15-18).
    """
    n = len(dist)
    if n <= 1:
        return 0.0, [0]

    other_nodes = list(range(1, n))
    # dp[(subset, j)] = (min cost to start at 0, visit all nodes in subset,
    #                     and end at node j)
    dp = {}
    parent = {}

    # Base case: subsets of size 1 (just node j), starting from depot
    for j in other_nodes:
        dp[(frozenset([j]), j)] = dist[0][j]
        parent[(frozenset([j]), j)] = 0

    # Build up subsets of increasing size
    for subset_size in range(2, n):
        for subset in itertools.combinations(other_nodes, subset_size):
            subset_fs = frozenset(subset)
            for j in subset:
                prev_subset = subset_fs - {j}
                best_cost = math.inf
                best_prev = None
                for k in prev_subset:
                    candidate = dp[(prev_subset, k)] + dist[k][j]
                    if candidate < best_cost:
                        best_cost = candidate
                        best_prev = k
                dp[(subset_fs, j)] = best_cost
                parent[(subset_fs, j)] = best_prev

    # Close the tour: return to depot (node 0) from the last visited node
    full_set = frozenset(other_nodes)
    best_cost = math.inf
    best_last = None
    for j in other_nodes:
        candidate = dp[(full_set, j)] + dist[j][0]
        if candidate < best_cost:
            best_cost = candidate
            best_last = j

    # Reconstruct the route by walking parent pointers backwards
    route = [0]
    subset = full_set
    last = best_last
    path = []
    while last != 0:
        path.append(last)
        prev = parent[(subset, last)]
        subset = subset - {last}
        last = prev
    route.extend(reversed(path))
    route.append(0)

    return best_cost, route


# ---------------------------------------------------------------------------
# 2. Heuristic Algorithm: Nearest-Neighbor
# ---------------------------------------------------------------------------
def nearest_neighbor(dist, start=0):
    """
    Constructs a fast, near-optimal route by always moving to the closest
    unvisited delivery address. Used for larger delivery batches where the
    exact algorithm would be too slow.

    Complexity: O(n^2) time.
    """
    n = len(dist)
    unvisited = set(range(n)) - {start}
    route = [start]
    current = start
    total_cost = 0.0

    while unvisited:
        nearest = min(unvisited, key=lambda j: dist[current][j])
        total_cost += dist[current][nearest]
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    total_cost += dist[current][start]
    route.append(start)
    return total_cost, route


# ---------------------------------------------------------------------------
# 3. Algorithm Selection (matches the CCP system design flowchart)
# ---------------------------------------------------------------------------
EXACT_LIMIT = 13  # safe upper bound for Held-Karp in this menu program


def solve(dist, n_addresses, force_method=None):
    """
    Chooses Exact (Held-Karp) for small n, Heuristic (Nearest-Neighbor)
    for larger n -- matching Section 5 of the CCP report.
    """
    if force_method == "exact" or (force_method is None and n_addresses <= EXACT_LIMIT):
        method = "Held-Karp (Exact)"
        start = time.perf_counter()
        cost, route = held_karp(dist)
        elapsed = time.perf_counter() - start
    else:
        method = "Nearest-Neighbor (Heuristic)"
        start = time.perf_counter()
        cost, route = nearest_neighbor(dist)
        elapsed = time.perf_counter() - start
    return method, cost, route, elapsed


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def print_route(locations, method, cost, route, elapsed):
    print(f"\n{'='*55}")
    print(f"  Algorithm    : {method}")
    print(f"  Total Dist   : {cost:.2f} km  (Real Karachi distances)")
    print(f"  Compute Time : {elapsed * 1000:.3f} ms")
    print(f"{'='*55}")
    print("  Optimized Rider Trip:")
    for step, node in enumerate(route, start=1):
        name = locations[node][0]
        lat  = locations[node][1]
        lng  = locations[node][2]
        arrow = " →" if step < len(route) else ""
        tag = " [DEPOT]" if node == 0 else f" ({lat:.4f}, {lng:.4f})"
        print(f"    {step:>2}. {name}{tag}{arrow}")
    print()


def list_locations(locations):
    print("\nCurrent IFBL delivery addresses (Real Karachi GPS Coordinates):")
    print(f"  {'ID':<4} {'Name':<30} {'Latitude':>10} {'Longitude':>11}")
    print("  " + "-"*58)
    for node_id, (name, lat, lng) in sorted(locations.items()):
        tag = " ← DEPOT" if node_id == 0 else ""
        print(f"  [{node_id:<2}] {name:<30} {lat:>10.4f} {lng:>11.4f}{tag}")
    print()


def random_maverick_route(dist):
    """A random (non-optimized) route, used as the baseline for Section 6:
    Testing & Evaluation - 'percentage reduction vs. random maverick rider routes'."""
    import random
    n = len(dist)
    nodes = list(range(1, n))
    random.shuffle(nodes)
    route = [0] + nodes + [0]
    cost = sum(dist[route[i]][route[i + 1]] for i in range(len(route) - 1))
    return cost, route


# ---------------------------------------------------------------------------
# Menu-Driven Program
# ---------------------------------------------------------------------------
def main_menu():
    locations = dict(LOCATIONS)  # mutable copy
    dist      = build_distance_matrix(locations)

    banner = (
        "\n============================================================\n"
        "   IFBL ROUTE OPTIMIZER - Minimum Length Trip Planning\n"
        "   Real Karachi GPS Coordinates | Haversine Distance\n"
        "============================================================"
    )
    print(banner)

    while True:
        n_addresses = len(locations) - 1
        print(f"\nMENU  ({n_addresses} delivery locations loaded)")
        print("  1. View delivery addresses")
        print(f"  2. Auto-optimize route (Exact if n<={EXACT_LIMIT}, else Heuristic)")
        print("  3. Force Exact (Held-Karp) solver")
        print("  4. Force Heuristic (Nearest-Neighbor) solver")
        print("  5. Compare optimized route vs. random maverick route")
        print("  6. Add a custom delivery location")
        print("  7. Exit")
        choice = input("Select an option (1-7): ").strip()

        if choice == "1":
            list_locations(locations)

        elif choice == "2":
            dist = build_distance_matrix(locations)
            method, cost, route, elapsed = solve(dist, n_addresses)
            print_route(locations, method, cost, route, elapsed)

        elif choice == "3":
            if n_addresses > 20:
                print("\nWarning: n is large; Held-Karp may be slow (exponential time).")
            dist = build_distance_matrix(locations)
            method, cost, route, elapsed = solve(dist, n_addresses, force_method="exact")
            print_route(locations, method, cost, route, elapsed)

        elif choice == "4":
            dist = build_distance_matrix(locations)
            method, cost, route, elapsed = solve(dist, n_addresses, force_method="heuristic")
            print_route(locations, method, cost, route, elapsed)

        elif choice == "5":
            dist = build_distance_matrix(locations)
            method, opt_cost, opt_route, _ = solve(dist, n_addresses)
            maverick_cost, maverick_route   = random_maverick_route(dist)
            reduction = (
                (maverick_cost - opt_cost) / maverick_cost * 100
                if maverick_cost > 0 else 0
            )
            print(f"\n  Optimized ({method}) : {opt_cost:.2f} km")
            print(f"  Random maverick      : {maverick_cost:.2f} km")
            print(f"  Reduction achieved   : {reduction:.1f}%")

        elif choice == "6":
            print("\n  Add Custom Delivery Location")
            name = input("  Location name       : ").strip()
            try:
                lat = float(input("  Latitude  (e.g. 24.9215) : "))
                lng = float(input("  Longitude (e.g. 67.0944) : "))
                new_id = max(locations.keys()) + 1
                locations[new_id] = (name, lat, lng)
                dist = build_distance_matrix(locations)
                print(f"  ✅ '{name}' added as node [{new_id}] ({lat}, {lng})")
            except ValueError:
                print("  ❌ Invalid coordinates. Please enter numbers.")

        elif choice == "7":
            print("\nExiting IFBL Route Optimizer. Goodbye!\n")
            break

        else:
            print("\nInvalid option. Please choose 1-7.")


if __name__ == "__main__":
    main_menu()
