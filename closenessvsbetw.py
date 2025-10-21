import matplotlib.pyplot as plt
import networkx as nx
import random
from tqdm import tqdm
import numpy as np

# Store win rates of betweenness vs closeness across different edge densities
node_count = 50
edge_ranges = range(60, 301, 20)  # From sparse to dense
betweenness_win_rates = []

def count_bfs_iterations(G, start_node):
    visited = set()
    queue = [start_node]
    iterations = 0

    while queue:
        next_queue = []
        for node in queue:
            if node not in visited:
                visited.add(node)
                next_queue.extend(n for n in G.neighbors(node) if n not in visited)
        queue = next_queue
        iterations += 1

        if len(visited) == len(G):
            break

    return iterations

# Run experiment for each edge count
for n_edges in tqdm(edge_ranges):
    trials = 200
    betweenness_better = 0
    valid_trials = 0

    for _ in range(trials):
        G = nx.gnm_random_graph(n=node_count, m=n_edges)
         # Skip if not connected
        if not nx.is_connected(G):
            continue

        closeness = nx.closeness_centrality(G)
        betweenness = nx.betweenness_centrality(G)

        max_closeness_value = max(closeness.values())
        max_betweenness_value = max(betweenness.values())

        best_closeness_nodes = [node for node, value in closeness.items() if value == max_closeness_value]
        best_betweenness_nodes = [node for node, value in betweenness.items() if value == max_betweenness_value]

        if len(best_closeness_nodes) > 1 or len(best_betweenness_nodes) > 1:
            continue
        if set(best_closeness_nodes) & set(best_betweenness_nodes):
            continue

        best_closeness_node = best_closeness_nodes[0]
        best_betweenness_node = best_betweenness_nodes[0]

        closeness_iters = count_bfs_iterations(G, best_closeness_node)
        betweenness_iters = count_bfs_iterations(G, best_betweenness_node)

        if betweenness_iters <= closeness_iters:
            betweenness_better += 1
        valid_trials += 1

    win_rate = betweenness_better / valid_trials if valid_trials > 0 else 0
    betweenness_win_rates.append(win_rate)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(edge_ranges, betweenness_win_rates, marker='o')
plt.title("Win Rate of Betweenness over Closeness (Lower BFS Iterations)")
plt.xlabel("Number of Edges in Graph (n=50)")
plt.ylabel("Proportion of Trials Where Betweenness Is Better")
plt.grid(True)
plt.ylim(0, 1)
plt.show()