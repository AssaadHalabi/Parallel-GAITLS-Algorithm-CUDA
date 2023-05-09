#include "init_rcl.h"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <limits>
#include <algorithm>
#include <ctime>

DominatingTreeSolution ITLS(const Graph &graph, int cutoff_time, DominatingTreeSolution DT) {
    std::unordered_set<int> tabu_list;
    DominatingTreeSolution DT_prime = DT;
    int num_vertices = graph.getNumVertices();
    std::vector<int> Dscore(num_vertices);

    for (int v = 0; v < num_vertices; ++v) {
        if (DT.getDominatingVertices().count(v) > 0) {
            Dscore[v] = -1 * compute_Dscore(v, DT, graph);
        } else {
            Dscore[v] = compute_Dscore(v, DT, graph);
        }
    }

    auto start_time = std::clock();
    double elapsed_time = 0;

    while (elapsed_time < cutoff_time) {
        if (!has_non_dominated_vertices(Dscore) && DT.isConnected()) {
            remove_redundant_vertices(DT, graph);
            connect_minimum_spanning_tree(DT, graph);
            if (DT.getTotalWeight() < DT_prime.getTotalWeight()) {
                DT_prime = DT;
            }
        }

        // Removing Phase
        while (!has_non_dominated_vertices(Dscore)) {
            int max_dscore = std::numeric_limits<int>::min();
            int vertex_to_remove = -1;
            for (int v = 0; v < num_vertices; ++v) {
                if (DT.getDominatingVertices().count(v) > 0 && tabu_list.count(v) == 0 && Dscore[v] > max_dscore) {
                    max_dscore = Dscore[v];
                    vertex_to_remove = v;
                }
            }
            if (vertex_to_remove != -1) {
                DT.removeVertex(vertex_to_remove);
                update_Dscore(Dscore, vertex_to_remove, DT, graph);
            }
        }

        // Dominating Phase
        tabu_list.clear();
        while (has_non_dominated_vertices(Dscore)) {
            auto shortest_paths = floyd_warshall(graph);
            int min_vertex = -1;
            double min_score = std::numeric_limits<double>::max();
            for (int v = 0; v < num_vertices; ++v) {
                if (DT.getDominatingVertices().count(v) == 0) {
                    double wscore = compute_Wscore(v, DT, shortest_paths);
                    double score = wscore / Dscore[v];
                    if (score < min_score) {
                        min_score = score;
                        min_vertex = v;
                    }
                }
            }
            if (min_vertex != -1) {
                DT.addVertex(min_vertex);
                tabu_list.insert(min_vertex);
                update_Dscore(Dscore, min_vertex, DT, graph);
            }
        }

        // Connecting Phase
        while (!DT.isConnected()) {
            auto shortest_paths = floyd_warshall(graph);
            auto components = DT.getDisconnectedComponents();
            auto [u, v] = DT.getShortestPathBetweenComponents(components, shortest_paths);
            DT.addVerticesAlongPath(shortest_paths[u][v]);
            for (auto vertex : DT.getPathVertices(shortest_paths[u][v])) {
                tabu_list.insert(vertex);
                update_Dscore(Dscore, vertex, DT, graph);
            }
        }

        elapsed_time = (std::clock() - start_time) / static_cast<double>(CLOCKS_PER_SEC);
    }

    return DT_prime;
}
