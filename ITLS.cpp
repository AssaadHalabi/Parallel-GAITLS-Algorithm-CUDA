#include "ITLS.h"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <limits>
#include <algorithm>
#include <ctime>

void removingPhase(DominatingTreeSolution &DT, std::unordered_set<int> &tabu_list, std::vector<int> &Dscore, const Graph &graph, int num_vertices)
{
    // Removing Phase
    while (!has_non_dominated_vertices(Dscore))
    {
        int max_dscore = std::numeric_limits<int>::min();
        int vertex_to_remove = -1;
        for (int v = 0; v < num_vertices; ++v)
        {
            if (DT.getDominatingVertices().count(v) > 0 && tabu_list.count(v) == 0 && Dscore[v] > max_dscore)
            {
                max_dscore = Dscore[v];
                vertex_to_remove = v;
            }
        }
        if (vertex_to_remove != -1)
        {
            DT.removeVertex(vertex_to_remove);
            update_Dscore(Dscore, vertex_to_remove, DT, graph);
        }
    }
}

void dominatingPhase(DominatingTreeSolution &DT, std::unordered_set<int> &tabu_list, std::vector<int> &Dscore, const Graph &graph, int num_vertices)
{
    // Dominating Phase
    tabu_list.clear();
    while (has_non_dominated_vertices(Dscore))
    {
        auto [shortest_paths, next] = floyd_warshall(graph);
        int min_vertex = -1;
        double min_score = std::numeric_limits<double>::max();
        for (int v = 0; v < num_vertices; ++v)
        {
            if (DT.getDominatingVertices().count(v) == 0)
            {
                double wscore = compute_Wscore(v, DT, shortest_paths);
                double score = wscore / Dscore[v];
                if (score < min_score)
                {
                    min_score = score;
                    min_vertex = v;
                }
            }
        }
        if (min_vertex != -1)
        {
            DT.addVertex(min_vertex);
            tabu_list.insert(min_vertex);
            update_Dscore(Dscore, min_vertex, DT, graph);
        }
    }
}

void connectingPhase(DominatingTreeSolution &DT, std::unordered_set<int> &tabu_list, std::vector<int> &Dscore, const Graph &graph)
{
    // Connecting Phase
    while (!DT.isConnected())
    {
        auto [shortest_paths, next] = floyd_warshall(graph);
        auto components = DT.getDisconnectedComponents();
        auto path = DT.getShortestPathBetweenComponents(components, shortest_paths, next);

        DT.addVerticesAlongPath(path);
        for (auto vertex : path)
        {
            tabu_list.insert(vertex);
            update_Dscore(Dscore, vertex, DT, graph);
        }
    }
}

DominatingTreeSolution ITLS(const Graph &graph, int max_iterations, DominatingTreeSolution DT)
{
    std::unordered_set<int> tabu_list;
    DominatingTreeSolution *DT_prime = new DominatingTreeSolution(DT);
    int num_vertices = graph.getNumVertices();
    std::vector<int> Dscore(num_vertices);

    for (int v = 0; v < num_vertices; ++v)
    {
        Dscore[v] = compute_Dscore(v, DT, graph);
    }

    int iteration = 0;
    while (iteration < max_iterations)
    {
        if (!has_non_dominated_vertices(Dscore) && DT.isConnected())
        {
            remove_redundant_vertices(DT, graph);
            connect_minimum_spanning_tree(DT, graph);
            if (DT.getTotalWeight() < DT_prime->getTotalWeight())
            {
                delete DT_prime; // Delete the old DT_prime before reassigning it
                DT_prime = new DominatingTreeSolution(DT);
            }
        }

        // Removing Phase
        removingPhase(DT, tabu_list, Dscore, graph, num_vertices);
        // Dominating Phase
        dominatingPhase(DT, tabu_list, Dscore, graph, num_vertices);
        // Connecting Phase
        connectingPhase(DT, tabu_list, Dscore, graph);

        ++iteration;
    }

    DominatingTreeSolution result = *DT_prime;
    delete DT_prime;
    return result;
}
