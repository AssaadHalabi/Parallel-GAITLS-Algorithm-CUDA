#include <iostream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <limits>
#include <random>
#include <memory>
#include <queue>
#include <tuple>
#include "graph.h"

std::vector<std::unique_ptr<DominatingTreeSolution>> init_RCL(const Graph &graph, int IndiNum, double alpha)
{
    int num_vertices = graph.getNumVertices();
    std::vector<std::unique_ptr<DominatingTreeSolution>> POPinit;

    // Compute the shortest path and predecessor for each vertex pair
    auto [shortest_paths, predecessors] = floyd_warshall(graph);

    while (IndiNum > 0)
    {
        DominatingTreeSolution DT(graph);
        std::unordered_set<int> CL;

        std::vector<int> Dscore(num_vertices);
        for (int v = 0; v < num_vertices; ++v)
        {
            Dscore[v] = graph.getDegree(v);
        }

        while (has_non_dominated_vertices(Dscore))
        {
            std::vector<int> RCL;

            if (DT.getDominatingVertices().empty())
            {
                int maxscore = *std::max_element(Dscore.begin(), Dscore.end());
                int minscore = *std::min_element(Dscore.begin(), Dscore.end());

                for (int v = 0; v < num_vertices; ++v)
                {
                    if (Dscore[v] >= minscore + alpha * (maxscore - minscore))
                    {
                        RCL.push_back(v);
                    }
                }
            }
            else
            {
                double maxscore = -std::numeric_limits<double>::infinity();
                double minscore = std::numeric_limits<double>::infinity();

                for (int v : CL)
                {
                    double Wuv = compute_Wscore(v, DT, shortest_paths);
                    double score = Dscore[v] / Wuv;

                    maxscore = std::max(maxscore, score);
                    minscore = std::min(minscore, score);
                }

                for (int v : CL)
                {
                    double Wuv = compute_Wscore(v, DT, shortest_paths);
                    double score = Dscore[v] / Wuv;

                    if (score >= minscore + alpha * (maxscore - minscore))
                    {
                        RCL.push_back(v);
                    }
                }
            }

            int AddVertex = RCL[rand() % RCL.size()];

            if (DT.getDominatingVertices().empty())
            {
                for (const auto &[v, _] : graph.getNeighbors(AddVertex))
                {
                    CL.insert(v);
                }
            }
            else
            {
                for (const auto &[v, _] : graph.getNeighbors(AddVertex))
                {
                    if (DT.getDominatingVertices().find(v) == DT.getDominatingVertices().end())
                    {
                        CL.insert(v);
                    }
                }
            }

            DT.addVertex(AddVertex);
            update_Dscore(Dscore, AddVertex, DT, graph);
        }

        remove_redundant_vertices(DT, graph);
        connect_minimum_spanning_tree(DT, graph);
        POPinit.push_back(std::make_unique<DominatingTreeSolution>(DT));
        IndiNum--;
    }

    return POPinit;
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>>> floyd_warshall(const Graph &graph)
{
    int num_vertices = graph.getNumVertices();
    std::vector<std::vector<double>> dist(num_vertices, std::vector<double>(num_vertices, std::numeric_limits<double>::max()));
    std::vector<std::vector<int>> next(num_vertices, std::vector<int>(num_vertices, -1));

    for (int i = 0; i < num_vertices; ++i)
    {
        dist[i][i] = 0;
        for (const auto &[neighbor, weight] : graph.getNeighbors(i))
        {
            dist[i][neighbor] = weight;
            next[i][neighbor] = neighbor;
        }
    }

    for (int k = 0; k < num_vertices; ++k)
    {
        for (int i = 0; i < num_vertices; ++i)
        {
            for (int j = 0; j < num_vertices; ++j)
            {
                if (dist[i][k] < std::numeric_limits<double>::max() &&
                    dist[k][j] < std::numeric_limits<double>::max() &&
                    dist[i][k] + dist[k][j] < dist[i][j])
                {
                    dist[i][j] = dist[i][k] + dist[k][j];
                    next[i][j] = next[i][k];
                }
            }
        }
    }

    return {dist, next};
}
std::vector<int> getPath(int u, int v, const std::vector<std::vector<int>> &next)
{
    if (next[u][v] == -1)
    {
        return {}; // There is no path between u and v
    }
    std::vector<int> path = {u};
    while (u != v)
    {
        u = next[u][v];
        path.push_back(u);
    }
    return path;
}

// Checks if there are non-dominated vertices in the graph
bool has_non_dominated_vertices(const std::vector<int> &Dscore)
{
    for (int score : Dscore)
    {
        if (score > 0)
        {
            return true;
        }
    }
    return false;
}

// Updates the Dscore for the neighbors and 2-hop neighbors of the added vertex
void update_Dscore(std::vector<int> &Dscore, int AddVertex, const DominatingTreeSolution &DT, const Graph &graph)
{
    for (const auto &[v, _] : graph.getNeighbors(AddVertex))
    {
        Dscore[v] = compute_Dscore(v, DT, graph);
        for (const auto &[u, _] : graph.getNeighbors(v))
        {
            Dscore[u] = compute_Dscore(u, DT, graph);
        }
    }
}
// Removes redundant vertices from the dominating tree
void remove_redundant_vertices(DominatingTreeSolution &DT, const Graph &graph)
{
    std::unordered_set<int> redundant_vertices;

    for (int vertex : DT.getDominatingVertices())
    {
        bool is_redundant = true;

        for (const auto &[neighbor, _] : graph.getNeighbors(vertex))
        {
            if (DT.getDominatingVertices().find(neighbor) == DT.getDominatingVertices().end())
            {
                is_redundant = false;
                break;
            }
        }

        if (is_redundant)
        {
            redundant_vertices.insert(vertex);
        }
    }

    for (int vertex : redundant_vertices)
    {
        DT.removeVertex(vertex);
    }
}

// Connects vertices in the dominating tree using Prim's algorithm to build a minimum spanning tree
void connect_minimum_spanning_tree(DominatingTreeSolution &DT, const Graph &graph)
{
    std::unordered_set<int> visited;
    std::priority_queue<std::tuple<double, int, int>> edge_queue;

    int start_vertex = *DT.getDominatingVertices().begin();
    visited.insert(start_vertex);

    for (const auto &[neighbor, weight] : graph.getNeighbors(start_vertex))
    {
        if (DT.getDominatingVertices().find(neighbor) != DT.getDominatingVertices().end())
        {
            edge_queue.push({-weight, start_vertex, neighbor});
        }
    }

    while (!edge_queue.empty())
    {
        auto [weight, u, v] = edge_queue.top();
        edge_queue.pop();

        if (visited.find(v) == visited.end())
        {
            DT.addEdge(u, v, -weight);
            visited.insert(v);

            for (const auto &[neighbor, weight] : graph.getNeighbors(v))
            {
                if (DT.getDominatingVertices().find(neighbor) != DT.getDominatingVertices().end() && visited.find(neighbor) == visited.end())
                {
                    edge_queue.push({-weight, v, neighbor});
                }
            }
        }
    }
}

int compute_Dscore(int vertex, const DominatingTreeSolution &solution, const Graph &graph)
{
    const auto &dominating_vertices = solution.getDominatingVertices();
    const auto &neighbors = graph.getNeighbors(vertex);

    if (dominating_vertices.find(vertex) != dominating_vertices.end())
    {
        std::unordered_set<int> only_dominated_by_vertex;
        for (const auto &[u, _] : neighbors)
        {
            if (dominating_vertices.find(u) == dominating_vertices.end())
            {
                only_dominated_by_vertex.insert(u);
            }
        }
        return -1 * static_cast<int>(only_dominated_by_vertex.size());
    }
    else
    {
        std::unordered_set<int> non_dominated_neighbors;
        for (const auto &[u, _] : neighbors)
        {
            if (dominating_vertices.find(u) == dominating_vertices.end())
            {
                non_dominated_neighbors.insert(u);
            }
        }
        return static_cast<int>(non_dominated_neighbors.size());
    }
}

double compute_Wscore(int vertex, const DominatingTreeSolution &solution, const std::vector<std::vector<double>> &shortest_paths)
{
    const auto &dominating_vertices = solution.getDominatingVertices();

    if (dominating_vertices.find(vertex) == dominating_vertices.end())
    {
        double min_shortest_path = std::numeric_limits<double>::infinity();

        for (int v_prime : dominating_vertices)
        {
            double shortest_path = shortest_paths[vertex][v_prime];

            if (shortest_path < min_shortest_path)
            {
                min_shortest_path = shortest_path;
            }
        }

        return min_shortest_path;
    }

    return 0;
}
