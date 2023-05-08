
#include <iostream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <limits>
#include <random>
#include <queue>
#include <tuple>
#include "graph.h"



std::vector<DominatingTreeSolution> init_RCL(const Graph &graph, int IndiNum, double alpha)
{
    int num_vertices = graph.getNumVertices();
    std::vector<DominatingTreeSolution> POPinit;

    // Compute the shortest path for each vertex pair
    std::vector<std::vector<double>> shortest_paths = floyd_warshall(graph);

    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    while (IndiNum > 0)
    {
        DominatingTreeSolution DT(graph);
        std::unordered_set<int> CL;

        std::vector<int> Dscore(num_vertices);
        for (int v = 0; v < num_vertices; ++v)
        {
            Dscore[v] = static_cast<int>(graph.getNeighbors(v).size());
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
                    double Wuv = min_weight(graph, v);
                    double score = Dscore[v] / Wuv;

                    maxscore = std::max(maxscore, score);
                    minscore = std::min(minscore, score);
                }

                for (int v : CL)
                {
                    double Wuv = min_weight(graph, v);
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
        connect_minimum_spanning_tree(DT, graph, shortest_paths);
        POPinit.push_back(DT);
        IndiNum--;
    }

    return POPinit;
}

std::vector<std::vector<double>> floyd_warshall(const Graph &graph)
{
    int num_vertices = graph.getNumVertices();
    std::vector<std::vector<double>> dist(num_vertices, std::vector<double>(num_vertices, std::numeric_limits<double>::max()));

    for (int i = 0; i < num_vertices; ++i)
    {
        dist[i][i] = 0;
        for (const auto &[neighbor, weight] : graph.getNeighbors(i))
        {
            dist[i][neighbor] = weight;
        }
    }

    for (int k = 0; k < num_vertices; ++k)
    {
        for (int i = 0; i < num_vertices; ++i)
        {
            for (int j = 0; j < num_vertices; ++j)
            {
                if (dist[i][k] != std::numeric_limits<double>::max() && dist[k][j] != std::numeric_limits<double>::max() && dist[i][k] + dist[k][j] < dist[i][j])
                {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    return dist;
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

// Finds the minimum weight of edges incident to vertex v
double min_weight(const Graph &graph, int vertex)
{
    double min_weight = std::numeric_limits<double>::infinity();
    for (const auto &[v, weight] : graph.getNeighbors(vertex))
    {
        min_weight = std::min(min_weight, weight);
    }
    return min_weight;
}

// Updates the Dscore for the neighbors and 2-hop neighbors of the added vertex
void update_Dscore(std::vector<int> &Dscore, int AddVertex, const DominatingTreeSolution &DT, const Graph &graph)
{
    for (const auto &[neighbor, _] : graph.getNeighbors(AddVertex))
    {
        if (DT.getDominatingVertices().find(neighbor) == DT.getDominatingVertices().end())
        {
            Dscore[neighbor]--;

            for (const auto &[two_hop_neighbor, _] : graph.getNeighbors(neighbor))
            {
                if (DT.getDominatingVertices().find(two_hop_neighbor) == DT.getDominatingVertices().end())
                {
                    Dscore[two_hop_neighbor]--;
                }
            }
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
void connect_minimum_spanning_tree(DominatingTreeSolution &DT, const Graph &graph, const std::vector<std::vector<double>> &shortest_paths)
{
    std::unordered_set<int> visited;
    std::priority_queue<std::tuple<double, int, int>> edge_queue;

    int start_vertex = *DT.getDominatingVertices().begin();
    visited.insert(start_vertex);

    for (int neighbor : DT.getDominatingVertices())
    {
        if (neighbor != start_vertex)
        {
            edge_queue.push({-shortest_paths[start_vertex][neighbor], start_vertex, neighbor});
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

            for (int neighbor : DT.getDominatingVertices())
            {
                if (visited.find(neighbor) == visited.end())
                {
                    edge_queue.push({-shortest_paths[v][neighbor], v, neighbor});
                }
            }
        }
    }
}
