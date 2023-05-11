#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <algorithm>
#include <tuple>
#include "graph.h"

Graph::Graph(int num_vertices) : num_vertices_(num_vertices)
{
    adjacency_list_.resize(num_vertices);
}

void Graph::addEdge(int u, int v, double weight)
{
    adjacency_list_[u].push_back({v, weight});
    adjacency_list_[v].push_back({u, weight});
}

const std::vector<std::pair<int, double>> &Graph::getNeighbors(int vertex) const
{
    return adjacency_list_[vertex];
}

int Graph::getNumVertices() const
{
    return num_vertices_;
}
int Graph::getDegree(int vertex) const
{
    return adjacency_list_[vertex].size();
}

DominatingTreeSolution::DominatingTreeSolution(const Graph &graph) : graph_(graph) {}

void DominatingTreeSolution::addVertex(int vertex)
{
    dominating_vertices_.insert(vertex);
}

void DominatingTreeSolution::removeVertex(int vertex)
{
    dominating_vertices_.erase(vertex);
}

void DominatingTreeSolution::addEdge(int u, int v, double weight)
{
    tree_edges_.push_back({u, v, weight});
    total_weight_ += weight;
}

void DominatingTreeSolution::removeEdge(int u, int v, double weight)
{
    tree_edges_.erase(std::remove(tree_edges_.begin(), tree_edges_.end(), std::make_tuple(u, v, weight)), tree_edges_.end());
    total_weight_ -= weight;
}

const std::unordered_set<int> &DominatingTreeSolution::getDominatingVertices() const
{
    return dominating_vertices_;
}

const std::vector<std::tuple<int, int, double>> &DominatingTreeSolution::getTreeEdges() const
{
    return tree_edges_;
}

double DominatingTreeSolution::getTotalWeight() const
{
    return total_weight_;
}

bool DominatingTreeSolution::isConnected() const
{
    if (dominating_vertices_.empty())
    {
        return false;
    }

    std::unordered_set<int> visited;
    int start_vertex = *dominating_vertices_.begin();
    DFS(start_vertex, visited);

    return visited.size() == dominating_vertices_.size();
}

void DominatingTreeSolution::DFS(int vertex, std::unordered_set<int> &visited) const
{
    visited.insert(vertex);

    for (const auto &[neighbor, weight] : graph_.getNeighbors(vertex))
    {
        if (dominating_vertices_.count(neighbor) > 0 && visited.count(neighbor) == 0)
        {
            DFS(neighbor, visited);
        }
    }
}

std::vector<std::unordered_set<int>> DominatingTreeSolution::getDisconnectedComponents() const
{
    std::vector<std::unordered_set<int>> components;
    std::unordered_set<int> unvisited(dominating_vertices_.begin(), dominating_vertices_.end());

    while (!unvisited.empty())
    {
        std::unordered_set<int> visited;
        int start_vertex = *unvisited.begin();
        DFS(start_vertex, visited);

        components.push_back(visited);

        for (const int visited_vertex : visited)
        {
            unvisited.erase(visited_vertex);
        }
    }

    return components;
}

std::vector<int> DominatingTreeSolution::getShortestPathBetweenComponents(const std::vector<std::unordered_set<int>> &components, const std::vector<std::vector<double>> &shortest_paths, const std::vector<std::vector<int>> &next) const
{
    double min_weight = std::numeric_limits<double>::infinity();
    std::pair<int, int> min_weight_pair;

    for (size_t i = 0; i < components.size(); ++i)
    {
        for (const int u : components[i])
        {
            for (size_t j = i + 1; j < components.size(); ++j)
            {
                for (const int v : components[j])
                {
                    if (shortest_paths[u][v] < min_weight)
                    {
                        min_weight = shortest_paths[u][v];
                        min_weight_pair = {u, v};
                    }
                }
            }
        }
    }

    // Reconstruct the shortest path from min_weight_pair using the next matrix
    std::vector<int> shortest_path;
    int u = min_weight_pair.first;
    int v = min_weight_pair.second;

    if (next[u][v] == -1)
    {
        return {}; // There is no path between u and v
    }
    while (u != v)
    {
        shortest_path.push_back(u);
        u = next[u][v];
    }
    shortest_path.push_back(v);

    return shortest_path;
}

void DominatingTreeSolution::addVerticesAlongPath(const std::vector<int> &path)
{
    for (const int vertex : path)
    {
        dominating_vertices_.insert(vertex);
    }
}

std::unordered_set<int> DominatingTreeSolution::getPathVertices(const std::vector<int> &path) const
{
    std::unordered_set<int> path_vertices;
    for (const int vertex : path)
    {
        path_vertices.insert(vertex);
    }
    return path_vertices;
}
bool DominatingTreeSolution::isDominatingSet() const
{
    // Create a set of all vertices in the graph
    std::unordered_set<int> allVertices;
    for (int i = 0; i < graph_.getNumVertices(); ++i)
    {
        allVertices.insert(i);
    }

    // Remove all dominating vertices from the set
    for (const int vertex : dominating_vertices_)
    {
        allVertices.erase(vertex);
    }

    // Check each remaining vertex
    for (const int vertex : allVertices)
    {
        bool isDominated = false;

        // Get all neighbors of the current vertex
        const std::vector<std::pair<int, double>> &neighbors = graph_.getNeighbors(vertex);

        // If one of the neighbors is in the dominating set, this vertex is dominated
        for (const std::pair<int, double> &neighbor : neighbors)
        {
            if (dominating_vertices_.count(neighbor.first))
            {
                isDominated = true;
                break;
            }
        }

        // If the vertex is not dominated, return false
        if (!isDominated)
        {
            return false;
        }
    }

    // If all vertices are dominated, return true
    return true;
}
