#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <algorithm>
#include <tuple>

class Graph
{
public:
    Graph(int num_vertices) : num_vertices_(num_vertices)
    {
        adjacency_list_.resize(num_vertices);
    }

    void addEdge(int u, int v, double weight)
    {
        adjacency_list_[u].push_back({v, weight});
        adjacency_list_[v].push_back({u, weight});
    }

    const std::vector<std::pair<int, double>> &getNeighbors(int vertex) const
    {
        return adjacency_list_[vertex];
    }

    int getNumVertices() const
    {
        return num_vertices_;
    }
    int getDegree(int vertex) const
    {
        return adjacency_list_[vertex].size();
    }

private:
    int num_vertices_;
    std::vector<std::vector<std::pair<int, double>>> adjacency_list_;
};
class DominatingTreeSolution
{
public:
    DominatingTreeSolution(const Graph &graph) : graph_(graph) {}

    void addVertex(int vertex)
    {
        dominating_vertices_.insert(vertex);
    }

    void removeVertex(int vertex)
    {
        dominating_vertices_.erase(vertex);
    }

    void addEdge(int u, int v, double weight)
    {
        tree_edges_.push_back({u, v, weight});
        total_weight_ += weight;
    }

    void removeEdge(int u, int v, double weight)
    {
        tree_edges_.erase(std::remove(tree_edges_.begin(), tree_edges_.end(), std::make_tuple(u, v, weight)), tree_edges_.end());
        total_weight_ -= weight;
    }

    const std::unordered_set<int> &getDominatingVertices() const
    {
        return dominating_vertices_;
    }

    const std::vector<std::tuple<int, int, double>> &getTreeEdges() const
    {
        return tree_edges_;
    }

    double getTotalWeight() const
    {
        return total_weight_;
    }

    bool isConnected() const
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

    void DFS(int vertex, std::unordered_set<int> &visited) const
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

    std::vector<std::unordered_set<int>> getDisconnectedComponents() const
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

    std::vector<int> getShortestPathBetweenComponents(const std::vector<std::unordered_set<int>> &components, const std::vector<std::vector<double>> &shortest_paths) const
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

        // Reconstruct the shortest path from min_weight_pair using the shortest_paths matrix
        std::vector<int> shortest_path;
        int u = min_weight_pair.first;
        int v = min_weight_pair.second;

        while (u != v)
        {
            shortest_path.push_back(u);
            for (int k = 0; k < static_cast<int>(shortest_paths.size()); ++k)
            {
                if (shortest_paths[u][k] + shortest_paths[k][v] == shortest_paths[u][v])
                {
                    u = k;
                    break;
                }
            }
        }
        shortest_path.push_back(v);

        return shortest_path;
    }

    void addVerticesAlongPath(const std::vector<int> &path)
    {
        for (const int vertex : path)
        {
            dominating_vertices_.insert(vertex);
        }
    }

    std::unordered_set<int> getPathVertices(const std::vector<int> &path) const
    {
        std::unordered_set<int> path_vertices;
        for (const int vertex : path)
        {
            path_vertices.insert(vertex);
        }
        return path_vertices;
    }

private:
    const Graph &graph_;
    std::unordered_set<int> dominating_vertices_;
    std::vector<std::tuple<int, int, double>> tree_edges_;
    double total_weight_ = 0;
};