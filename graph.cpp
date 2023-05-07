#include <vector>
#include <unordered_map>

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

private:
    int num_vertices_;
    std::vector<std::vector<std::pair<int, double>>> adjacency_list_;
};
class DominatingTreeSolution
{
public:
    DominatingTreeSolution(const Graph& graph) : graph_(graph) {}

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

    const std::unordered_set<int>& getDominatingVertices() const
    {
        return dominating_vertices_;
    }

    const std::vector<std::tuple<int, int, double>>& getTreeEdges() const
    {
        return tree_edges_;
    }

    double getTotalWeight() const
    {
        return total_weight_;
    }

private:
    const Graph& graph_;
    std::unordered_set<int> dominating_vertices_;
    std::vector<std::tuple<int, int, double>> tree_edges_;
    double total_weight_ = 0;
};