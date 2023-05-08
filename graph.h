#ifndef GRAPH_DOMINATING_TREE_H
#define GRAPH_DOMINATING_TREE_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <tuple>

class Graph
{
public:
    Graph(int num_vertices);

    void addEdge(int u, int v, double weight);
    const std::vector<std::pair<int, double>> &getNeighbors(int vertex) const;
    int getNumVertices() const;

private:
    int num_vertices_;
    std::vector<std::vector<std::pair<int, double>>> adjacency_list_;
};

class DominatingTreeSolution
{
public:
    DominatingTreeSolution(const Graph& graph);

    void addVertex(int vertex);
    void removeVertex(int vertex);
    void addEdge(int u, int v, double weight);
    void removeEdge(int u, int v, double weight);
    const std::unordered_set<int>& getDominatingVertices() const;
    const std::vector<std::tuple<int, int, double>>& getTreeEdges() const;
    double getTotalWeight() const;

private:
    const Graph& graph_;
    std::unordered_set<int> dominating_vertices_;
    std::vector<std::tuple<int, int, double>> tree_edges_;
    double total_weight_ = 0;
};

#endif // GRAPH_DOMINATING_TREE_H
