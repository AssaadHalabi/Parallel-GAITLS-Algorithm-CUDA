#include <unordered_set>
#include <vector>
#include <limits>

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
