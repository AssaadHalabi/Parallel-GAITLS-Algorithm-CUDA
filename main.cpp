#include <iostream>
#include <fstream>
#include <sstream>
#include "GAITLS.h"
Graph readGraphFromFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Unable to open file " << filename << std::endl;
        // You could potentially throw an exception here or return an empty Graph
        return Graph(0);
    }

    int numNodes, numEdges;
    file >> numNodes >> numEdges;

    // Create the graph
    Graph graph(numNodes);

    int x, y;
    double w;
    for (int i = 0; i < numEdges; ++i)
    {
        if (!(file >> x >> y >> w))
        {
            std::cerr << "Error reading edge data from file" << std::endl;
            // You could potentially throw an exception here or return the partially completed Graph
            return graph;
        }
        graph.addEdge(x, y, w);
    }

    file.close();

    return graph;
}

int main()
{
    // Set parameters
    int max_iterations = 10000;
    int cutoff_time = 5000;
    int IndiNum = 50;
    double alpha = 0.9;
    double mutationRate = 0.5;

    // Initialize the graph
    Graph graph = readGraphFromFile("ins_500_3.txt");

    // Initialize the population using RCL
    std::vector<DominatingTreeSolution *> POP = init_RCL(graph, IndiNum, alpha);

    // Print the initial solutions
    std::cout << "Initial solutions:" << std::endl;
    for (const auto &solution : POP)
    {
        std::cout << "Total Weight: " << solution->getTotalWeight() << std::endl;
    }

    // Apply the ITLS algorithm
    std::cout << "\nApplying ITLS..." << std::endl;
    for (auto &individual : POP)
    {
        DominatingTreeSolution *old_individual = individual;
        individual = new DominatingTreeSolution(ITLS(graph, max_iterations, *old_individual));
        delete old_individual;
        std::cout << "Total Weight after ITLS: " << individual->getTotalWeight() << std::endl;
    }

    // Apply the GAITLS algorithm
    std::cout << "\nApplying GAITLS..." << std::endl;
    DominatingTreeSolution bestSolution = GAITLS(graph, cutoff_time, IndiNum, alpha, mutationRate);
    std::cout << "Best Total Weight after GAITLS: " << bestSolution.getTotalWeight() << std::endl;

    // Clean up memory
    for (auto &individual : POP)
    {
        delete individual;
    }

    return 0;
}
