#include <iostream>
#include <fstream>
#include <sstream>
#include "GAITLS.h"
#include "graph.h"
#include "ITLS.h"
#include "init_RCL.h"

int main()
{
    // Set parameters
    int max_iterations = 100;
    int cutoff_time = 5000;
    int IndiNum = 50;
    double alpha = 0.5;
    double mutationRate = 0.1;

    // Initialize the graph
    Graph graph(5);
    graph.addEdge(0, 1, 1.0);
    graph.addEdge(1, 2, 2.0);
    graph.addEdge(2, 3, 6.0);
    graph.addEdge(4, 2, 4.0);
    graph.addEdge(4, 5, 5.0);

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
