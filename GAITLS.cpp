#include "GAITLS.h"

DominatingTreeSolution MutationHD(const DominatingTreeSolution &solution, double mutationRate, const Graph &graph)
{
    // Copy the solution
    DominatingTreeSolution mutatedSolution = solution;

    // Get the dominating vertices
    std::unordered_set<int> dominatingVertices = mutatedSolution.getDominatingVertices();

    // Calculate the number of vertices to remove
    int numVerticesToRemove = mutationRate * dominatingVertices.size();

    // Create Dscore and tabu list
    std::vector<int> Dscore(graph.getNumVertices(), 0);
    std::unordered_set<int> tabu_list;

    // Removing Phase
    for (int i = 0; i < numVerticesToRemove; ++i)
    {
        removingPhase(mutatedSolution, tabu_list, Dscore, graph, graph.getNumVertices());
    }

    // Repair the solution
    // 1. Dominating phase
    dominatingPhase(mutatedSolution, tabu_list, Dscore, graph, graph.getNumVertices());

    // 2. Connecting phase
    connectingPhase(mutatedSolution, tabu_list, Dscore, graph);

    return mutatedSolution;
}

DominatingTreeSolution GAITLS(const Graph &graph, int cutoff_time, int IndiNum, double alpha, double mutationRate)
{
    // Step 1: Initialize the population
    std::vector<DominatingTreeSolution *> POP = init_RCL(graph, IndiNum, alpha);

    // Step 2: Set the best solution as the individual with the best objective in POP
    auto best_solution_it = std::min_element(POP.begin(), POP.end(),
                                             [](const DominatingTreeSolution *a, const DominatingTreeSolution *b)
                                             {
                                                 return a->getTotalWeight() < b->getTotalWeight();
                                             });
    DominatingTreeSolution *DT_star = new DominatingTreeSolution(*(*best_solution_it));

    auto start_time = std::clock();
    double elapsed_time = 0;

    // Step 3: Start the main loop until the elapsed time is less than cutoff time
    while (elapsed_time < cutoff_time)
    {

        // Step 3.1: Apply the ITLS algorithm to each individual in POP
        for (auto &individual : POP)
        {
            individual = new DominatingTreeSolution(ITLS(graph, cutoff_time, *individual));

            // Update the best solution if the current individual is better
            if (individual->getTotalWeight() < DT_star->getTotalWeight())
            {
                delete DT_star; // Clean up old memory
                DT_star = new DominatingTreeSolution(*individual);
            }
        }

        // Step 3.2: Apply the MutationHD operator to each individual in POP
        for (auto &individual : POP)
        {
            individual = new DominatingTreeSolution(MutationHD(*individual, mutationRate, graph));

            // Update the best solution if the current individual is better
            if (individual->getTotalWeight() < DT_star->getTotalWeight())
            {
                delete DT_star; // Clean up old memory
                DT_star = new DominatingTreeSolution(*individual);
            }
        }

        elapsed_time = (std::clock() - start_time) / static_cast<double>(CLOCKS_PER_SEC);
    }

    // Step 4: Return the best solution
    DominatingTreeSolution final_solution = *DT_star;
    delete DT_star; // Clean up memory

    // Clean up memory for all solutions in POP
    for (auto &individual : POP)
    {
        delete individual;
    }

    return final_solution;
}
