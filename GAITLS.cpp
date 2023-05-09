#include <chrono>
#include "init_rcl.h"

bool isBetterThan(const DominatingTreeSolution &solution1, const DominatingTreeSolution &solution2);
DominatingTreeSolution ITLS(DominatingTreeSolution individual, const Graph &graph);


DominatingTreeSolution MutationHD(DominatingTreeSolution individual, const Graph &graph) {
    // Create a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Create a distribution over the vertices
    std::uniform_int_distribution<> dis(0, graph.getNumVertices() - 1);

    // Step 1: Randomly remove vertices until the solution is no longer a dominating set
    while (individual.getDominatingVertices().size() > 0) {
        // Select a random vertex to remove
        int vertex = dis(gen);
        
        // If the vertex is part of the dominating set, remove it
        if (individual.getDominatingVertices().count(vertex) > 0) {
            individual.removeVertex(vertex);
        }

        // Check if the current solution is still a dominating set
        // If it's not, break the loop
        std::unordered_set<int> dominating_vertices = individual.getDominatingVertices();
        for (int v = 0; v < graph.getNumVertices(); ++v) {
            const auto &neighbors = graph.getNeighbors(v);
            bool dominated = false;
            for (const auto &neighbor : neighbors) {
                if (dominating_vertices.count(neighbor.first) > 0) {
                    dominated = true;
                    break;
                }
            }
            if (!dominated) {
                goto end_loop; // If a vertex is not dominated, break the loop
            }
        }
        continue;
    end_loop:;
    }

    // Step 2: Conduct the dominating and connecting phases to repair the solution
    // Assuming we have these functions defined somewhere
    makeFeasible(individual, graph);
    connect_minimum_spanning_tree(individual, graph);

    // Step 3: Return the new solution
    return individual;
}


DominatingTreeSolution GAITLS(const Graph &graph, std::chrono::milliseconds cutoff_time, int IndiNum, double alpha) {
    // Step 1: Initialize the population
    std::vector<DominatingTreeSolution> POP = init_RCL(graph, IndiNum, alpha);
    
    // Step 2: Set the best solution as the individual with the best objective in POP
    DominatingTreeSolution DT_star = *std::min_element(POP.begin(), POP.end(), isBetterThan);
    
    auto start_time = std::chrono::high_resolution_clock::now();

    // Step 3: Start the main loop until the elapsed time is less than cutoff time
    while (std::chrono::high_resolution_clock::now() - start_time < cutoff_time) {
        // Step 3.1: Apply the ITLS algorithm to each individual in POP
        for (int index = 0; index < POP.size(); ++index) {
            POP[index] = ITLS(POP[index], graph);
            // Update the best solution if the current individual is better
            if (isBetterThan(POP[index], DT_star)) {
                DT_star = POP[index];
            }
        }

        // Step 3.2: Apply the MutationHD operator to each individual in POP
        for (int index = 0; index < POP.size(); ++index) {
            POP[index] = MutationHD(POP[index], graph);
            // Update the best solution if the current individual is better
            if (isBetterThan(POP[index], DT_star)) {
                DT_star = POP[index];
            }
        }
    }

    // Step 4: Return the best solution
    return DT_star;
}
