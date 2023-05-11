#include <iostream>
#include <fstream>
#include <sstream>
#include "graph.h"
#include "init_rcl.h"
#include "ITLS.h"

int main() {
    // printf("Running CPU version\n");
    // Timer timer;
    // startTime(&timer);
    
    // Set parameters
    int max_iterations = 100;
    int cutoff_time = 5000;
    int IndiNum = 50;
    double alpha = 0.5;
    double mutationRate = 0.1;

    // Initialize the graph
    // Graph graph(0);
    // readGraphFromFile(graph, "ins_050_1.txt");
    Graph graph(5);
    graph.addEdge(0, 1, 1.0);
    graph.addEdge(1, 2, 2.0);
    graph.addEdge(2, 3, 6.0);
    graph.addEdge(4, 2, 4.0);
    graph.addEdge(4, 5, 5.0);



    // Initialize the population using RCL
    std::vector<std::unique_ptr<DominatingTreeSolution>> POP = init_RCL(graph, IndiNum, alpha);
    
    // Print the initial solutions
    std::cout << "Initial solutions:" << std::endl;
    for (const auto& solution : POP) {
        std::cout << "Total Weight: " << solution->getTotalWeight() << std::endl;
    }

    // Apply the ITLS algorithm
    std::cout << "\nApplying ITLS..." << std::endl;
    for (auto &individual : POP) {
        individual = std::make_unique<DominatingTreeSolution>(ITLS(graph, max_iterations, *individual));
        std::cout << "Total Weight after ITLS: " << individual->getTotalWeight() << std::endl;
    }

    // Apply the GAITLS algorithm
    std::cout << "\nApplying GAITLS..." << std::endl;
    DominatingTreeSolution bestSolution = GAITLS(graph, cutoff_time, IndiNum, alpha, mutationRate);
    std::cout << "Best Total Weight after GAITLS: " << bestSolution.getTotalWeight() << std::endl;

    //stopTime(&timer);
    //printElapsedTime(timer, "    CPU time", CYAN);

    return 0;
}


void readGraphFromFile(Graph &graph, const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file " << filename << std::endl;
        return;
    }

    int numNodes, numEdges;
    file >> numNodes >> numEdges;

    // If Graph has not been initialized yet
    if (graph.getNumVertices() == 0) {
        graph = Graph(numNodes);
    }

    int x, y;
    double w;
    for (int i = 0; i < numEdges; ++i) {
        if (!(file >> x >> y >> w)) {
            std::cerr << "Error reading edge data from file" << std::endl;
            return;
        }
        graph.addEdge(x, y, w);
    }

    file.close();
}