#include <iostream>
#include <vector>
#include <algorithm> // for std::max_element
#include <ctime> // for clock() and CLOCKS_PER_SEC

using namespace std;

// Function to initialize the population
vector<vector<int>> InitRCL(Graph G, int IndiNum, double alpha) {
    vector<vector<int>> POP;
    for (int i = 0; i < IndiNum; i++) {
        vector<int> individual = ConstructGreedy(G, alpha);
        POP.push_back(individual);
    }
    return POP;
}

// Function to apply the ITLS algorithm to an individual
vector<int> ITLS(vector<int> individual, Graph G) {
    bool improved = true;
    while (improved) {
        improved = false;
        for (int i = 0; i < individual.size(); i++) {
            for (int j = i+1; j < individual.size(); j++) {
                vector<int> candidate = individual;
                candidate[i] = j;
                if (IsDominating(candidate, G) && Objective(candidate, G) > Objective(individual, G)) {
                    individual = candidate;
                    improved = true;
                    break;
                }
            }
            if (improved) {
                break;
            }
        }
    }
    return individual;
}

// Function to apply the MutationHD operator to an individual
vector<int> MutationHD(vector<int> individual) {
    int i = rand() % individual.size();
    int j = rand() % individual.size();
    while (j == i) {
        j = rand() % individual.size();
    }
    swap(individual[i], individual[j]);
    return individual;
}

// Function to run the hybrid framework combining GA with ITLS
vector<int> GAITLS(Graph G, int cutoff_time, int IndiNum, double alpha) {
    // Step 1: Initialize the population
    vector<vector<int>> POP = InitRCL(G, IndiNum, alpha);

    // Step 2: Set the best solution as the individual with the best objective in POP
    vector<int> DT_star = *max_element(POP.begin(), POP.end(), [G](const vector<int>& a, const vector<int>& b) {
        return Objective(a, G) < Objective(b, G);
    });

    // Step 3: Start the main loop until the elapsed time is less than cutoff time
    clock_t start_time = clock();
    while (((double)(clock() - start_time) / CLOCKS_PER_SEC) < cutoff_time) {
        int index = 0;
        // Step 3.1: Apply the ITLS algorithm to each individual in POP
        for (auto& individual : POP) {
            individual = ITLS(individual, G);
            // Update the best solution if the current individual is better
            if (Objective(individual, G) > Objective(DT_star, G)) {
                DT_star = individual;
            }
            index++;
        }
        index = 0;
        // Step 3.2: Apply the MutationHD operator to each individual in POP
        for (auto& individual : POP) {
            individual = MutationHD(individual);
            // Update the best solution if the current individual is better
            if (Objective(individual, G) > Objective(DT_star, G)) {
                DT_star = individual;
            }
            index++;
        }
    }

    // Step 4: Return the best solution
    return DT_star;
}
