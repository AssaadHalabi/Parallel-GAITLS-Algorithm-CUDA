#include <iostream>
#include <vector>
#include <climits>

#define N 8 // Number of nodes in your graph

bool is_dominating(int* graph, bool* set, int size) {
    std::vector<bool> dominated(size, false);

    // Check each node
    for (int idx = 0; idx < size; ++idx) {
        dominated[idx] = false;

        // If the node is in the set, it's dominated
        if (set[idx]) {
            dominated[idx] = true;
        } else {
            // Check each neighbor to see if it's in the set
            for (int j = 0; j < size; ++j) {
                if (graph[idx * size + j] && set[j]) {
                    dominated[idx] = true;
                    break;
                }
            }
        }
    }

    // Check if all nodes are dominated
    bool allDominated = true;
    for (int idx = 0; idx < size; ++idx) {
        allDominated &= dominated[idx];
    }

    return allDominated;
}

void find_min_dom_set(int* graph, int* min_size, bool* min_set) {
    // Iterate over all possible sets (2^N)
    for (int idx = 0; idx < (1 << N); ++idx) {
        // Generate a candidate set based on the idx
        bool set[N];
        for (int i = 0; i < N; ++i) {
            set[i] = idx & (1 << i);
        }

        // Count the size of the set
        int size = 0;
        for (int i = 0; i < N; ++i) {
            if (set[i]) ++size;
        }

        // Check if the set is dominating
        if (is_dominating(graph, set, N) && size < *min_size) {
            // If it is, and it's smaller than the current minimum, update the minimum
            *min_size = size;

            // Update the min_set
            for (int i = 0; i < N; ++i) {
                min_set[i] = set[i];
            }
        }
    }
}

int main() {
    int graph[N * N]; // Your graph represented as a flattened adjacency matrix
    int min_size = INT_MAX; // Initially, the smallest dominating set could be all nodes
    bool min_set[N] = {}; // Initially, no node is in the set

    // TODO: Initialize graph with your actual graph data

    // Find the minimum dominating set
    find_min_dom_set(graph, &min_size, min_set);

    // Print the result
    std::cout << "Minimum dominating set size: " << min_size << std::endl;
    std::cout << "Minimum dominating set: ";
    for (int i = 0; i < N; ++i) {
        if (min_set[i]) std::cout << i << " ";
    }
    std::cout << std::endl;

    return 0;
}
