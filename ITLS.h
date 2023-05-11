#ifndef DOMINATING_TREE_SOLUTION_H
#define DOMINATING_TREE_SOLUTION_H

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <memory>
#include <limits>
#include <algorithm>
#include <ctime>
#include "init_RCL.h"

class Graph;                  // Forward declaration
class DominatingTreeSolution; // Forward declaration

int compute_Dscore(int v, const DominatingTreeSolution &DT, const Graph &graph);
bool has_non_dominated_vertices(const std::vector<int> &Dscore);
void remove_redundant_vertices(DominatingTreeSolution &DT, const Graph &graph);
void connect_minimum_spanning_tree(DominatingTreeSolution &DT, const Graph &graph);
void update_Dscore(std::vector<int> &Dscore, int vertex, const DominatingTreeSolution &DT, const Graph &graph);
double compute_Wscore(int v, const DominatingTreeSolution &DT, const std::vector<std::vector<int>> &shortest_paths);
void removingPhase(DominatingTreeSolution &DT, std::unordered_set<int> &tabu_list, std::vector<int> &Dscore, const Graph &graph, int num_vertices);
void connectingPhase(DominatingTreeSolution &DT, std::unordered_set<int> &tabu_list, std::vector<int> &Dscore, const Graph &graph);
void dominatingPhase(DominatingTreeSolution &DT, std::unordered_set<int> &tabu_list, std::vector<int> &Dscore, const Graph &graph, int num_vertices);

DominatingTreeSolution ITLS(const Graph &graph, int cutoff_time, DominatingTreeSolution DT);

#endif // DOMINATING_TREE_SOLUTION_H
