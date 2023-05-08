#ifndef INIT_RCL_H
#define INIT_RCL_H

#include <iostream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <limits>
#include <random>
#include <queue>
#include <tuple>
#include "graph.h"

std::vector<DominatingTreeSolution> init_RCL(const Graph &graph, int IndiNum, double alpha);
std::vector<std::vector<double>> floyd_warshall(const Graph &graph);
bool has_non_dominated_vertices(const std::vector<int> &Dscore);
void update_Dscore(std::vector<int> &Dscore, int AddVertex, const DominatingTreeSolution &DT, const Graph &graph);
void remove_redundant_vertices(DominatingTreeSolution &DT, const Graph &graph);
void connect_minimum_spanning_tree(DominatingTreeSolution &DT, const Graph &graph);
int compute_Dscore(int vertex, const DominatingTreeSolution &solution, const Graph &graph);
double compute_Wscore(int vertex, const DominatingTreeSolution &solution, const std::vector<std::vector<double>> &shortest_paths);

#endif // INIT_RCL_H
