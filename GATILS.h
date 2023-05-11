#ifndef GAITLS_H
#define GAITLS_H

#include "graph.h"
#include "init_rcl.h"
#include "ITLS.h"
#include <vector>
#include <unordered_set>
#include <memory>
#include <algorithm>
#include <ctime>

DominatingTreeSolution GAITLS(const Graph &graph, int cutoff_time, int IndiNum, double alpha, double mutationRate);
DominatingTreeSolution MutationHD(const DominatingTreeSolution &solution, double mutationRate, const Graph &graph);

#endif
