#ifndef LOCALLY_GREEDY_CUDA_H
#define LOCALLY_GREEDY_CUDA_H

#include <vector>
#include "mgraph.h"

int LocallyGreedyCUDA(std::vector<int> &x, const mGraph &g, double &tlen, double &rprate);

#endif
