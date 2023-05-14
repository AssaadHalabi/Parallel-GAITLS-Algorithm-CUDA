#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <ctime>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cuda_runtime.h>
#include "mgraph.h"


__device__ int mceil(int x)
{
    int a, b;
    a = x / 2;
    b = 2 * a;
    if (x > b)
        a++;
    return a;
}
struct mynode
{
    int index, degree;
};
struct mycmp1
{
    __host__ __device__ bool operator()(const mynode &a, const mynode &b) const
    {
        return a.degree < b.degree;
    }
};
__global__ void init_arrays(int *Delta, int *unsat, int *fixThre, int g_MAX_INDEX_OF_NODE, Edge *V, int *sumnd)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < g_MAX_INDEX_OF_NODE)
    {
        Edge *pH = V[idx].headEdge;
        if (pH != NULL)
        {
            int da = V[idx].neighNum;
            int db = mceil(da);
            Delta[idx] = db;
            fixThre[idx] = db;
            unsat[idx] = da;
            atomicAdd(sumnd, db);
        }
    }
}

int LocallyGreedyCUDA(std::vector<int> &x, const mGraph &g, double &tlen, double &rprate)
{
    using namespace std;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start timer
    cudaEventRecord(start);

    // NOTE:
    //----1. A "positive" node refers to the node that is included by a (partial) PIDS.
    //----2. A node is said "satisfied" if more than half of its neighbors are "positive", and otherwise, this node is ragared as "unsatisfied".

    // Delta(v): the number of positive neighbors that node v lacked to be satisfied, which will decrease as the procedure runs
    // unsat(v): the number of node v's unsatisfied neighbors
    // fixThre(v): the number of positive neighbors that node v needs to be satisfied, which is a fixed threshold

    // sumnd: the sum of every node's Delta(), which will decrease as the procedure runs
    // s: the node set of the input graph g
    int i, j;
    set<int> s;

    Edge *pH;

    int da, db;

    // Allocate and initialize host vectors
    std::vector<int> Delta(g.MAX_INDEX_OF_NODE, 0);
    std::vector<int> unsat(g.MAX_INDEX_OF_NODE, 0);
    std::vector<int> fixThre(g.MAX_INDEX_OF_NODE, 0);
    int sumnd = 0;

    // Allocate device vectors and variable
    thrust::device_vector<int> d_Delta = Delta;
    thrust::device_vector<int> d_unsat = unsat;
    thrust::device_vector<int> d_fixThre = fixThre;
    thrust::device_vector<int> d_sumnd(1, 0);

    // Invoke the CUDA kernel
    init_arrays<<<(g.MAX_INDEX_OF_NODE + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(d_Delta.data()),
        thrust::raw_pointer_cast(d_unsat.data()),
        thrust::raw_pointer_cast(d_fixThre.data()),
        g.MAX_INDEX_OF_NODE, g.V,
        thrust::raw_pointer_cast(d_sumnd.data()));
    cudaCheckError();

    // Ensure all GPU computations are finished before proceeding
    cudaDeviceSynchronize();

    // Copy data from device to host
    Delta = d_Delta;
    unsat = d_unsat;
    fixThre = d_fixThre;
    // Copy back the sumnd result from device to host
    sumnd = d_sumnd[0];

    const int len = s.size();
    std::vector<mynode> h_T(len);
    int id;
    set<int>::iterator spointer = s.begin();
    set<int>::iterator epointer = s.end();
    for (i = 0; spointer != epointer; spointer++, i++)
    {
        id = *spointer;
        h_T[i] = mynode(id, g.V[id].neighNum);
    }
    // sort the nodes by degree in ascending order
    thrust::device_vector<mynode> d_T = h_T;
    thrust::sort(d_T.begin(), d_T.end(), mycmp1());
    thrust::copy(d_T.begin(), d_T.end(), h_T.begin());

    // If you need the data back in your original array:
    mynode T[len];
    std::copy(h_T.begin(), h_T.end(), T);

    // res: the (partial) PIDS, which will be inserted with a number of nodes one by one  as the procedure runs and becomes a valid PIDS in the end
    set<int> res;
    set<int>::iterator sp2;
    set<int>::iterator ep2;

    // based on the order computed before, check every node one by one to confirm whether current node is satisfied or not
    // if the current node v is satisfied, then skip the following processing to the check towards the next node
    // otherwise, based on a certain strategy, add v's neighbors into the partial PIDS res one by one until v becomes satisfied

    // the for loop terminates when every node has been checked or the total demand has decreased to zero
    for (i = 0; i < len && sumnd > 0; i++)
    {
        int vi = T[i].index;
        int R = Delta[vi];
        if (R > 0)
        {
            pH = g.V[vi].headEdge;
            set<int> nid;
            // construct a candidators set by absorbing vi's neighbors that haven't been included by the (partial) PIDS res
            while (pH != NULL)
            {
                nb = pH->edVertex;
                if (res.find(nb) == res.end())
                    nid.insert(nb);
                pH = pH->next;
            }

            int nb2;
            int maxunsat;
            int w, u, x, y;
            // select R vi's neighbors from the candidators set and insert them into the (partial) PIDS res one by one
            for (j = 0; j < R; j++)
            {
                spointer = nid.begin();
                epointer = nid.end();
                // select the node with maximum unsat(), if there are several nodes sharing the same maximum unsat(), then choose the node with maximum ID
                for (maxunsat = 0; spointer != epointer; spointer++)
                {
                    w = *spointer;
                    if (unsat[w] >= maxunsat)
                    {
                        maxunsat = unsat[w];
                        u = w;
                    }
                }
                res.insert(u);
                // decrease the total demand sumnd by unsat[u]
                sumnd -= unsat[u];
                // update Delata() and unsat() of relevant nodes
                pH = g.V[u].headEdge;
                while (pH != NULL)
                {
                    x = pH->edVertex;
                    // only when Delta(x)>0, the addition of u contributes to reducing the demand of x
                    if (Delta[x] > 0)
                    {
                        Delta[x]--;
                        // only when x's Delta() decreases from 1 to 0 can further lead relevant nodes' unsat() to decrease
                        if (Delta[x] == 0)
                        {
                            Edge *pH2 = g.V[x].headEdge;
                            while (pH2 != NULL)
                            {
                                y = pH2->edVertex;
                                unsat[y]--;
                                pH2 = pH2->next;
                            }
                        }
                    }
                    pH = pH->next;
                }
                // remove the selected node u from the candidators set
                nid.erase(u);
            }
            nid.clear();
        }
    }

    spointer = res.begin();
    epointer = res.end();
    for (; spointer != epointer; spointer++)
    {
        x.push_back(*spointer);
    }
    // stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    // Calculate elapsed time in milliseconds
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Convert elapsed time to seconds and assign to tlen
    tlen = milliseconds / 1000.0;

    // check whether the result is a valid PIDS
    int signal = 1;
    // rrate: |the generated PIDS|/|the node set of the input graph|
    // aprate: the average positive rate of the nodes in the graph
    double rrate, aprate;
    // naV: the number of the current node's positive neighbors
    int naV;
    aprate = 0;
    spointer = s.begin();
    epointer = s.end();
    id = *spointer;
    for (; spointer != epointer; spointer++)
    {
        id = *spointer;
        naV = fixThre[*spointer] - Delta[*spointer];
        aprate += 1.00 * naV / g.V[id].neighNum;
        if (naV < g.V[id].neighNum / 2)
        {
            signal = 0;
            break;
        }
    }
    aprate = aprate / len * 100;
    if (signal == 0)
        cout << "NOT satisfied!"
             << "(" << id << ")" << endl;
    else
        cout << "Satisfied!" << endl;
    cout << "the result rate: " << aprate << endl;
    cout << "average positive rate: " << aprate << endl;
    rprate = aprate;

    // Free the device memory after we're done
    // No need to free the host memory as std::vector does it automatically

    // Destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return signal;
}
