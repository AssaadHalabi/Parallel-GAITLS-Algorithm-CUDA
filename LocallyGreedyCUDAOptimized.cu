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
#include <cuda_runtime.h>
#include <stdio.h>
#include "mgraph.h"

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

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
__global__ void init_arrays(int *Delta, int *unsat, int *fixThre, int g_MAX_INDEX_OF_NODE, Vertex *V, int *sumnd)
{
    __shared__ int shared_Delta[256];
    __shared__ int shared_unsat[256];
    __shared__ int shared_fixThre[256];
    __shared__ int shared_neighNum[256];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < g_MAX_INDEX_OF_NODE)
    {
        Edge *pH = V[idx].headEdge;
        if (pH != NULL)
        {
            // Load V[idx].neighNum into shared memory
            shared_neighNum[threadIdx.x] = V[idx].neighNum;
            __syncthreads();

            // Use shared_neighNum[threadIdx.x] instead of V[idx].neighNum
            int da = shared_neighNum[threadIdx.x];
            int db = mceil(da);
            shared_Delta[threadIdx.x] = db;
            shared_fixThre[threadIdx.x] = db;
            shared_unsat[threadIdx.x] = da;
            atomicAdd(sumnd, db);
            
        }
    }

    __syncthreads();

    if (idx < g_MAX_INDEX_OF_NODE)
    {
        Delta[idx] = shared_Delta[threadIdx.x];
        unsat[idx] = shared_unsat[threadIdx.x];
        fixThre[idx] = shared_fixThre[threadIdx.x];
    }
}

__global__ void update_values(int *Delta, int *unsat, Vertex *V, int u)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Iterate over all neighbors x of u
    Edge *pH = V[u].headEdge;
    while (pH != NULL)
    {
        int x = pH->edVertex;
        if (Delta[x] > 0)
        {
            atomicSub(&Delta[x], 1);
            if (Delta[x] == 0)
            {
                Edge *pH2 = V[x].headEdge;
                while (pH2 != NULL)
                {
                    int y = pH2->edVertex;
                    atomicSub(&unsat[y], 1);
                    pH2 = pH2->next;
                }
            }
        }
        pH = pH->next;
    }
}

__global__ void find_max_unsat(int *unsat, int *nid, int nid_size, int *max_unsat, int *u)
{
    __shared__ int sdata[256];
    __shared__ int sindex[256];

    // Load data into shared memory
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nid_size)
    {
        sdata[threadIdx.x] = unsat[nid[i]];
        sindex[threadIdx.x] = nid[i];
    }
    else
    {
        sdata[threadIdx.x] = INT_MIN;
        sindex[threadIdx.x] = -1;
    }
    __syncthreads();

    // Perform parallel reduction to find max value and index
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            if (sdata[threadIdx.x] < sdata[threadIdx.x + s])
            {
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
                sindex[threadIdx.x] = sindex[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (threadIdx.x == 0)
    {
        atomicMax(max_unsat, sdata[0]);
        if (*max_unsat == sdata[0])
        {
            *u = sindex[0];
        }
    }
}

int LocallyGreedyCUDAOptimized(std::vector<int> &x, const mGraph &g, double &tlen, double &rprate)
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

    // Create a device vector for the T vector
    thrust::device_vector<mynode> d_T = h_T;

    // Invoke the CUDA kernel
    init_arrays<<<(g.MAX_INDEX_OF_NODE + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(d_Delta.data()),
        thrust::raw_pointer_cast(d_unsat.data()),
        thrust::raw_pointer_cast(d_fixThre.data()),
        g.MAX_INDEX_OF_NODE, g.V,
        thrust::raw_pointer_cast(d_sumnd.data()));
    cudaError_t err1 = cudaGetLastError(); // get the last error that occurred
    cudaCheckError(err1);

    // Ensure all GPU computations are finished before proceeding
    cudaDeviceSynchronize();

    // Copy data from device to host
    thrust::copy(d_Delta.begin(), d_Delta.end(), Delta.begin());
    thrust::copy(d_unsat.begin(), d_unsat.end(), unsat.begin());
    thrust::copy(d_fixThre.begin(), d_fixThre.end(), fixThre.begin());
    sumnd = d_sumnd[0];

    // sort the nodes by degree in ascending order
    thrust::sort(d_T.begin(), d_T.end(), mycmp1());
    thrust::copy(d_T.begin(), d_T.end(), h_T.begin());

    // If you need the data back in your original array:
    mynode T[len];
    std::copy(h_T.begin(), h_T.end(), T);

    // res: the (partial) PIDS, which will be inserted with a number of nodes one by one  as the procedure runs and becomes a valid PIDS in the end
    set<int> res;
    

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
                // Copy nid set to device vector
                thrust::device_vector<int> d_nid(nid.size());
                thrust::copy(nid.begin(), nid.end(), d_nid.begin());

                // Allocate device memory for max_unsat and u
                thrust::device_vector<int> d_max_unsat(1, INT_MIN);
                thrust::device_vector<int> d_u(1);

                // Invoke kernel to find max_unsat and u
                find_max_unsat<<<(nid.size() + 255) / 256, 256>>>(
                    thrust::raw_pointer_cast(d_unsat.data()),
                    thrust::raw_pointer_cast(d_nid.data()),
                    nid.size(),
                    thrust::raw_pointer_cast(d_max_unsat.data()),
                    thrust::raw_pointer_cast(d_u.data()));
                cudaError_t err2 = cudaGetLastError(); // get the last error that occurred
                cudaCheckError(err2);

                // Copy result back to host
                int u = d_u[0];

                res.insert(u);
                sumnd -= unsat[u];
                // Update Delta(x) and unsat(y)
                update_values<<<1, 1>>>(
                    thrust::raw_pointer_cast(d_Delta.data()),
                    thrust::raw_pointer_cast(d_unsat.data()),
                    g.V,
                    u);
                cudaError_t err3 = cudaGetLastError(); // get the last error that occurred
                cudaCheckError(err3);
                // remove the selected node u from the candidators set
                nid.erase(u);
            }
            nid.clear();
        }
    }
    set<int>::iterator sp2;
    set<int>::iterator ep2;
    sp2 = res.begin();
    ep2 = res.end();
    for (; sp2 != ep2; sp2++)
    {
        x.push_back(*sp2);
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
    sp2 = s.begin();
    ep2 = s.end();
    id = *sp2;
    for (; sp2 != ep2; sp2++)
    {
        id = *sp2;
        naV = fixThre[*sp2] - Delta[*sp2];
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
