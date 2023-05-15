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

__device__ int mceil_device(int x)
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

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < g_MAX_INDEX_OF_NODE)
    {
        Edge *pH = V[idx].headEdge;
        if (pH != NULL)
        {
            int da = V[idx].neighNum;
            int db = mceil_device(da);
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

int LocallyGreedyCUDA(std::vector<int> &x, const mGraph &g, double &tlen, double &rprate)
{
    using namespace std;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int i, j;
    set<int> s;

    Edge *pH;
    int da, db;
    int nb, id;
    // populate s with all nodes
    for (i = 0; i < g.MAX_INDEX_OF_NODE; i++)
    {
        pH = g.V[i].headEdge;
        if (pH != NULL)
            s.insert(i);
    }
    std::vector<int> Delta(g.MAX_INDEX_OF_NODE, 0);
    std::vector<int> unsat(g.MAX_INDEX_OF_NODE, 0);
    std::vector<int> fixThre(g.MAX_INDEX_OF_NODE, 0);
    int sumnd = 0;

    thrust::device_vector<int> d_Delta = Delta;
    thrust::device_vector<int> d_unsat = unsat;
    thrust::device_vector<int> d_fixThre = fixThre;
    thrust::device_vector<int> d_sumnd(1, 0);

    // Allocate memory on the device for the vertices
    Vertex *d_V;
    cudaMalloc(&d_V, g.vNum * sizeof(Vertex));

    // Copy the vertices from the host to the device
    cudaMemcpy(d_V, g.V, g.vNum * sizeof(Vertex), cudaMemcpyHostToDevice);
    init_arrays<<<(g.MAX_INDEX_OF_NODE + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(d_Delta.data()),
        thrust::raw_pointer_cast(d_unsat.data()),
        thrust::raw_pointer_cast(d_fixThre.data()),
        g.MAX_INDEX_OF_NODE, d_V,
        thrust::raw_pointer_cast(d_sumnd.data()));

    // Ensure all GPU computations are finished before proceeding
    cudaDeviceSynchronize();
    cudaFree(d_V);
    // Copy data from device to host
    thrust::copy(d_Delta.begin(), d_Delta.end(), Delta.begin());
    thrust::copy(d_unsat.begin(), d_unsat.end(), unsat.begin());
    thrust::copy(d_fixThre.begin(), d_fixThre.end(), fixThre.begin());
    // Copy back the sumnd result from device to host
    sumnd = d_sumnd[0];

    const int len = s.size();
    std::vector<mynode> h_T(len);
    set<int>::iterator spointer = s.begin();
    set<int>::iterator epointer = s.end();
    for (i = 0; spointer != epointer; spointer++, i++)
    {
        id = *spointer;
        h_T[i] = mynode{id, g.V[id].neighNum};
    }

    thrust::device_vector<mynode> d_T = h_T;
    thrust::sort(d_T.begin(), d_T.end(), mycmp1());
    thrust::copy(d_T.begin(), d_T.end(), h_T.begin());

    mynode T[len];
    std::copy(h_T.begin(), h_T.end(), T);

    set<int> res;

    for (i = 0; i < len && sumnd > 0; i++)
    {
        int vi = T[i].index;
        int R = Delta[vi];
        if (R > 0)
        {
            pH = g.V[vi].headEdge;
            set<int> nid;
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

            for (j = 0; j < R; j++)
            {
                spointer = nid.begin();
                epointer = nid.end();
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
                sumnd -= unsat[u];
                pH = g.V[u].headEdge;
                while (pH != NULL)
                {
                    x = pH->edVertex;
                    if (Delta[x] > 0)
                    {
                        Delta[x]--;
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
        if (g.V[id].neighNum == 0)
        {
            std::cout << "Node " << id << " has no neighbors.\n";
        }
        else
        {
            naV = fixThre[*spointer] - Delta[*spointer];
            aprate += 1.00 * naV / g.V[id].neighNum;
        }
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

    // Destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return signal;
}
