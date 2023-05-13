#include <cstddef>

struct Edge {
    int stVertex;  //start point ID of the edge
    int edVertex;  //end point ID of the edge
    Edge *next;

    Edge(int stPt, int edPt) {
        this->stVertex = stPt;
        this->edVertex = edPt;
        this->next = NULL;
    }
};

struct Vertex {
    int neighNum;   //the number of all neighbors
    int indegree, outdegree;
    Edge *headEdge;

    Vertex() {
        neighNum = 0;
        indegree = 0;
        outdegree = 0;
        headEdge = NULL;
    }
};

class mGraph {
public:
    int MAX_INDEX_OF_NODE;  //max index of node
    int vNum;   //|V|
    int eNum;   //|E|
    Vertex *V;

    mGraph() {
        vNum = 0;
        eNum = 0;
        V = NULL;
    }

    mGraph(int mindex, int n, int m) {
        if (mindex <= 0)
            MAX_INDEX_OF_NODE = 0;
        else if (mindex < n)
            MAX_INDEX_OF_NODE = n;
        else {
            MAX_INDEX_OF_NODE = mindex;
        }
        if (n <= 0) {
            vNum = 0;
            V = NULL;
        } else {
            vNum = n;
            V = new Vertex[MAX_INDEX_OF_NODE];
        }
        if (m <= 0) {
            eNum = 0;
        } else {
            eNum = m;
        }
    }

    mGraph(const mGraph &g)//copy the construction function
    {
        vNum = g.vNum;
        if (vNum > 0) {
            V = new Vertex[vNum];
            for (int i = 0; i < vNum; ++i) {
                V[i].neighNum = g.V[i].neighNum;
                Edge *pEg = g.V[i].headEdge;
                if (pEg != NULL) {
                    V[i].headEdge = new Edge(pEg->stVertex, pEg->edVertex);
                    pEg = pEg->next;
                }
                Edge *pE = V[i].headEdge;
                while (pEg != NULL) {
                    pE->next = new Edge(pEg->stVertex, pEg->edVertex);
                    pE = pE->next;
                    pEg = pEg->next;
                }
            }
        } else {
            V = NULL;
        }
    }

    ~mGraph() {
        if (V != NULL) {
            for (int i = 0; i < vNum; ++i) {
                Edge *pE = V[i].headEdge;
                V[i].headEdge = NULL;
                while (pE != NULL) {
                    Edge *pDel = pE;
                    pE = pE->next;
                    delete pDel;
                }
            }
            delete[] V;
            V = NULL;
        }
    }

    void addSingleEdge(int s, int e) {
        Edge *pE = new Edge(s, e);
        if (V[s].headEdge == NULL || V[s].headEdge->edVertex >= e) {
            pE->next = V[s].headEdge;
            V[s].headEdge = pE;
        } else {
            Edge *pH = V[s].headEdge;
            while (pH->next != NULL && pH->next->edVertex < e)
                pH = pH->next;
            pE->next = pH->next;
            pH->next = pE;
        }
        V[s].neighNum++;
        V[s].outdegree++;
        V[e].indegree++;
        return;
    }
};