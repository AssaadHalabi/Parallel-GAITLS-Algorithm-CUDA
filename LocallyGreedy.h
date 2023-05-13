#include "iostream"
#include "vector"
#include "set"
#include "algorithm"
#include "mgraph.h"
#include "ctime"

//return an interger no less than x/2
int mceil(int x) {
    int a, b;
    a = x / 2;
    b = 2 * a;
    if (x > b)
        a++;
    return a;
}

struct mynode {
    int index, degree;
};

bool mycmp1(mynode a, mynode b) {
    return a.degree < b.degree;
}

//parameter description:
//x: an empty vector, which will be assigned the generated PIDS at the end of the procedure
//g: the input graph, which has been constructed before the call of the following procedure
//tlen: a timer, which will be assigned the time cost of this algorithm once the main procedure is finished
//rprate: a rate, which will be assigned the average positive rate of nodes in g at the end of the procedure
int LocallyGreedy(std::vector<int> &x, const mGraph &g, double &tlen, double &rprate) {
    using namespace std;
    clock_t ts, te; //timestamps，used to calculate the actual running time of this procedure
    ts = clock();

    //NOTE: 
    //----1. A "positive" node refers to the node that is included by a (partial) PIDS.
    //----2. A node is said "satisfied" if more than half of its neighbors are "positive", and otherwise, this node is ragared as "unsatisfied".

    //Delta(v): the number of positive neighbors that node v lacked to be satisfied, which will decrease as the procedure runs
    //unsat(v): the number of node v's unsatisfied neighbors
    //fixThre(v): the number of positive neighbors that node v needs to be satisfied, which is a fixed threshold
    int *Delta = new int[g.MAX_INDEX_OF_NODE]();
    int *unsat = new int[g.MAX_INDEX_OF_NODE]();
    int *fixThre = new int[g.MAX_INDEX_OF_NODE]();

    //sumnd: the sum of every node's Delta(), which will decrease as the procedure runs
    //s: the node set of the input graph g
    int i, j, sumnd;
    set<int> s;

    Edge *pH;

    int da, db;

    //initialize Delta(v),unsat(v),fixThre(v),sumnd and s
    for (i = 0, sumnd = 0; i < g.MAX_INDEX_OF_NODE; i++) {
        pH = g.V[i].headEdge;
        if (pH != NULL) {
            da = g.V[i].neighNum;
            db = mceil(da);
            Delta[i] = db;
            fixThre[i] = db;
            unsat[i] = da;
            sumnd += db;
            s.insert(i);
        }
    }

    const int len = s.size();
    mynode T[len];
    int id, nb;
    set<int>::iterator spointer = s.begin();
    set<int>::iterator epointer = s.end();
    for (i = 0; spointer != epointer; spointer++, i++) {
        id = *spointer;
        T[i].index = id;
        T[i].degree = g.V[id].neighNum;
    }
    //sort the nodes by degree in ascending order
    sort(T, T + len, mycmp1);
    //res: the (partial) PIDS, which will be inserted with a number of nodes one by one  as the procedure runs and becomes a valid PIDS in the end
    set<int> res;
    set<int>::iterator sp2;
    set<int>::iterator ep2;

    //based on the order computed before, check every node one by one to confirm whether current node is satisfied or not
    //if the current node v is satisfied, then skip the following processing to the check towards the next node 
    //otherwise, based on a certain strategy, add v's neighbors into the partial PIDS res one by one until v becomes satisfied

    //the for loop terminates when every node has been checked or the total demand has decreased to zero  
    for (i = 0; i < len && sumnd > 0; i++) {
        int vi = T[i].index;
        int R = Delta[vi];
        if (R > 0) {
            pH = g.V[vi].headEdge;
            set<int> nid;
            //construct a candidators set by absorbing vi's neighbors that haven't been included by the (partial) PIDS res
            while (pH != NULL) {
                nb = pH->edVertex;
                if (res.find(nb) == res.end())
                    nid.insert(nb);
                pH = pH->next;
            }

            int nb2;
            int maxunsat;
            int w, u, x, y;
            //select R vi's neighbors from the candidators set and insert them into the (partial) PIDS res one by one
            for (j = 0; j < R; j++) {
                spointer = nid.begin();
                epointer = nid.end();
                //select the node with maximum unsat(), if there are several nodes sharing the same maximum unsat(), then choose the node with maximum ID
                for (maxunsat = 0; spointer != epointer; spointer++) {
                    w = *spointer;
                    if (unsat[w] >= maxunsat) {
                        maxunsat = unsat[w];
                        u = w;
                    }
                }
                res.insert(u);
                //decrease the total demand sumnd by unsat[u]
                sumnd -= unsat[u];
                //update Delata() and unsat() of relevant nodes
                pH = g.V[u].headEdge;
                while (pH != NULL) {
                    x = pH->edVertex;
                    //only when Delta(x)>0, the addition of u contributes to reducing the demand of x 
                    if (Delta[x] > 0) {
                        Delta[x]--;
                        //only when x's Delta() decreases from 1 to 0 can further lead relevant nodes' unsat() to decrease
                        if (Delta[x] == 0) {
                            Edge *pH2 = g.V[x].headEdge;
                            while (pH2 != NULL) {
                                y = pH2->edVertex;
                                unsat[y]--;
                                pH2 = pH2->next;
                            }
                        }
                    }
                    pH = pH->next;
                }
                //remove the selected node u from the candidators set
                nid.erase(u);
            }
            nid.clear();
        }
    }

    spointer = res.begin();
    epointer = res.end();
    for (; spointer != epointer; spointer++) {
        x.push_back(*spointer);
    }
    te = clock();
    //tlen: the time cost of the above procedure
    tlen = double(te - ts) / CLK_TCK;

    //check whether the result is a valid PIDS
    int signal = 1;
    //rrate: |the generated PIDS|/|the node set of the input graph| 
    //aprate: the average positive rate of the nodes in the graph
    double rrate, aprate;
    //naV: the number of the current node's positive neighbors
    int naV;
    aprate = 0;
    spointer = s.begin();
    epointer = s.end();
    id = *spointer;
    for (; spointer != epointer; spointer++) {
        id = *spointer;
        naV = fixThre[*spointer] - Delta[*spointer];
        aprate += 1.00 * naV / g.V[id].neighNum;
        if (naV < g.V[id].neighNum / 2) {
            signal = 0;
            break;
        }
    }
    aprate = aprate / len * 100;
    if (signal == 0)
        cout << "NOT satisfied!" << "(" << id << ")" << endl;
    else
        cout << "Satisfied!" << endl;
    cout << "the result rate: " << aprate << endl;
    cout << "average positive rate: " << aprate << endl;
    rprate = aprate;
    delete[] Delta;
    delete[] unsat;
    delete[] fixThre;

    return signal;
}