#include <iostream>
#include "iomanip"
#include "vector"
#include "set"
#include "cstring"
#include <fstream>
#include <cassert>
#include "ctime"

#include "LocallyGreedy.h"

using namespace std;

//return a string vector, which is obtained by splitting the original string s according to the separator sep
std::vector<std::string> split(const std::string &s, const std::string &sep) {
    using namespace std;
    vector<string> result(0);   //use result save the spilted result

    if (s.empty())    //judge the arguments
    {
        return result;
    }
    string::size_type pos_begin = s.find_first_not_of(sep);//find first element of srcstr

    string::size_type dlm_pos;//the delimeter postion
    string temp;              //use third-party temp to save splited element
    while (pos_begin != string::npos)//if not a next of end, continue spliting
    {
        dlm_pos = s.find(sep, pos_begin);//find the delimeter symbol
        if (dlm_pos != string::npos) {
            temp = s.substr(pos_begin, dlm_pos - pos_begin);
            pos_begin = dlm_pos + sep.length();
        } else {
            temp = s.substr(pos_begin);
            pos_begin = dlm_pos;
        }
        if (!temp.empty())
            result.push_back(temp);
    }
    return result;
}

//load graph data from the local datasets file
//the graphs used in the experiment are all preprocessed
//during the graph preprocessing phase, the isolated nodes, redundant edges and self-loops are all removed to ensure any of these graphs has valid PIDS
//parameter description:
//file: the input file data
//g: an object of clss mGraph, which has been created based on the given statistics of the input graph
//gsize: a counter, which will be assigned the number of nodes in the input graph
void loadFile(const string &ifile, mGraph &g, int &gsize) {
    ifstream infile;
    infile.open(ifile.data());
    assert(infile.is_open());
    string s;
    int iskip = 0;
    vector<string> temp;
    //in our input graph data, the first 3 lines are statistical information of the graph and we need skip these data
    for (; iskip < 3; iskip++) {
        getline(infile, s);
    }
    //sep: the character that separates the starting point and the ending point of an edge
    string sep = " ";
    int n1, n2;
    set<int> gnodes;
    while (getline(infile, s)) {
        temp = split(s, sep);
        n1 = stoi(temp[0]);
        n2 = stoi(temp[1]);
        gnodes.insert(n1);
        gnodes.insert(n2);
        g.addSingleEdge(n1, n2);
        // as the graph is undirected, make the edge double-sided
        g.addSingleEdge(n2, n1);
    }
    infile.close();
    gsize = gnodes.size();
}

int main() {
    const int LOOP = 20;
    clock_t t0, t1;
    double d1, d2, d3;

    //some basic information about the datasets
    string datasets[4] = {"karate", "dolphins", "football", "jazz"};
    int n[4] = {34, 62, 115, 198};//the number of nodes in corresponding graphs
    int e[4] = {78, 159, 613, 2742};//the number of edges in corresponding graphs
    int mid[4] = {40, 70, 120, 200};//the maximum node ID in corresponding graphs

    string ofile =
            "C:\\Users\\ASUS\\Desktop\\PIDS\\outfile\\test.txt";//the path of the output file
    ofstream outfile;
    outfile.open(ofile, std::ios::out | std::ios::app);
    assert(outfile.is_open());
    string mcontext = "";
    int gsize = 0, wsize = 0;
    double rprate = 0;

    for (int testID = 0; testID < 4; testID++) {
        cout << "\t" << testID << endl;

        t0 = clock();
        mGraph G0(mid[testID], n[testID], e[testID]);
        t1 = clock();
        d1 = double(t1 - t0) / CLK_TCK;

        string mtmp =
                "\n------------ test " + to_string(testID) + ": " + datasets[testID] + ".txt ------------\n" +
                "\t(1/3) Time for constructing mGraph object: " + to_string(d1) + " s\n";

        t0 = clock();
        //the path of input file
        string file = "C:\\Users\\ASUS\\Desktop\\PIDS\\inputfile\\";
        file += datasets[testID] + ".txt";

        loadFile(file, G0, gsize);
        t1 = clock();
        d2 = double(t1 - t0) / CLK_TCK;
        mtmp += "\t(2/3) Time for loading mGraph data: " + to_string(d2) + " s\n";

        //cout << mtmp << endl;

        //accumulate the time cost for our algorithm generating a PIDS over LOOP runs
        double trun = 0;
        //sbest: the minimum size of the output results
        //sworst: the maximum size of the output results
        //savg: accumulate the size of the output results
        //snow: the size of the output result in current turn
        int sbest = n[testID], sworst = 0, savg = 0, snow;
        for (int i = 0; i < LOOP; i++) {
            cout << "LOOP " << i + 1 << endl;
            vector<int> w;
            d2 = 0;
            LocallyGreedy(w, G0, d2, rprate);
            trun += d2;

            snow = w.size();
            savg += snow;

            if (sbest > snow)
                sbest = snow;
            if (sworst < snow)
                sworst = snow;
        }
        trun = trun / LOOP;
        wsize = savg / LOOP;

        mtmp += "\t(3/3) Time for find PIDS: " + to_string(trun) + " s\n";
        mtmp += "\t|P|: \tBEST= " + to_string(sbest) +
                ", WORST= " + to_string(sworst) +
                ", AVG= " + to_string(wsize) + "\n";
        mtmp += "\t(|P|/|V|)*100%=\t" + to_string(1.00 * wsize / gsize * 100) + "\n";
        mtmp += "\tAverage positive rate=\t" + to_string(rprate) + "\n";

        mcontext = mtmp;
        outfile << mcontext;
    }
    outfile.close();

    return 0;
}