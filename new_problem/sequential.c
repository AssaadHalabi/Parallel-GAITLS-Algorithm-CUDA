#include<stdio.h>
#include<stdlib.h>

// Node definition for adjacency list
typedef struct node {
    int vertex;
    struct node* next;
} node;

// Graph definition
typedef struct graph {
    int numVertices;
    node** adjLists;
    int* coverSize;
    int* inDominatingSet;
    int* isDominated;
} graph;

node* createNode(int v) {
    node* newNode = malloc(sizeof(node));
    newNode->vertex = v;
    newNode->next = NULL;
    return newNode;
}

graph* createGraph(int vertices) {
    graph* g = malloc(sizeof(graph));
    g->numVertices = vertices;
    g->adjLists = malloc(vertices * sizeof(node*));
    g->coverSize = malloc(vertices * sizeof(int));
    g->inDominatingSet = malloc(vertices * sizeof(int));
    g->isDominated = malloc(vertices * sizeof(int));

    for (int i = 0; i < vertices; i++) {
        g->adjLists[i] = NULL;
        g->coverSize[i] = 0;
        g->inDominatingSet[i] = 0;
        g->isDominated[i] = 0;
    }
    return g;
}

void addEdge(graph* g, int src, int dest) {
    // Adding edge from src to dest
    node* newNode = createNode(dest);
    newNode->next = g->adjLists[src];
    g->adjLists[src] = newNode;

    // Adding edge from dest to src
    newNode = createNode(src);
    newNode->next = g->adjLists[dest];
    g->adjLists[dest] = newNode;
}

void calculateCoverSize(graph* g) {
    for (int i = 0; i < g->numVertices; i++) {
        node* temp = g->adjLists[i];
        while (temp) {
            g->coverSize[i]++;
            temp = temp->next;
        }
    }
}

void findDominatingSet(graph* g) {
    calculateCoverSize(g);

    for (int i = 0; i < g->numVertices; i++) {
        int max_cover_node = -1;

        for (int v = 0; v < g->numVertices; v++) {
            if (!g->isDominated[v] && !g->inDominatingSet[v] &&
                (max_cover_node == -1 || g->coverSize[v] > g->coverSize[max_cover_node])) {
                max_cover_node = v;
            }
        }

        if (max_cover_node != -1) {
            g->inDominatingSet[max_cover_node] = 1;

            node* temp = g->adjLists[max_cover_node];
            while (temp) {
                if (!g->isDominated[temp->vertex]) {
                    g->isDominated[temp->vertex] = 1;

                    node* neighborNode = g->adjLists[temp->vertex];
                    while (neighborNode) {
                        g->coverSize[neighborNode->vertex]--;
                        neighborNode = neighborNode->next;
                    }
                }

                temp = temp->next;
            }
        }
    }

    printf("Dominating set: ");
    for (int v = 0; v < g->numVertices; v++) {
        if (g->inDominatingSet[v]) {
            printf("%d ", v);
        }
    }
    printf("\n");
}

int main() {
    int vertices = 5;
    graph* g = createGraph(vertices);

    addEdge(g, 0, 1);
    addEdge(g, 0, 2);
    addEdge(g, 1, 2);
    addEdge(g, 1, 3);
    addEdge(g, 2, 3);
    addEdge(g, 2, 4);

    findDominatingSet(g);

    // Free memory
    for (int i = 0; i < g->numVertices; i++) {
        node* adjList = g->adjLists[i];
        node* temp = adjList;
        while (adjList) {
            temp = adjList;
            adjList = adjList->next;
            free(temp);
        }
    }
    free(g->adjLists);
    free(g->coverSize);
    free(g->inDominatingSet);
    free(g->isDominated);
    free(g);

    return 0;
}
