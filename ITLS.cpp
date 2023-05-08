Solution ITLS(const Graph &G, double cutoff_time, Solution &DT) {
    set<int> tabu_list;
    Solution DT_prime = DT;
    vector<int> Dscore(G.V);

    for (int v = 0; v < G.V; ++v) {
        if (DT.DT.count(v)) {
            Dscore[v] = -1 * compute_Dscore(G, DT, v); // Assuming the provided pseudo code
        } else {
            Dscore[v] = compute_Dscore(G, DT, v); // Assuming the provided pseudo code
        }
    }

    clock_t start_time = clock();

    while (static_cast<double>(clock() - start_time) / CLOCKS_PER_SEC < cutoff_time) {
        if (!has_non_dominated_vertex(G, DT) && is_connected(G, DT)) {
            remove_redundant_vertices(G, DT); // Assuming the provided pseudo code
            construct_minimum_spanning_tree(G, DT); // Assuming the provided pseudo code

            if (compare_solutions(DT, DT_prime)) { // Assuming the provided pseudo code
                DT_prime = DT;
            }
        }

        // Removing Phase
        while (!has_non_dominated_vertex(G, DT)) {
            int v = remove_vertex(G, DT, Dscore, tabu_list); // Assuming the provided pseudo code
            update_Dscore(G, DT, Dscore, tabu_list);
        }

        // Dominating Phase
        tabu_list.clear();
        while (has_non_dominated_vertex(G, DT)) {
            int v = add_vertex(G, DT, Dscore); // Assuming the provided pseudo code
            tabu_list.insert(v);
            update_Dscore(G, DT, Dscore, tabu_list);
        }

        // Connecting Phase
        while (!is_connected(G, DT)) {
            connect_components(G, DT, Dscore, tabu_list); // Assuming the provided pseudo code
            update_Dscore(G, DT, Dscore, tabu_list);
        }
    }

    return DT_prime;
}