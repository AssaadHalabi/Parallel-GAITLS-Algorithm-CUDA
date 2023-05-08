#include <vector>
#include <algorithm>
#include <random>
#include <iterator>
#include <limits>



class Population
{
public:
    Population(const Graph &graph, int population_size) : graph_(graph), population_size_(population_size)
    {
        initializePopulation();
    }

    void initializePopulation()
    {
        for (int i = 0; i < population_size_; ++i)
        {
            DominatingTreeSolution solution(graph_);
            initRCL(solution);
            population_.push_back(solution);
        }
    }
    

    void mutate(DominatingTreeSolution &solution)
    {
        mutateHighDiversity(solution);
    }

    void initRCL(DominatingTreeSolution &solution)
    {
       
    }

    DominatingTreeSolution &selectParent()
    {
        // Implement your preferred selection method, e.g., tournament selection, roulette wheel selection, etc.
        // This is just a simple random selection for illustration purposes
        std::uniform_int_distribution<int> dist(0, population_size_ - 1);
        int selected_index = dist(rng_);
        return population_[selected_index];
    }

    DominatingTreeSolution crossover(const DominatingTreeSolution &parent1, const DominatingTreeSolution &parent2)
    {
        DominatingTreeSolution offspring(graph_);
        // Implement the GAITLS crossover method
        // ...
        return offspring;
    }

    void createOffspring()
    {
        std::vector<DominatingTreeSolution> offspring;

        for (int i = 0; i < population_size_; ++i)
        {
            DominatingTreeSolution &parent1 = selectParent();
            DominatingTreeSolution &parent2 = selectParent();
            DominatingTreeSolution child = crossover(parent1, parent2);
            mutate(child);
            offspring.push_back(child);
        }

        population_ = std::move(offspring);
    }

    const DominatingTreeSolution &getBestSolution() const
    {
        return *std::min_element(population_.begin(), population_.end(),
                                 [](const DominatingTreeSolution &a, const DominatingTreeSolution &b)
                                 {
                                     return a.getTotalWeight() < b.getTotalWeight();
                                 });
    }

private:
    const Graph &graph_;
    int population_size_;
    std::vector<DominatingTreeSolution> population_;
    std::mt19937 rng_{std::random_device{}()};
};