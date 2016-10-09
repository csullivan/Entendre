#pragma once
#include "Genome.hh"
#include "NeuralNet.hh"
#include <vector>


class Population : public uses_random_numbers,
                   public requires<Probabilities> {

public:
  /// Construct a population, starting from a seed genome
  Population(Genome& first,
             std::shared_ptr<RNG>,std::shared_ptr<Probabilities>);

  /// Construct a population, starting from the specified population.
  Population(std::vector<Genome> population,
             std::shared_ptr<RNG>,std::shared_ptr<Probabilities>);

  Population(const Population&);

  Population& operator=(Population&&);
  /// Evaluate the fitness function for each neural net.
  template<class Callable>
  void Evaluate(Callable&& fitness) {
    organisms.clear();
    auto n = 0u;
    for (auto& net : networks) {
      Organism org = {};
      org.fitness = fitness(net);
      org.genome = &population[n++];
      organisms.push_back(org);
    }
  }

  /// Reproduce, using the fitness function given.
  template <class Callable>
  Population Reproduce(Callable&& fitness) {
    Evaluate(std::forward<Callable>(fitness));
    return Reproduce();
  }

  /// Reproduce, using the already evaluated fitness function.
  /**
     Assumes that Evaluate() has already been called.
   */
  Population Reproduce();

  Population operator=(const Population& rhs);

private:
  struct Organism {
    // A proxy class for class Genome, containing its
    // evolutionary and species adjusted fitness
    Genome* operator->() { return genome; }
    Genome& operator*() { return *genome; }
    float fitness;
    float adj_fitness;
    Genome* genome;
  };

  void build_networks();

  std::vector<Genome> population;
  std::vector<NeuralNet> networks;
  std::vector<Organism> organisms;
};
