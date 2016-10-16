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
      org.network = &net;
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

  Population& operator=(const Population& rhs);

  /// Returns the best neural net in the population.
  /**
     Uses the fitness value calculated by the most recent call to Evaluate or Reproduce.
     If neither has been called, returns nullptr.
   */
  NeuralNet* BestNet() const;

  /// Returns the number of species in the population
  /**
     Uses the speciation from the most recent call to Reproduce.
     If Reproduce has not been called, returns 0.
   */
  unsigned int NumSpecies() const;

private:
  struct Organism {
    // A proxy class for class Genome, containing its
    // evolutionary and species adjusted fitness
    Genome* operator->() { return genome; }
    Genome& operator*() { return *genome; }
    float fitness;
    float adj_fitness;
    unsigned int species;
    Genome* genome;
    NeuralNet* network;
  };

  void build_networks();

  std::vector<Genome> population;
  std::vector<NeuralNet> networks;
  std::vector<Organism> organisms;
};
