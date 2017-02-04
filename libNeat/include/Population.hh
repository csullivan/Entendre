#pragma once
#include "Genome.hh"
#include "NeuralNet.hh"

#include <vector>
#include <limits>
#include <unordered_map>

struct Organism {
  Organism(const Genome& gen)
    : fitness(std::numeric_limits<double>::quiet_NaN()),
      adj_fitness(std::numeric_limits<double>::quiet_NaN()),
      genome(gen) , network(NeuralNet(gen)) { ; }
  float fitness;
  float adj_fitness;
  Genome genome;
  NeuralNet network;
};

struct Species {
  std::vector<Organism> organisms;

  unsigned int id;
  Genome representative;
  unsigned int age;
  double best_fitness;
};

class Population : public uses_random_numbers,
                   public requires<Probabilities> {
public:
  /// Construct a population, starting from a seed genome
  Population(Genome& first,
             std::shared_ptr<RNG>,std::shared_ptr<Probabilities>);

  Population(const Population&) = default;
  Population(Population&&) = default;
  Population& operator=(const Population& rhs) = default;
  Population& operator=(Population&&) = default;

  /// Evaluate the fitness function for each neural net.
  template<class Callable>
  void Evaluate(Callable&& fitness) {
    for(auto& spec : species) {
      for(auto& org : spec.organisms) {
        org.fitness = fitness(org.network);
      }
    }
    CalculateAdjustedFitness();
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
  unsigned int NumSpecies();
  unsigned int NumViableSpecies();

  unsigned int SpeciesSize(size_t i) const;

  std::pair<double, double> MeanStdDev() const;

private:
  /// Construct a population, starting from the specified population of organisms and species
  Population(std::vector<Species> species,
             std::shared_ptr<RNG>,std::shared_ptr<Probabilities>);

  void Speciate(std::vector<Species>& species,
                const std::vector<Genome>& genomes);
  void CalculateAdjustedFitness();

  std::vector<Species> species;
};
