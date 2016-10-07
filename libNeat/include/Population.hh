#pragma once
#include "Genome.hh"
#include "NeuralNet.hh"
#include <vector>


struct Organism {
  // A proxy class for class Genome, containing its
  // evolutionary and species adjusted fitness
  Genome* operator->() { return genome; }
  Genome& operator*() { return *genome; }
  float fitness;
  float adj_fitness;
  Genome* genome;
  NeuralNet* net;

  // Consider adding a NeuralNet pointer. Justification
  // would be that the population evaluates the networks
  // and so already has accessed to these objects, which
  // can then be passed in to the mutate functions of the
  // Genome, so that the nets don't need to be regenerated
};

class Population : public uses_random_numbers,
                   public requires<Probabilities> {
public:
  Population(Genome&,std::shared_ptr<RNG>,std::shared_ptr<Probabilities>);

  void BuildNetworks();

  void Reproduce(std::vector<Organism>&);

  //std::vector<Organism> Evaluate(std::vector<double>);

  template <class Callable, class... Args>
  std::vector<Organism> Evaluate(Callable&& fitness, Args&&... args) {

    std::vector<Organism> output;
    auto n = 0u;
    for (auto& net : networks) {
      Organism org = {};
      org.fitness = fitness(net,std::forward<Args>(args)...);
      org.genome = &population[n++];
      org.net = &net;
      output.push_back(org);
    }

    return output;
  }

  template<typename Callable>
  void RegisterFitness(Callable fitness) { fitness = fitness; }
private:
  std::vector<Genome> population;
  std::vector<NeuralNet> networks;

  // std::function<double(const NeuralNet&)> fitness;
};
