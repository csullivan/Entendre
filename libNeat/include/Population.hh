#pragma once
#include "Genome.hh"
#include <vector>


struct Organism {
  // A proxy class for class Genome, containing its
  // evolutionary and species adjusted fitness
  Genome* operator->() { return genome; }
  Genome& operator*() { return *genome; }
  float fitness;
  float adj_fitness;
  Genome* genome;
  // NeuralNet* net;

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
  void Reproduce(std::vector<Organism>&);
private:
  std::vector<Genome> population;
};
