#include "Organism.hh"
#include "Network.hh"
#include "Genome.hh"

Organism::Organism(const std::shared_ptr<Genome>& g) : genotype(g) {
  phenotype = std::make_shared<Network>(genotype);
}

void Organism::Sense(const std::vector<double>& sensory_inputs) {
  phenotype->LoadInputs(sensory_inputs);
}

double Organism::Activate() {
  return phenotype->Activate();
}
