#include <memory>
#include <vector>

class Organism {
public:
  std::shared_ptr<Genome> genotype;
  std::shared_ptr<Network> phenotype;
  std::shared_ptr<Species> species;
  double fitness;
};
