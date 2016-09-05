#include <memory>
#include <vector>

class Genome;
class Network;
class Species;

class Organism {
  std::shared_ptr<Genome> genotype;
  std::shared_ptr<Network> phenotype;
  std::shared_ptr<Species> species;
public:
  Organism(const std::shared_ptr<Genome>&);
  //std::shared_ptr<Network> Net() { return phenotype; }
  void Sense(const std::vector<double>& sensory_inputs);
  double Activate();

  double fitness;
};
