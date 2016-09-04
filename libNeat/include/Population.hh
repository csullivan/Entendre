#include <memory>
#include <vector>

class Organism;
class Species;

class Population {
public:
  std::vector<std::shared_ptr<Organism> > organisms;
  std::vector<std::shared_ptr<Species> > species;
  int largest_innovation;
};
