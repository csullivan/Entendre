#include <memory>
#include <vector>

class Organism;

struct Fitness {
  double current;
  double max;
  double alltime;
};
class Species {
public:
  int id;
  int age;
  Fitness fitness;
  std::vector<std::shared_ptr<Organism> > members;

};
