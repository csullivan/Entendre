#include <memory>
#include <vector>


class Node;
class Gene;
class Network;

class Genome {
public:
  int id;
  std::vector<std::shared_ptr<Node> > nodes;
  std::vector<std::shared_ptr<Gene> > genes;
  std::shared_ptr<Network> phenotype;
};
