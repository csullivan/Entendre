#include <memory>
#include <vector>

class Genome;

class Network {
public:
  int id;
  std::vector<std::shared_ptr<Node> > nodes;
  std::shared_ptr<Genome> genotype;
  std::vector<std::shared_ptr<Node> > inputs;
  std::vector<std::shared_ptr<Node> > outputs;
};
