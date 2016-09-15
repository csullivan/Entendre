#pragma once
#include <vector>
#include "NeuralNet.hh"
#include "Random.hh"

namespace Constants {
  const float Match = 0.5;
  const float PerturbWeight = 0.9;
  const float StepSize = 0.1;
  const float ResetWeightScale = 4.0;
}

struct Gene {
  bool enabled;
  unsigned long innovation_number;
  Connection link;
  bool operator==(const Gene& other) {
    return
      (link.origin == other.link.origin) &&
      (link.dest == other.link.dest) &&
      (link.type == other.link.type);
  }
};

class Genome : public uses_random_numbers {
public:
  Genome operator()(const Genome& father);
  operator NeuralNet() const;
  void operator=(const Genome&);
  Genome& AddNode(NodeType type);
  Genome& AddGene(unsigned int origin, unsigned int dest, ConnectionType type,
               bool status, double weight);
  void WeightMutate();
  void LinkMutate();
  void NodeMutate();
  void Mutate();
  void PrintInnovations();

  static unsigned long Hash(unsigned long origin,
                            unsigned long dest,
                            unsigned long previous_hash) {
    return ((origin*746151647) xor (dest*15141163) xor (previous_hash*94008721));
  }


private:
  std::vector<Node> nodes;
  std::vector<Gene> genes;
};
