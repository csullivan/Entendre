#pragma once
#include <vector>
#include <unordered_map>
#include "NeuralNet.hh"
#include "Random.hh"
#include "Requirements.hh"


struct ConnectionGene {
  unsigned int origin;
  unsigned int dest;
  double weight;
  bool enabled;
  bool operator==(const ConnectionGene& other) {
    return (origin == other.origin) && (dest == other.dest);
  }
};

struct NodeGene {
  NodeGene() = delete;
  NodeGene(NodeType& type_) : type(type_), innovation(0) {;}
  NodeType type;
  unsigned long innovation;
};

class Genome : public uses_random_numbers,
               public requires<Probabilities> {
public:
  Genome operator()(const Genome& father);
  operator NeuralNet() const;
  void operator=(const Genome&);
  Genome& AddNode(NodeType type);
  Genome& AddConnection(unsigned int origin, unsigned int dest,
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
  static unsigned long Hash(unsigned long id,
                            unsigned long previous_hash) {
    return ((id*10000169) xor (previous_hash*44721359));
  }


private:
  std::vector<NodeGene> node_genes;
  std::unordered_map<unsigned long, ConnectionGene> connection_genes;
};
