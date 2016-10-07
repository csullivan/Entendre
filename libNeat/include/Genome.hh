#pragma once
#include <vector>
#include <unordered_map>
#include "NeuralNet.hh"
#include "Random.hh"
#include "Requirements.hh"
#include "insertion_preserving_unordered.hh"

struct ConnectionGene {
  unsigned long origin;
  unsigned long dest;
  double weight;
  bool enabled;
  bool operator==(const ConnectionGene& other) {
    return (origin == other.origin) && (dest == other.dest);
  }
};

struct NodeGene {
  NodeGene() = delete;
  NodeGene(const NodeType& type_) : type(type_), innovation(0) {;}
  NodeGene(const NodeType& type_, unsigned long innov) : type(type_), innovation(innov) {;}
  NodeType type;
  unsigned long innovation;
};

class ReachabilityChecker;

class Genome : public uses_random_numbers,
               public requires<Probabilities> {
public:
  operator NeuralNet() const;
  Genome operator=(const Genome&);
  //Genome& operator=(Genome);
  Genome& AddNode(NodeType type);
  Genome& AddConnection(unsigned long origin, unsigned long dest,
                        bool status, double weight);
  Genome& RandomizeWeights();
  Genome  MateWith(const Genome& father);
  Genome  MateWith(Genome* father);
  void    Mutate();
  void    Mutate(const NeuralNet&);
  void    MutateConnection(const NeuralNet&);
  void    MutateNode();
  void    MutateWeights();
  void    MutateReEnableGene();
  void    MutateToggleGeneStatus();
  bool    ConnectivityCheck(unsigned int node_index, const ReachabilityChecker& checker) const;
  float   GeneticDistance(const Genome&) const;
  void    PrintInnovations() const;
  size_t  Size() { return connection_genes.size(); }

private:
  static unsigned long Hash(unsigned long origin,unsigned long dest,unsigned long previous_hash) { return ((origin*746151647) xor (dest*15141163) xor (previous_hash*94008721) xor (5452515049)); }
  static unsigned long Hash(unsigned long id,unsigned long previous_hash) { return ((id*10000169) xor (previous_hash*44721359) xor (111181111));  }


private:
  std::vector<NodeGene> node_genes;
  std::unordered_map<unsigned long,unsigned int> node_lookup;
  insertion_ordered_map<unsigned long, ConnectionGene> connection_genes;
};
