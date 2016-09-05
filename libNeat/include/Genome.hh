#pragma once
#include <memory>
#include <vector>
#include "Node.hh"
#include "Gene.hh"

class Network;

class Genome { friend class Network;
  int id;
  std::vector<std::shared_ptr<Node>> nodes;
  std::vector<std::shared_ptr<Gene>> genes;
  std::shared_ptr<Network> phenotype;
public:
  Genome(int id_) :id(id_) {;}
  Genome(int id_, std::vector<std::shared_ptr<Node>>,
         std::vector<std::shared_ptr<Gene>>);
  Genome& AddNode(Node::Type,Node::Function);
  Genome& AddGene(int in, int out, double weight, Gene::Status, int innovation);

  static std::shared_ptr<Network> BuildNetwork(const std::shared_ptr<Genome>&);
};
Genome make_genome(int);
