#include "Genome.hh"
#include "Node.hh"
#include "Gene.hh"
#include "Network.hh"
#include <memory>

Genome make_genome(int id) {
  return Genome(id);
}


Genome::Genome(int id_,
               std::vector<std::shared_ptr<Node>> nodes_,
	       std::vector<std::shared_ptr<Gene>> genes_)
  : id(id_), nodes(nodes_), genes(genes_) {
}

Genome& Genome::AddNode(Node::Type t,Node::Function f) {
  nodes.push_back(std::make_shared<Node>(t,f));
  return *this;
}
Genome& Genome::AddGene(int origin, int dest, double weight, Gene::Status s, int innovation) {
  genes.push_back(std::make_shared<Gene>(nodes.at(origin),nodes.at(dest),weight,s,innovation));
  return *this;
}

std::shared_ptr<Network> Genome::BuildNetwork(const std::shared_ptr<Genome>& gnome) {
  return std::make_shared<Network>(gnome);
}
