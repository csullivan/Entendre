#include "Genome.hh"
#include <vector>
#include <iostream>

Genome::operator NeuralNet() const {
  NeuralNet net(nodes);
  // add network connections where appropriate
  for (auto const& gene : genes) {
    if (gene.enabled) {
      net.add_connection(gene.link.origin,gene.link.dest,gene.link.weight);
    }
  }

  return net;
}

Genome& Genome::AddNode(NodeType type) {
  nodes.emplace_back(type);
  return *this;
}
Genome& Genome::AddGene(unsigned int origin, unsigned int dest, ConnectionType type,
             bool status, double weight) {

  unsigned long innovation = 0;
  if (genes.size()) {
    innovation = Hash(dest,origin,genes.back().innovation_number);
  }
  else {
    innovation = Hash(dest,origin,innovation);
  }


  genes.push_back({status,innovation,Connection(origin,dest,type,weight)});
  return *this;
}

void Genome::WeightMutate() {
  if (!generator) { throw std::runtime_error("No RNG engine set. Replace this with class specific exception."); }
  for (auto& gene : genes) {
    // perturb weight by a small amount
    if(random()<Constants::PerturbWeight) {
      gene.link.weight += Constants::StepSize*(2*random()-1);
    } else { // otherwise randomly set weight within reset range
      gene.link.weight = (random() - 0.5)*Constants::ResetWeightScale;
    }
  }
}

void Genome::LinkMutate() {

}

void Genome::NodeMutate() {

}

void Genome::Mutate() {

}

void Genome::PrintInnovations() {
  for (auto const& gene : genes) {
    std::cout << gene.innovation_number << std::endl;
  }
}
