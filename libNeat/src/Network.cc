#include "Network.hh"
#include "Genome.hh"
#include "Node.hh"
#include <iostream>

Network::Network(const std::shared_ptr<Genome>& genome,
                 int id_ /* default = -1 */)
  : id(id_), genotype(genome) {

  if (id == -1) { id = genotype->id; }


  // sort nodes into inputs & outputs for network
  for (auto const& node : genome->nodes) {
    auto network_node = std::make_shared<Node>(Node::Type(*node),Node::Function(*node));

    if (Node::Function(*node) == Node::Function::Bias ||
        Node::Function(*node) == Node::Function::Input ) {
      inputs.push_back(network_node);
    }
    else if (Node::Function(*node) == Node::Function::Output) {
      outputs.push_back(network_node);
    }
    nodes.push_back(network_node);
  }

  // construct network connections
  // for (auto const& gene : genome->genes) {
  //   if (gene == Gene::Status::Enabled) {

  //   }
  // }

}

void Network::LoadInputs(const std::vector<double>& sensory_inputs) {
  if (sensory_inputs.size() != inputs.size()) {
    throw NetworkSensorSize();
  }

  auto input = sensory_inputs.begin();
  for (auto& inode : this->inputs) {
    *inode << *input++;
  }
}

double Network::Activate() {

  return 0.0;
}
