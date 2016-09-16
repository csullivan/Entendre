#include "Genome.hh"
#include <algorithm>
#include <iostream>

Genome::operator NeuralNet() const {
  NeuralNet net(node_genes);
  // add network connections where appropriate
  for (auto const& gene : connection_genes) {
    if (gene.second.enabled) {
      net.add_connection(gene.second.origin,gene.second.dest,gene.second.weight);
    }
  }

  return net;
}

void Genome::operator=(const Genome& rhs) {
  this->node_genes = rhs.node_genes;
  this->connection_genes = rhs.connection_genes;
  generator = rhs.generator;
}

// Generalized genome crossover
Genome Genome::operator()(const Genome& father) {
  // Implicit assumption: Mother must always be the more
  // fit genome. i.e. child = mother(father) such that
  // fitness(mother) > fitness(father)

  auto& mother = *this;
  Genome child;

  const auto& match          = required()->match;
  const auto& single_greater = required()->single_greater;
  const auto& single_lesser  = required()->single_lesser;

  for (auto const& maternal : mother.connection_genes) {

    // Find all shared genes, look up by hash
    auto paternal = father.connection_genes.find(maternal.first);
    if (paternal != father.connection_genes.end()) {
      // matching genes
      if (random()<match) {
        // if key doesn't already exist in child,
        // then the maternal gene is inserted
        child.connection_genes.insert(maternal);
      } else {
        // paternal gene is taken
        child.connection_genes.insert(*paternal);
      }
    } else {
      // non matching gene, randomly insert maternal gene
      if (random()<single_greater) {
        child.connection_genes.insert(maternal);
      }
    }
  }

  // Standard NEAT bails out here
  if (single_lesser == 0.0) { return child; }

  // allow for merging of structure from less fit parent
  for (auto const& paternal : father.connection_genes) {
    if (random()<single_lesser) {
      child.connection_genes.insert(paternal);
    }
  }

  return child;
}

Genome& Genome::AddNode(NodeType type) {
  node_genes.emplace_back(type);
  return *this;
}

// Public API for adding structure, should not used internally
// Likely this should be performed differently.
Genome& Genome::AddConnection(unsigned int origin, unsigned int dest,
                              bool status, double weight) {
  static unsigned long last_innovation = 0;
  unsigned long innovation = 0;

  if (last_innovation == 0) {
    last_innovation = Hash(origin,dest,0);
  }

  if (node_genes.at(origin).innovation == 0) {
    origin = node_genes[origin].innovation = Hash(origin,last_innovation);

  }
  if (node_genes.at(dest).innovation == 0) {
    dest = node_genes[dest].innovation = Hash(dest,last_innovation);
  }

  innovation = Hash(origin,dest,last_innovation);
  last_innovation = innovation;

  connection_genes[innovation] = {origin,dest,weight,status};
  return *this;
}

void Genome::WeightMutate() {
  if (!generator) { throw std::runtime_error("No RNG engine set. Replace this with class specific exception."); }
  for (auto& gene : connection_genes) {
    // perturb weight by a small amount
    if(random()<required()->mutate_weight) {
      gene.second.weight += required()->step_size*(2*random()-1);
    } else { // otherwise randomly set weight within reset range
      gene.second.weight = (random() - 0.5)*required()->reset_weight;
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
  for (auto const& gene : connection_genes) {
    std::cout << gene.first << std::endl;
  }
}
