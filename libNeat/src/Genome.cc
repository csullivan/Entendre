#include "Genome.hh"
#include <algorithm>
#include <iostream>
#include <climits>

Genome::operator NeuralNet() const{
  NeuralNet net(node_genes);
  // add network connections where appropriate
  //for (auto const& gene : connection_genes) {
  for (auto n=0u; n<connection_genes.size(); n++) {
    if (connection_genes[n].enabled) {
      int i = node_lookup.at(connection_genes[n].origin);
      int j = node_lookup.at(connection_genes[n].dest);
      net.add_connection(i,j,connection_genes[n].weight);
    }
  }

  return net;
}

Genome Genome::operator=(const Genome& rhs) {
  this->node_genes = rhs.node_genes;
  this->connection_genes = rhs.connection_genes;
  generator = rhs.generator;
  requires<Probabilities>::operator=(rhs);
  return *this;
}

float Genome::GeneticDistance(const Genome& other) {
  double weight_diffs = 0.0;
  unsigned long nUnshared = 0;
  auto nGenes = std::max(connection_genes.size(),other.connection_genes.size());

  // loop over this genomes genes
  for (auto const& gene : connection_genes) {
    auto other_gene = other.connection_genes.find(gene.first);
    // sum the absolute weight differences of the shared genes
    if (other_gene != other.connection_genes.end()) {
      weight_diffs += std::abs(other_gene->second.weight-gene.second.weight);
    }
    // count the number of unshared genes
    else {
      nUnshared++;
    }
  }
  // loop over other genomes genes and count the unshared genes
  for (auto const& other_gene : other.connection_genes) {
    auto gene = connection_genes.find(other_gene.first);
    if (gene == connection_genes.end()) {
      nUnshared++;
    }
  }

  return (required()->genetic_c1*nUnshared)/nGenes + required()->genetic_c2*weight_diffs/nGenes;
}

// Generalized genome crossover
Genome Genome::operator()(const Genome& father) {
  // Implicit assumption: Mother must always be the more
  // fit genome. i.e. child = mother(father) such that
  // fitness(mother) > fitness(father)
  auto& mother = *this;
  Genome child;
  child.set_generator(mother.get_generator());
  child.required(mother.required());

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
  static unsigned long last_innov = 0,
    idxin = 0, idxout = 0, idxhidden = ULONG_MAX/2;

  unsigned long innovation = 0;
  switch(type) {
  case NodeType::Bias:
    innovation = Hash(0,0);
    break;
  case NodeType::Input:
    innovation = Hash(idxin++,last_innov);
    break;
  case NodeType::Output:
    innovation = Hash(idxout--,last_innov);
    break;
  case NodeType::Hidden:
    innovation = Hash(idxhidden,last_innov);
    break;
  }
  node_genes.emplace_back(type,innovation);
  last_innov = innovation;
  return *this;
}

// Public API for adding structure, should not used internally
// Likely this should be performed differently.
Genome& Genome::AddConnection(unsigned long origin, unsigned long dest,
                              bool status, double weight) {
  static unsigned long last_innovation = 0;
  unsigned long innovation = 0;

  // first gene only
  if (last_innovation == 0) {
    last_innovation = Hash(origin,dest,0);
  }

  // build look up table from innovation hash to vector index
  //std::cout << node_genes[origin].innovation << " " << origin << std::endl;
  //std::cout << node_genes[dest].innovation << " " << dest << std::endl;

  node_lookup[node_genes[origin].innovation] = origin;
  node_lookup[node_genes[dest].innovation] = dest;

  innovation = Hash(node_genes[origin].innovation,
                    node_genes[dest].innovation,
                    last_innovation);

  last_innovation = innovation;

  connection_genes.insert({innovation,{node_genes[origin].innovation,node_genes[dest].innovation,weight,status}});

  return *this;
}

void Genome::Mutate(const NeuralNet& net) {

  // structural mutation
  if (random() < required()->mutate_node) {
    MutateNode(net);
  }
  else if (random() < required()->mutate_link) {
    MutateConnection(net);
  }
  else {
    // internal mutation (non-topological)
    if (random() < required()->mutate_weights) { MutateWeights(); }
    if (random() < required()->toggle_status) { MutateToggleGeneStatus(); }
    if (random() < required()->mutate_reenable) { MutateRenableGene(); }
  }



}

void Genome::MutateWeights() {
  if (!generator) { throw std::runtime_error("No RNG engine set. Replace this with class specific exception."); }
  for (auto& gene : connection_genes) {
    // perturb weight by a small amount
    if(random()<required()->perturb_weight) {
      gene.second.weight += required()->step_size*(2*random()-1);
    } else { // otherwise randomly set weight within reset range
      gene.second.weight = (random() - 0.5)*required()->reset_weight;
    }
  }
}

void Genome::MutateConnection(const NeuralNet& net) {

}

void Genome::MutateNode(const NeuralNet& net) {

}

void Genome::MutateToggleGeneStatus() {
  // Randomly toggle on/off a connection gene.
  // This is in the NEAT implementation but not discussed in the paper
  // Leaving this unimplemented for now. Note, that when this is added
  // a check will be needed that disabling a gene can only occur if
  // the destination node of that gene has other enabled input connections.
}

void Genome::MutateRenableGene() {

}

void Genome::PrintInnovations() {
  for (auto const& gene : connection_genes) {
    std::cout << gene.first << std::endl;
  }
}
