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
    innovation = Hash(0,last_innov);
    break;
  case NodeType::Input:
    innovation = Hash(idxin++,last_innov);
    break;
  case NodeType::Output:
    innovation = Hash(idxout--,last_innov);
    break;
  case NodeType::Hidden:
    innovation = Hash(idxhidden++,last_innov);
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

  node_lookup.insert({node_genes[origin].innovation, origin});
  node_lookup.insert({node_genes[dest].innovation, dest});

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
    MutateNode();
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
  auto type = (random() < required()->add_recurrent) ?
    ConnectionType::Recurrent : ConnectionType::Normal;

  unsigned int idxorigin = 0;
  unsigned int idxdest = 0;

  while (true) {
    idxorigin = random()*node_genes.size();
    idxdest = random()*node_genes.size();

    if (IsSensor(node_genes[idxdest].type)) {
      continue; // this swap is disabled for now, see notice below
      if (IsSensor(node_genes[idxorigin].type)) {
        continue; // if both are sensor nodes
      }
      else {
        // swap origin and destination
        auto idxtemp = idxdest;
        idxdest = idxorigin;
        idxorigin = idxtemp;
        // Notice: this swap makes MutateConnection faster, but I found
        // that the distribution of found connections was less
        // uniform. For a small network, this might be an issue,
        // but I doubt it will make a difference for medium size
        // networks. Further investigation is necessary.
      }
    }

    bool unique = true;
    for (auto& gene : connection_genes) {
      if (gene.second.origin == node_genes[idxorigin].innovation &&
          gene.second.dest == node_genes[idxdest].innovation) {
        unique = false;
        break;
      }
    }
    if (unique) {
      int i = node_lookup[node_genes[idxorigin].innovation];
      int j = node_lookup[node_genes[idxdest].innovation];
      bool is_loop = net.would_make_loop(i,j);

      if ( (is_loop && type != ConnectionType::Recurrent) ||
           (!is_loop && type != ConnectionType::Normal)) { continue; }

      break; // origin and dest found for new connection

    }
    else { continue; }

  }
  //std::cout << idxorigin << " -> " << idxdest << std::endl;
  AddConnection(idxorigin,idxdest,true,(random()-0.5)*required()->reset_weight);
}

void Genome::MutateNode() {
  auto selected = connection_genes.begin();
  // pick random gene to splice
  do {
    selected = connection_genes.begin();
    unsigned int idxgene = random()*connection_genes.size();
    std::advance(selected,idxgene);
    // continue searching if the origin of the selected gene is the bias node
  }
  while(node_genes[node_lookup[selected->second.origin]].type == NodeType::Bias);

  // add a new node:
  // use the to-be disabled gene's innovation as ingredient for this new nodes innovation hash
  node_genes.emplace_back(NodeType::Hidden,Hash(selected->first,node_genes.back().innovation));
  // pre-gene: from selected genes origin to new node
  AddConnection(node_lookup[selected->second.origin],node_genes.size()-1,true,1.0);
  // post-gene: from new node to selected genes destination
  AddConnection(node_genes.size()-1,node_lookup[selected->second.dest],true,selected->second.weight);
  // disable old gene, and we're done
  selected->second.enabled = false;

  // Notice: since pre-gene is added _first_, if the selected gene is recurrent, the post-gene
  // will both have the weight of the selected gene, and be the recurrent gene; the pre-gene
  // will have weight 1.0 and be a normal gene (in NEAT they make the pre-gene be recurrent)
  // but for the same reason that the post-gene weight = selected-gene weight, I think it should
  // be the other way around.
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
  std::cout << std::endl;
  for (auto const& gene : connection_genes) {
    std::cout << "                Enabled: " << gene.second.enabled << "  |  "<< node_lookup[gene.second.origin] << " -> " << node_lookup[gene.second.dest] << "  Innovation: " << gene.first << std::endl;
  } std::cout << std::endl;
}
