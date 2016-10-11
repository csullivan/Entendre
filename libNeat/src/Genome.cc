#include "Genome.hh"
#include <algorithm>
#include <iostream>
#include <climits>
#include <unordered_set>
#include "ReachabilityChecker.hh"

Genome::Genome() : num_inputs(0), last_conn_innov(0), last_node_innov(0),
                   idxinput(0), idxoutput(0), idxhidden(ULONG_MAX/2) { ; }

Genome::operator NeuralNet() const{

  // populate reachability checker
  ReachabilityChecker checker(node_genes.size(),num_inputs);
  for (auto n=0u; n<connection_genes.size(); n++) {
    if (connection_genes[n].second.enabled) {
      int i = node_lookup.at(connection_genes[n].second.origin);
      int j = node_lookup.at(connection_genes[n].second.dest);
      checker.AddConnection(i,j);
    }
  }

  // use reachability checker to determine if a node is unconnected
  std::unordered_set<unsigned int> exclusions;
  for (auto n=0u; n<node_genes.size(); n++) {
    // if the node is not reachable from either inputs
    // or outputs, add to the exclusion list
    if (!ConnectivityCheck(n,checker)) {
      exclusions.insert(n);
    }
  }

  // build neural net from only genes that connect to nodes
  // which have a path to an output and from an input
  NeuralNet net(node_genes);
  for (auto n=0u; n<connection_genes.size(); n++) {
    if (connection_genes[n].second.enabled) {
      int i = node_lookup.at(connection_genes[n].second.origin);
      int j = node_lookup.at(connection_genes[n].second.dest);
      if (exclusions.count(i) || exclusions.count(j)) { continue; }
      net.add_connection(i,j,connection_genes[n].second.weight);
    }
  }


  return net;
}

Genome Genome::operator=(const Genome& rhs) {
  this->num_inputs = rhs.num_inputs;
  this->node_genes = rhs.node_genes;
  this->connection_genes = rhs.connection_genes;
  this->node_lookup = rhs.node_lookup;
  this->last_conn_innov = rhs.last_conn_innov;
  this->last_node_innov = rhs.last_node_innov;
  this->idxinput = rhs.idxinput;
  this->idxoutput = rhs.idxoutput;
  this->idxhidden = rhs.idxhidden;
  this->generator = rhs.generator;
  this->required_ = rhs.required_;
  return *this;
}

float Genome::GeneticDistance(const Genome& other) const {
  double weight_diffs = 0.0;
  unsigned long nUnshared = 0;
  unsigned long nShared = 0;
  auto nGenes = std::max(connection_genes.size(),other.connection_genes.size());

  // loop over this genomes genes
  for (auto const& gene : connection_genes) {
    auto other_gene = other.connection_genes.find(gene.first);
    // sum the absolute weight differences of the shared genes
    if (other_gene != other.connection_genes.end()) {
      weight_diffs += std::abs(other_gene->second.weight-gene.second.weight);
      nShared++;
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
  //return (required()->genetic_c1*nUnshared)/nGenes + required()->genetic_c2*weight_diffs/nShared;
  //return (required()->genetic_c1*nUnshared) + required()->genetic_c2*weight_diffs/nShared;
}

Genome Genome::GeneticAncestry() const {
  Genome descendant;
  descendant.last_conn_innov = this->last_conn_innov;
  descendant.last_node_innov = this->last_node_innov;
  descendant.set_generator(this->get_generator());
  descendant.required(this->required());
  // add sensor nodes
  for (auto i=0u; i<num_inputs; i++) {
    descendant.AddNode(this->node_genes[i].type);
  }
  // add outputs
  for (auto i=0u; i<node_genes.size(); i++) {
    if (this->node_genes[i].type == NodeType::Output) {
      descendant.AddNode(this->node_genes[i].type);
    }
  }
  return descendant;
}

Genome Genome::MateWith(Genome* father) {
  return MateWith(*father);
}
// Generalized genome crossover
Genome Genome::MateWith(const Genome& father) {
  // Implicit assumption: Mother must always be the more
  // fit genome. i.e. child = mother(father) such that
  // fitness(mother) > fitness(father)
  auto& mother = *this;
  auto child = this->GeneticAncestry();


  const auto& match          = required()->match;
  const auto& single_greater = required()->single_greater;
  const auto& single_lesser  = required()->single_lesser;

  auto add_node_to_child = [](const auto& parent_genome, const auto& parent, auto& child) {
    auto origin_node_innov = parent.second.origin;
    auto dest_node_innov = parent.second.dest;
    if (child.node_lookup.count(origin_node_innov) == 0) {
      child.node_genes.emplace_back(
        parent_genome.node_genes[parent_genome.node_lookup.at(origin_node_innov)].type,
        origin_node_innov);
      child.node_lookup[origin_node_innov] = child.node_genes.size()-1;
    }
    if (child.node_lookup.count(dest_node_innov) == 0) {
      child.node_genes.emplace_back(
        parent_genome.node_genes[parent_genome.node_lookup.at(dest_node_innov)].type,
        dest_node_innov);
      child.node_lookup[dest_node_innov] = child.node_genes.size()-1;
    }
  };


  // NOTICE: This loop needs to be over the interal vector of keys,
  // since the order is not preserved when looping over the map
  for (auto i = 0u; i< mother.connection_genes.size(); i++) {
    auto const& maternal_gene = mother.connection_genes[i];
    // Find all shared genes, look up by hash
    auto paternal_gene = father.connection_genes.find(maternal_gene.first);
    if (paternal_gene != father.connection_genes.end()) {
      // matching genes
      if (random()<match) {
        // if key doesn't already exist in child,
        // then the maternal_gene gene is inserted
        child.connection_genes.insert(maternal_gene);
        add_node_to_child(mother,maternal_gene,child);
      } else {

        // paternal_gene gene is taken
        child.connection_genes.insert(*paternal_gene);
        add_node_to_child(father,*paternal_gene,child);
      }
    } else {
      // non matching gene, randomly insert maternal_gene gene
      if (random()<single_greater) {
        child.connection_genes.insert(maternal_gene);
        add_node_to_child(mother,maternal_gene,child);
      }
    }
  }


  // Standard NEAT bails out here
  if (single_lesser == 0.0) { return child; }

  // allow for merging of structure from less fit parent
  for (auto i = 0u; i < father.connection_genes.size(); i++) {
    auto const& paternal_gene = father.connection_genes[i];
    if (random()<single_lesser) {
      child.connection_genes.insert(paternal_gene);
      add_node_to_child(father,paternal_gene,child);
    }
  }

  return child;
}

Genome& Genome::RandomizeWeights() {
  for (auto& gene : connection_genes) {
    gene.second.weight = (random() - 0.5)*required()->reset_weight;
  }
  return *this;
}

Genome& Genome::AddNode(NodeType type) {
  // TO DO: consider whether idxhidden is appropriately defined


  if (IsSensor(type)) { num_inputs++; }

  unsigned long innovation = 0;
  switch(type) {
  case NodeType::Bias:
    innovation = Hash(0,last_node_innov);
    break;
  case NodeType::Input:
    innovation = Hash(idxinput++,last_node_innov);
    break;
  case NodeType::Output:
    innovation = Hash(idxoutput--,last_node_innov);
    break;
  case NodeType::Hidden:
    innovation = Hash(idxhidden++,last_node_innov);
    break;
  }
  // add to lookup table
  node_lookup.insert({innovation, node_genes.size()});
  // add to node gene list
  node_genes.emplace_back(type,innovation);
  last_node_innov = innovation;
  return *this;
}

Genome& Genome::AddConnection(unsigned long origin, unsigned long dest,
                              bool status, double weight) {

  unsigned long innovation = 0;

  // first gene only
  if (last_conn_innov == 0) {
    last_conn_innov = Hash(origin,dest,0);
  }

  // build look up table from innovation hash to vector index

  node_lookup.insert({node_genes[origin].innovation, origin});
  node_lookup.insert({node_genes[dest].innovation, dest});

  innovation = Hash(node_genes[origin].innovation,
                    node_genes[dest].innovation,
                    last_conn_innov);

  last_conn_innov = innovation;
  connection_genes.insert({innovation,{node_genes[origin].innovation,node_genes[dest].innovation,weight,status}});
  return *this;
}

void Genome::Mutate() {

  // structural mutation
  if (random() < required()->mutate_node) {
    MutateNode();
  }
  if (random() < required()->mutate_link) {
    MutateConnection();
  }
  // internal mutation (non-topological)
  if (random() < required()->mutate_weights) { MutateWeights(); }
  if (random() < required()->toggle_status) { MutateToggleGeneStatus(); }
  //if (random() < required()->mutate_reenable) { MutateReEnableGene(); }


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

void Genome::MutateConnection() {
  ReachabilityChecker checker(node_genes.size(),num_inputs);
  for (auto n=0u; n<connection_genes.size(); n++) {
    int i = node_lookup.at(connection_genes[n].second.origin);
    int j = node_lookup.at(connection_genes[n].second.dest);
    checker.AddConnection(i,j);
  }

  int idxorigin, idxdest;

  if (random() < required()->add_recurrent) {
    // recurrent
    std::tie(idxorigin,idxdest) = checker.RandomRecurrentConnection(*get_generator());
  }else {
    // normal
    std::tie(idxorigin,idxdest) = checker.RandomNormalConnection(*get_generator());
  }

  if (idxorigin == -1 || idxdest == -1) {
    return;
  }

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
  // Leaving this unimplemented for now. Note, that this could cause a
  // dangling node such that it has no inputs beyon the to-be disabled gene.
  // The NeuralNet will handle this by searching for said dangling nodes
  // and remove them from the derived network. Thus this function is quite
  // simple.

  unsigned int idx = random()*connection_genes.size();
  auto selected = connection_genes.begin();
  std::advance(selected,idx);
  selected->second.enabled = !selected->second.enabled;
}

void Genome::MutateReEnableGene() {
  std::vector<unsigned int> disabled_indices;
  // create a list of all disabled connection genes
  for (auto i = 0u; i <connection_genes.size(); i++) {
    const auto& gene = connection_genes[i].second;
    if (gene.enabled == false) { disabled_indices.push_back(i); }
  }
  // if no disabled genes exist, bail out
  if (!disabled_indices.size()) { return; }
  // pick a random disabled gene to reenable
  unsigned int idx = random()*disabled_indices.size();
  auto selected = connection_genes.begin();
  std::advance(selected,disabled_indices[idx]);
  selected->second.enabled = true;
}

void Genome::PrintInnovations() const {
  std::cout << std::endl;
  for (auto const& gene : connection_genes) {
    std::cout << "                Enabled: " << gene.second.enabled << "  |  "<< gene.second.origin << " -> " << gene.second.dest << std::endl;
  } std::cout << std::endl;
}


bool Genome::ConnectivityCheck(unsigned int node_index, const ReachabilityChecker& checker) const {
  // bias node should always be present
  if (node_genes[node_index].type == NodeType::Bias) { return true; }

  // check that there is a path from at least one input to the current node
  bool path_from_input = false;
  for (auto input_index = 0u; input_index < node_genes.size(); input_index++) {
    const NodeGene& node = node_genes[input_index];
    if (node.type != NodeType::Input) { continue; }
    if (checker.IsReachableEither(input_index,node_index)) {
      path_from_input = true; break;
    }
  }
  if (!path_from_input) { return false; }

  // check that there is a path from at least one output to the current node
  bool path_from_output = false;
  for (auto output_index = 0u; output_index < node_genes.size(); output_index++) {
    const NodeGene& node = node_genes[output_index];
    if (node.type != NodeType::Output) { continue; }
    if (checker.IsReachableEither(node_index,output_index)) {
      path_from_output = true; break;
    }
  }
  if (!path_from_output) { return false; }

  return true;
}
