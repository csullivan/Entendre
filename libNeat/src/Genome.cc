#include "Genome.hh"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <sstream>

#include "Hash.hh"

Genome::Genome() : num_inputs(0),
                   last_innovation(0) { }

void Genome::MakeNet(NeuralNet& net) const {
  AssertInputNodesFirst();
  AssertNoConnectionsToInput();

  // populate reachability checker
  ReachabilityChecker checker(node_genes.size(),num_inputs);
  for(auto& gene : connection_genes) {
    if (gene.enabled) {
      int i = node_lookup.at(gene.origin);
      int j = node_lookup.at(gene.dest);
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
  //NeuralNet net(node_genes);
  for (auto& gene: node_genes) {
    net.add_node(gene.type, gene.func);
  }
  for(auto& gene : connection_genes) {
    if (gene.enabled) {
      int i = node_lookup.at(gene.origin);
      int j = node_lookup.at(gene.dest);
      if(exclusions.count(i) == 0 &&
         exclusions.count(j) == 0) {
        net.add_connection(i,j,gene.weight);
      }
    }
  }
}


Genome Genome::ConnectedSeed(int num_inputs, int num_outputs,
                             ActivationFunction func) {
  Genome output;

  output.AddNode(NodeType::Bias);
  for(int i=0; i<num_inputs; i++) {
    output.AddNode(NodeType::Input);
  }
  for(int i=0; i<num_outputs; i++) {
    output.AddNode(NodeType::Output, func);
  }

  for(int from=0; from<num_inputs+1; from++) {
    for(int to=num_inputs+1; to<num_inputs+1+num_outputs; to++) {
      output.AddConnection(from, to, true, 1.0);
    }
  }

  return output;
}

//std::unique_ptr<NeuralNet> Genome::MakeNet() const {
//
//}

Genome& Genome::operator=(const Genome& rhs) {
  this->num_inputs = rhs.num_inputs;
  this->node_genes = rhs.node_genes;
  this->node_lookup = rhs.node_lookup;
  this->connection_genes = rhs.connection_genes;
  this->connection_lookup = rhs.connection_lookup;
  this->connections_existing = rhs.connections_existing;
  this->last_innovation = rhs.last_innovation;
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
  for (auto& gene : connection_genes) {
    // sum the absolute weight differences of the shared genes
    auto other_gene = other.GetConnByInnovation(gene.innovation);
    if(other_gene) {
      weight_diffs += std::abs(other_gene->weight - gene.weight);
      nShared++;
    }
    // count the number of unshared genes
    else {
      nUnshared++;
    }
  }
  // loop over other genomes genes and count the unshared genes
  for (auto& other_gene : other.connection_genes) {
    if (connection_lookup.count(other_gene.innovation) == 0) {
      nUnshared++;
    }
  }

  return
    required()->genetic_distance_structural*nUnshared +
    required()->genetic_distance_weights*weight_diffs/nShared;
}

Genome Genome::GeneticAncestry() const {
  Genome descendant;
  descendant.set_generator(this->get_generator());
  descendant.required(this->required());
  // Add all non-hidden nodes.
  for(auto& gene : this->node_genes) {
    if(gene.type == NodeType::Input ||
       gene.type == NodeType::Bias ||
       gene.type == NodeType::Output) {
      descendant.AddNodeByInnovation(gene.type, gene.func, gene.innovation);
    }
  }

  descendant.last_innovation = this->last_innovation;

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


  const auto& match          = required()->matching_gene_choose_mother;
  const auto& single_greater = required()->keep_non_matching_mother_gene;
  const auto& single_lesser  = required()->keep_non_matching_father_gene;

  auto add_conn_to_child = [](const auto& parent, const auto& conn_gene, auto& child) {
    const NodeGene* origin = parent.GetNodeByInnovation(conn_gene.origin);
    const NodeGene* dest = parent.GetNodeByInnovation(conn_gene.dest);
    assert(origin);
    assert(dest);
    child.AddNodeGene(*origin);
    child.AddNodeGene(*dest);
    child.AddConnectionGene(conn_gene);
  };

  for(auto& maternal_gene : mother.connection_genes) {
    // Find all shared genes, look up by hash
    if (father.connection_lookup.count(maternal_gene.innovation) > 0) {
      // matching genes
      if (random()<match) {
        // if key doesn't already exist in child,
        // then the maternal_gene gene is inserted
        add_conn_to_child(mother,maternal_gene,child);
      } else {
        // paternal_gene gene is taken
        auto& paternal_gene = father.connection_genes.at(father.connection_lookup.at(maternal_gene.innovation));
        add_conn_to_child(father,paternal_gene,child);
      }
    } else {
      // non matching gene, randomly insert maternal_gene gene
      if (random()<single_greater) {
        add_conn_to_child(mother,maternal_gene,child);
      }
    }
  }

  // Standard NEAT bails out here
  if (single_lesser > 0.0) {
    // allow for merging of structure from less fit parent
    for(auto& paternal_gene : father.connection_genes) {
      if(mother.connection_lookup.count(paternal_gene.innovation) == 0 &&
         random()<single_lesser) {
        add_conn_to_child(father,paternal_gene,child);
      }
    }
  }

  assert(child.node_genes.size() == child.node_lookup.size());
  assert(child.connection_genes.size() == child.connection_lookup.size());
  child.AssertNoDuplicateConnections();

  return child;
}

Genome& Genome::RandomizeWeights() {
  for (auto& gene : connection_genes) {
    gene.weight = (random() - 0.5)*required()->weight_mutation_reset_range;
  }
  return *this;
}

Genome& Genome::AddNode(NodeType type, ActivationFunction func) {
  // User-defined nodes, no real innovation number
  // Instead, make something up to ensure unique ids for each.

  unsigned long innovation = Hasher::hash(last_innovation, type, func);

  return AddNodeByInnovation(type, func, innovation);
}

Genome& Genome::AddNodeByInnovation(NodeType type, ActivationFunction func,
                                    unsigned long innovation) {
  assert(node_lookup.count(innovation) == 0);
  NodeGene gene(type, func, innovation);
  AddNodeGene(gene);
  return *this;
}

Genome& Genome::AddConnection(unsigned long origin, unsigned long dest,
                              bool status, double weight) {
  // build look up table from innovation hash to vector index
  // node_lookup.insert({node_genes[origin].innovation, origin});
  // node_lookup.insert({node_genes[dest].innovation, dest});

  AddConnectionByInnovation(node_genes[origin].innovation,
                            node_genes[dest].innovation,
                            status, weight);
  return *this;
}

void Genome::AddConnectionByInnovation(unsigned long origin, unsigned long dest,
                                       bool status, double weight) {
  ConnectionGene gene;
  gene.innovation = Hasher::hash(last_innovation, origin, dest);
  gene.origin = origin;
  gene.dest = dest;
  gene.weight = weight;
  gene.enabled = status;

  AddConnectionGene(gene);
}

void Genome::AddNodeGene(NodeGene gene) {
  if(node_lookup.count(gene.innovation) != 0) {
    return;
  }

  bool needs_resort = IsSensor(gene.type) && (node_genes.size() != num_inputs);

  if(IsSensor(gene.type)) {
    num_inputs++;
  }


  if(needs_resort) {
    node_genes.push_back(gene);

    // Move all sensor nodes to the front of the list.
    std::stable_partition(node_genes.begin(), node_genes.end(),
                          [](const NodeGene& gene) { return IsSensor(gene.type); }
    );
    // Regenerate the lookup table.
    node_lookup.clear();
    for(unsigned int i=0; i<node_genes.size(); i++) {
      node_lookup[node_genes[i].innovation] = i;
    }
  } else {
    // Just add the new gene to the lookup.
    node_lookup[gene.innovation] = node_genes.size();
    node_genes.push_back(gene);
  }

  last_innovation = gene.innovation;

  AssertInputNodesFirst();
}

void Genome::AddConnectionGene(ConnectionGene gene) {
  if(connection_lookup.count(gene.innovation) != 0 ||
     node_lookup.count(gene.origin) == 0 ||
     node_lookup.count(gene.dest) == 0 ||
     connections_existing.count({gene.origin, gene.dest}) ||
     IsSensor(GetNodeByInnovation(gene.dest)->type)) {
    return;
  }

  auto dest = GetNodeByInnovation(gene.dest);
  assert(dest);
  assert(!IsSensor(dest->type));

  connection_lookup[gene.innovation] = connection_genes.size();
  connection_genes.push_back(gene);
  connections_existing.insert({gene.origin, gene.dest});
  last_innovation = gene.innovation;
}

const NodeGene* Genome::GetNodeByN(unsigned int i) const {
  if(i < node_genes.size()) {
    return &node_genes[i];
  } else {
    return nullptr;
  }
}

const NodeGene* Genome::GetNodeByInnovation(unsigned long innovation) const {
  auto iter = node_lookup.find(innovation);
  if(iter != node_lookup.end()) {
    return &node_genes[iter->second];
  } else {
    return nullptr;
  }
}

const ConnectionGene* Genome::GetConnByN(unsigned int i) const {
  if(i < connection_genes.size()) {
    return &connection_genes[i];
  } else {
    return nullptr;
  }
}

const ConnectionGene* Genome::GetConnByInnovation(unsigned long innovation) const {
  auto iter = connection_lookup.find(innovation);
  if(iter != connection_lookup.end()) {
    return &connection_genes[iter->second];
  } else {
    return nullptr;
  }
}

NodeGene* Genome::GetNodeByN(unsigned int i) {
  return const_cast<NodeGene*>(
    const_cast<const Genome*>(this)->GetNodeByN(i)
  );
}

NodeGene* Genome::GetNodeByInnovation(unsigned long innovation) {
  return const_cast<NodeGene*>(
    const_cast<const Genome*>(this)->GetNodeByInnovation(innovation)
  );
}

ConnectionGene* Genome::GetConnByN(unsigned int i) {
  return const_cast<ConnectionGene*>(
    const_cast<const Genome*>(this)->GetConnByN(i)
  );
}

ConnectionGene* Genome::GetConnByInnovation(unsigned long innovation) {
  return const_cast<ConnectionGene*>(
    const_cast<const Genome*>(this)->GetConnByInnovation(innovation)
  );
}

void Genome::Mutate() {

  // structural mutation
  if (random() < required()->mutation_prob_add_node) {
    MutateNode();
  }
  if (random() < required()->mutation_prob_add_connection) {
    MutateConnection();
  }
  // internal mutation (non-topological)
  if (random() < required()->mutation_prob_adjust_weights) { MutateWeights(); }
  if (random() < required()->mutation_prob_toggle_connection) { MutateToggleGeneStatus(); }
  if (random() < required()->mutation_prob_reenable_connection) { MutateReEnableGene(); }
}

void Genome::MutateWeights() {
  bool is_severe = random() < required()->weight_mutation_is_severe;

  for (auto& gene : connection_genes) {

    if(is_severe) {// caution to the wind, reset everything!
      gene.weight = (random() - 0.5)*required()->weight_mutation_reset_range;
    } else { // otherwise perturb weight by a small amount
      gene.weight += required()->weight_mutation_small_adjust*(2*random()-1);
    }
  }
}

void Genome::MutateConnection() {
  ReachabilityChecker checker(node_genes.size(),num_inputs);
  for(auto& gene : connection_genes) {
    int i = node_lookup.at(gene.origin);
    int j = node_lookup.at(gene.dest);
    checker.AddConnection(i,j);
  }

  int idxorigin, idxdest;

  bool add_recurrent = random() < required()->new_connection_is_recurrent;
  if (add_recurrent) {
    // recurrent
    std::tie(idxorigin,idxdest) = checker.RandomRecurrentConnection(*get_generator());
  }else {
    // normal
    std::tie(idxorigin,idxdest) = checker.RandomNormalConnection(*get_generator());
  }

  if (idxorigin == -1 || idxdest == -1) {
    return;
  }

  AddConnection(idxorigin, idxdest, true, (random()-0.5)*required()->weight_mutation_reset_range);
}

void Genome::MutateNode() {
  const int n_tries = 20;

  // The connection to be split.
  // Note that we can't use a pointer or iterator,
  //    because that would be invalidated when we append to connection_genes.
  ConnectionGene split_conn;
  bool found = false;
  // pick random gene to splice
  for(int i=0; i<n_tries; i++) {
    split_conn = connection_genes[random()*connection_genes.size()];

    // can splice any connection gene that is enabled, and doesn't come from the bias node
    if(GetNodeByInnovation(split_conn.origin)->type != NodeType::Bias &&
       split_conn.enabled) {
      found = true;
      break;
    }
  }

  if(!found) {
    return;
  }

  // add a new node:
  // use the to-be disabled gene's innovation as ingredient for this new nodes innovation hash
  auto new_node_innov = Hasher::hash(last_innovation, split_conn.innovation);
  // TODO:  1. HyperNEAT performs a random roulette choice based on probabilities
  //        set for each activation function which are exposed in the parameters.
  //        Currently, all activation functions have equal probability
  //        2. Additionally, the node type should be used in innovation hashing
  auto func = required()->use_compositional_pattern_producing_networks ?
    ActivationFunction(random()*(int(ActivationFunction::MaxNodeType)+1)) :
    ActivationFunction::Sigmoid;

  AddNodeByInnovation(NodeType::Hidden, func, new_node_innov);

  // disable old gene
  GetConnByInnovation(split_conn.innovation)->enabled = false;
  // pre-gene: from selected genes origin to new node
  AddConnectionByInnovation(split_conn.origin, new_node_innov, true, 1.0);
  // post-gene: from new node to selected genes destination
  AddConnectionByInnovation(new_node_innov, split_conn.dest, true, split_conn.weight);

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
  selected->enabled = !selected->enabled;
}

void Genome::MutateReEnableGene() {
  std::vector<unsigned int> disabled_indices;
  // create a list of all disabled connection genes
  for (auto i = 0u; i <connection_genes.size(); i++) {
    const auto& gene = connection_genes[i];
    if (gene.enabled == false) { disabled_indices.push_back(i); }
  }
  // if no disabled genes exist, bail out
  if (!disabled_indices.size()) { return; }
  // pick a random disabled gene to reenable
  unsigned int idx = random()*disabled_indices.size();
  auto selected = connection_genes.begin();
  std::advance(selected,disabled_indices[idx]);
  selected->enabled = true;
}

void Genome::PrintInnovations() const {
  std::cout << std::endl;
  for (auto const& gene : connection_genes) {
    std::cout << "                Enabled: " << gene.enabled << "  |  "<< gene.origin << " -> " << gene.dest << std::endl;
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

bool Genome::IsStructurallyEqual(const Genome& other) const {
  if(node_genes.size() != other.node_genes.size() ||
     connection_genes.size() != other.connection_genes.size()) {
    return false;
  }

  for(auto& gene : node_genes) {
    if(!other.GetNodeByInnovation(gene.innovation)) {
      return false;
    }
  }

  for(auto& gene : connection_genes) {
    if(!other.GetConnByInnovation(gene.innovation)) {
      return false;
    }
  }

  return true;
}

void Genome::AssertNoDuplicateConnections() const {
  for(auto& gene1 : connection_genes) {
    for(auto& gene2 : connection_genes) {
      assert( &gene1 == &gene2 ||
              gene1.origin != gene2.origin ||
              gene1.dest != gene2.dest);
    }
  }
}

void Genome::AssertInputNodesFirst() const {
  for(unsigned int i=0; i<num_inputs; i++) {
    assert(IsSensor(node_genes[i].type));
  }
  for(unsigned int i=num_inputs; i<node_genes.size(); i++) {
    assert(!IsSensor(node_genes[i].type));
  }
}

void Genome::AssertNoConnectionsToInput() const {
  for(auto& conn_gene : connection_genes) {
    auto dest = GetNodeByInnovation(conn_gene.dest);
    assert(!IsSensor(dest->type));
  }
}

std::ostream& operator<<(std::ostream& os, const Genome& genome) {
  std::map<unsigned int, std::string> names;
  unsigned int num_inputs = 0;
  unsigned int num_outputs = 0;
  unsigned int num_hidden = 0;

  for(unsigned int i=0; i<genome.node_genes.size(); i++) {
    std::stringstream ss;
    switch(genome.node_genes[i].type) {
      case NodeType::Input:
        ss << "I" << num_inputs++;
        break;
      case NodeType::Output:
        ss << "O" << num_outputs++;
        break;
      case NodeType::Bias:
        ss << "B";
        break;
      default: // all hidden nodes
        ss << "H" << num_hidden++;
        break;
    }
    names[genome.node_genes[i].innovation] = ss.str();
  }

  for(const auto& item : names) {
    std::cout << "Node " << item.first << " = " << item.second << std::endl;
  }

  for (auto n=0u; n<genome.connection_genes.size(); n++) {
    auto&& conn = genome.connection_genes[n];

    os << names[conn.origin]
       << " ---> "
       << names[conn.dest];

    if(n != genome.connection_genes.size()-1) {
      os << "\n";
    }
  }
  return os;
}
