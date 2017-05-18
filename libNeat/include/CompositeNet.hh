#pragma once
#include "NeuralNet.hh"
#include "Genome.hh"

template<typename NetType>
std::unique_ptr<NeuralNet> BuildCompositeNet(const std::vector<Genome*>& genomes, bool hetero_inputs) {
  std::unique_ptr<NeuralNet> net = std::make_unique<NetType>();

  std::vector<std::unordered_set<unsigned int>> exclusion_lists(genomes.size());


  for (auto i=0u; i<genomes.size(); i++) {
    auto& genome = *genomes[i];

    genome.AssertInputNodesFirst();
    genome.AssertNoConnectionsToInput();


    ReachabilityChecker checker(genome.node_genes.size(),genome.num_inputs);
    for(auto& gene : genome.connection_genes) {
      if (gene.enabled) {
        int i = genome.node_lookup.at(gene.origin);
        int j = genome.node_lookup.at(gene.dest);
        checker.AddConnection(i,j);
      }
    }

    // use reachability checker to determine if a node is unconnected
    std::unordered_set<unsigned int>& exclusions = exclusion_lists[i];
    for (auto n=0u; n<genome.node_genes.size(); n++) {
      // if the node is not reachable from either inputs
      // or outputs, add to the exclusion list
      if (!genome.ConnectivityCheck(n,checker)) {
        exclusions.insert(n);
      }
    }
  }

  // manually add single bias node
  net->add_node(NodeType::Bias); // first

  // add input
  for (auto i=0u; i<genomes.size(); i++) {
    auto& genome = *genomes[i];

    // add all inputs assuming they are heterogenous across networks
    if (hetero_inputs || i==0) {
      for (auto& gene : genome.node_genes) {
        if (IsInput(gene.type)) {
          net->add_node(gene.type);
        }
      }
    }

  }

  // add output nodes
  auto num_outputs = 0;
  for (auto i=0u; i<genomes.size(); i++) {
    auto& genome = *genomes[i];

    // add all outputs
    for (auto& gene : genome.node_genes) {
      if (gene.type == NodeType::Output) {
        net->add_node(gene.type);
        num_outputs++;
      }
    }

  }

  // add hidden nodes
  for (auto i=0u; i<genomes.size(); i++) {
    auto& genome = *genomes[i];

    // add all hidden nodes
    for (auto& gene : genome.node_genes) {
      if (gene.type == NodeType::Hidden) {
        net->add_node(gene.type);
      }
    }

  }

  // add connections
  auto num_subnets = genomes.size();
  auto num_sensors_per_subnet = genomes[0]->num_inputs; // includes bias automatically
  auto num_outputs_per_subnet = num_outputs/num_subnets;

  auto subnet_input_node = 0;
  auto subnet_output_node = (hetero_inputs) ? num_subnets*(num_sensors_per_subnet-1) + 1 : num_sensors_per_subnet;
  auto subnet_hidden_node = subnet_output_node + num_subnets*num_outputs_per_subnet;

  for (auto n=0u; n<genomes.size(); n++) {
    auto& genome = *genomes[n];
    auto num_hidden = genome.node_genes.size() - num_sensors_per_subnet - num_outputs_per_subnet;


    for(auto& gene : genome.connection_genes) {
      if (gene.enabled) {
        int i = genome.node_lookup.at(gene.origin);
        int j = genome.node_lookup.at(gene.dest);

        int i_composite;
        int j_composite;
        // assumes inputs are before all other nodes, outputs are before hidden, hidden are the last nodes
        if (hetero_inputs) {
          i_composite
            = (IsBias(genome.node_genes[i].type)) ?  0
            : (IsInput(genome.node_genes[i].type)) ?  i + subnet_input_node
            : (IsOutput(genome.node_genes[i].type)) ? (i-num_sensors_per_subnet) + subnet_output_node
            : (i-num_sensors_per_subnet-num_outputs_per_subnet) + subnet_hidden_node; // hidden
          j_composite
            = (IsBias(genome.node_genes[j].type)) ?  0
            : (IsInput(genome.node_genes[j].type)) ?  j + subnet_input_node
            : (IsOutput(genome.node_genes[j].type)) ? (j-num_sensors_per_subnet) + subnet_output_node
            : (j-num_sensors_per_subnet-num_outputs_per_subnet) + subnet_hidden_node; // hidden
        } else {
          i_composite
            = (IsSensor(genome.node_genes[i].type)) ?  i
            : (IsOutput(genome.node_genes[i].type)) ? (i-num_sensors_per_subnet) + subnet_output_node
            : (i-num_sensors_per_subnet-num_outputs_per_subnet) + subnet_hidden_node; // hidden
          j_composite
            = (IsSensor(genome.node_genes[j].type)) ?  j
            : (IsOutput(genome.node_genes[j].type)) ? (j-num_sensors_per_subnet) + subnet_output_node
            : (j-num_sensors_per_subnet-num_outputs_per_subnet) + subnet_hidden_node; // hidden
        }

        if(exclusion_lists[n].count(i) == 0 &&
           exclusion_lists[n].count(j) == 0) {
          net->add_connection(i_composite,j_composite,gene.weight,n);
        }
      }
    }

    // increment the node pointers by the number of nodes of a specific type for the current subnet
    if (hetero_inputs){ subnet_input_node  += (num_sensors_per_subnet-1); }
    subnet_output_node += num_outputs_per_subnet;
    subnet_hidden_node += num_hidden;
  }

  auto& connections = net->get_connections();
  // sort composite net connections by the set index which is now the subnet set number
  std::sort(connections.begin(),connections.end(),[](const Connection& a, const Connection& b) { return a.set < b.set; });

  unsigned int first = 0;
  unsigned int set_size = 0;
  unsigned int prev_set = 0;
  for (auto i=0u; i<connections.size(); i++) {
    auto& conn = connections[i];

    if (conn.set != prev_set || i == connections.size()-1) {
      if (i == connections.size()-1) { set_size++; }
      if(set_size > 0) {
        net->sort_connections(first,set_size);
      }
      prev_set =conn.set;
      first = i;
      set_size = 0;
    }

    set_size++;
  }

  return net;
}
