#include "ConcurrentNeuralNet.hh"
#include "ConsecutiveNeuralNet.hh"

#include <cassert>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>
#include <algorithm>

void ConcurrentNeuralNet::sort_connections() {
  if(connections_sorted) {
    return;
  }

  unsigned int max_iterations =
    connections.size()*connections.size()*connections.size()+1;

  bool change_applied = false;
  for(auto i_try=0u; i_try < max_iterations; i_try++) {
    change_applied = false;

    for(auto i=0u; i<connections.size(); i++) {
      for(auto j=i+1; j<connections.size(); j++) {
        Connection& conn1 = connections[i];
        Connection& conn2 = connections[j];

        switch(compare_connections(conn1,conn2)) {
        case EvaluationOrder::GreaterThan:
          if (conn1.set <= conn2.set) {
            conn1.set = conn2.set + 1;
            change_applied = true;
          }
          break;

        case EvaluationOrder::LessThan:
          if(conn2.set <= conn1.set) {
            conn2.set = conn1.set + 1;
            change_applied = true;
          }
          break;

        case EvaluationOrder::NotEqual:
          if(conn1.set == conn2.set) {
            conn2.set = conn1.set + 1;
            change_applied = true;
          }
          break;

        case EvaluationOrder::Unknown:
          break;
        }
      }
    }

    if(!change_applied) {
      break;
    }
  }

  if(change_applied) {
    throw std::runtime_error("Sort Error: change_applied == true on last possible iteration");
  }

  // sort connections based on evaluation set number
  std::sort(connections.begin(),connections.end(),[](Connection a, Connection b){ return a.set < b.set; });
  connections_sorted = true;

  build_action_list();
}

void ConcurrentNeuralNet::ConcurrentNeuralNet::build_action_list() {

  std::map<unsigned int, size_t> connection_sets;
  for (auto const& conn : connections) { connection_sets[conn.set]++; }

  std::set<unsigned int> cleared_nodes, sigmoided_nodes;

  unsigned int begin = 0;
  for (auto& set : connection_sets) {
    size_t end = begin+set.second;

    // enumerate all destination nodes of the current set
    std::vector<unsigned int> origins, destinations;
    for (auto i = begin; i < end; i++) {
      origins.push_back(connections[i].origin);
      destinations.push_back(connections[i].dest);
    }
    begin = end;

    // determine if node has been cleared in prior set
    size_t num_to_clear = 0;
    std::vector<unsigned int> nodes_to_clear;
    for (auto& node : destinations) {
      if (cleared_nodes.count(node)==0) {
        num_to_clear++;
        nodes_to_clear.push_back(node);
        cleared_nodes.insert(node);
      }
    }
    // determine if node has been sigmoided in prior set
    size_t num_to_sigmoid = 0;
    std::vector<unsigned int> nodes_to_sigmoid;
    for (auto& node : origins) {
      if (node >= num_inputs && sigmoided_nodes.count(node)==0) {
        num_to_sigmoid++;
        nodes_to_sigmoid.push_back(node);
        sigmoided_nodes.insert(node);
      }
    }

    // add only new to-be cleared nodes
    action_list.push_back(num_to_clear);
    for (auto& node : nodes_to_clear) {
      action_list.push_back(node);
    }
    // add only new to-be sigmoided nodes
    action_list.push_back(num_to_sigmoid);
    for (auto& node : nodes_to_sigmoid) {
      action_list.push_back(node);
    }
    // add the number of connections to evaluate
    action_list.push_back(set.second);

    // if this is the last connection set
    if (begin == connections.size()) {
      // add final # of nodes to be cleared (none)
      action_list.push_back(0);

      // determine if node has been sigmoided in prior set
      num_to_sigmoid = 0;
      nodes_to_sigmoid.clear();
      for (auto& node : destinations) {
        if (sigmoided_nodes.count(node)==0) {
          num_to_sigmoid++;
          nodes_to_sigmoid.push_back(node);
          sigmoided_nodes.insert(node);
        }
      }

      // sigmoid output nodes
      action_list.push_back(num_to_sigmoid);
      for (auto& node : nodes_to_sigmoid) {
        action_list.push_back(node);
      }

    }
  }

  // print action list
  // for (auto& item : action_list) {
  //   std::cout << item << " ";
  // } std::cout << std::endl;
}


ConcurrentNeuralNet::EvaluationOrder ConcurrentNeuralNet::compare_connections(const Connection& a, const Connection& b) {
  if (a.type == ConnectionType::Recurrent && a.origin == b.dest) { return EvaluationOrder::LessThan; }
  if (b.type == ConnectionType::Recurrent && b.origin == a.dest) { return EvaluationOrder::GreaterThan; }
  if (a.type == ConnectionType::Normal && a.dest == b.origin) { return EvaluationOrder::LessThan; }
  if (b.type == ConnectionType::Normal && b.dest == a.origin) { return EvaluationOrder::GreaterThan; }
  if (a.dest == b.dest) { return EvaluationOrder::NotEqual; }
  return EvaluationOrder::Unknown;
}

////////////////////////////////////////////////////////////////////////////

std::vector<_float_> node_values(std::vector<Node>& nodes) {
  return std::vector<_float_>(nodes.begin(),nodes.end());
}

ConcurrentNeuralNet::ConcurrentNeuralNet(ConsecutiveNeuralNet&& net)
  : NeuralNet(node_values(net.nodes),std::move(net.connections)) {
  sigma = net.sigma;
  connections_sorted = net.connections_sorted;
  for (auto const& node : net.nodes) {
    if (IsSensor(node.type)) { num_inputs++; }
    else if (node.type == NodeType::Output) { num_outputs++; }
  }
}

void ConcurrentNeuralNet::clear_nodes(unsigned int* list, unsigned int n) {
  for(auto i=0u; i<n; i++) {
    nodes[list[i]] = 0;
  }
}

void ConcurrentNeuralNet::sigmoid_nodes(unsigned int* list, unsigned int n) {
  for(auto i=0u; i<n; i++) {
    nodes[list[i]] = sigmoid(nodes[list[i]]);
  }
}

void ConcurrentNeuralNet::apply_connections(Connection* list, unsigned int n) {
  for(auto i=0u; i<n; i++) {
    Connection& conn = list[i];
    if(conn.origin == conn.dest) {
      // Special case for self-recurrent nodes
      // Be sure not to zero-out before this step.
      nodes[conn.origin] *= conn.weight;
    } else {
      nodes[conn.dest] += conn.weight*nodes[conn.origin];
    }
  }
}

std::vector<_float_> ConcurrentNeuralNet::evaluate(std::vector<_float_> inputs) {
  assert(inputs.size() == num_inputs);
  sort_connections();

  // copy inputs in to network
  std::copy(inputs.begin(),inputs.end(),nodes.begin());

  auto i = 0u;
  int how_many_zero_out = action_list[i++];
  clear_nodes(&action_list[i], how_many_zero_out);
  i += how_many_zero_out;

  int how_many_sigmoid = action_list[i++];
  sigmoid_nodes(&action_list[i], how_many_sigmoid);
  i += how_many_sigmoid;

  int current_conn = 0;
  while(i<action_list.size()) {
    int how_many_conn = action_list[i++];
    apply_connections(&connections[current_conn], how_many_conn);
    current_conn += how_many_conn;

    int how_many_zero_out = action_list[i++];
    clear_nodes(&action_list[i], how_many_zero_out);
    i += how_many_zero_out;

    int how_many_sigmoid = action_list[i++];
    sigmoid_nodes(&action_list[i], how_many_sigmoid);
    i += how_many_sigmoid;
  }

  std::vector<_float_> outputs(nodes.end()-num_outputs,nodes.end());

  return outputs;
}
