#include "ConcurrentNeuralNet.hh"

#include <cassert>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>
#include <algorithm>

#include "logging.h"

ConcurrentNeuralNet::EvaluationOrder ConcurrentNeuralNet::compare_connections(const Connection& a, const Connection& b) {
  // A recurrent connection must be used before the origin is overwritten.
  if (a.type == ConnectionType::Recurrent && a.origin == b.dest) { return EvaluationOrder::LessThan; }
  if (b.type == ConnectionType::Recurrent && b.origin == a.dest) { return EvaluationOrder::GreaterThan; }

  // A normal connection must occur after every connection incoming to its origin has completed.
  if (a.type == ConnectionType::Normal && a.dest == b.origin) { return EvaluationOrder::LessThan; }
  if (b.type == ConnectionType::Normal && b.dest == a.origin) { return EvaluationOrder::GreaterThan; }

  // Two connections writing to the same destination must be in different sets.
  if (a.dest == b.dest) {
    // A self-recurrent connection happens at the same time as
    // zero-ing out, and so must occur first of all connections
    // writing to that node.
    if(a.origin == a.dest) {
      return EvaluationOrder::LessThan;
    } else if (b.origin == b.dest) {
      return EvaluationOrder::GreaterThan;
    } else {
      // Choice here is absolutely arbitrary.
      // This is arbitrary, and consistent.
      if(a.origin < b.origin) {
        return EvaluationOrder::GreaterThan;
      } else {
        return EvaluationOrder::LessThan;
      }
    }
  }

  // else a & b are not adjacent and cannot be compared
  return EvaluationOrder::Unknown;
}

void ConcurrentNeuralNet::sort_connections() {
  if(connections_sorted) {
    return;
  }

  for(auto& conn : connections) {
    conn.set = 0;
  }
  for(auto i=0u; i<connections.size(); i++) {
    for(auto j=i+1; j<connections.size(); j++) {
      Connection& conn1 = connections[i];
      Connection& conn2 = connections[j];
      switch(compare_connections(conn1,conn2)) {
        case EvaluationOrder::GreaterThan:
          conn1.set++;
          break;

        case EvaluationOrder::LessThan:
          conn2.set++;
          break;

        case EvaluationOrder::Unknown:
          break;
      }
    }
  }

  auto split_iter = connections.begin();
  size_t current_set_num = 0;
  while(split_iter != connections.end()) {
    auto next_split = std::partition(split_iter, connections.end(),
                                     [](const Connection& conn) {
                                       return conn.set == 0;
                                     });
    assert(next_split != split_iter);

    // These could be run now, no longer need to track number of
    // depencencies.
    for(auto iter = split_iter; iter<next_split; iter++) {
      iter->set = current_set_num;
    }
    current_set_num++;

    // Decrease number of dependencies for everything else.
    for(auto iter_done = split_iter; iter_done<next_split; iter_done++) {
      for(auto iter_not_done = next_split; iter_not_done<connections.end(); iter_not_done++) {
        if (compare_connections(*iter_done,*iter_not_done) == EvaluationOrder::LessThan) {
          iter_not_done->set--;
        }
      }
    }

    split_iter = next_split;
  }

  build_action_list();
  connections_sorted = true;
}

void ConcurrentNeuralNet::ConcurrentNeuralNet::build_action_list() {

  unsigned int num_connection_sets = connections.back().set+1;
  std::vector<unsigned int> connection_set_sizes(num_connection_sets, 0);
  for(auto& conn : connections) {
    connection_set_sizes[conn.set]++;
  }

  std::vector<unsigned int> earliest_zero_out_indices(nodes.size(), 0);
  std::vector<unsigned int> earliest_sigmoid_indices(nodes.size(), 0);

  std::vector<unsigned int> latest_zero_out_indices(nodes.size(), num_connection_sets);
  std::vector<unsigned int> latest_sigmoid_indices(nodes.size(), num_connection_sets);

  std::set<unsigned int> self_recurrent_nodes;

  for(auto& conn : connections) {
    // delay earliest possible zeroing of recurrent connections origins
    // until recurrent connections are applied
    if(conn.type == ConnectionType::Recurrent) {
      earliest_zero_out_indices[conn.origin] = std::max(
        earliest_zero_out_indices[conn.origin],
        conn.set + 1);
    }

    earliest_sigmoid_indices[conn.dest] = std::max(
      earliest_sigmoid_indices[conn.dest],
      conn.set + 1);

    latest_zero_out_indices[conn.dest] = std::min(
      latest_zero_out_indices[conn.dest],
      conn.set);

    if(conn.type == ConnectionType::Normal) {
      latest_sigmoid_indices[conn.origin] = std::min(
        latest_sigmoid_indices[conn.origin],
        conn.set);
    }

    if(conn.origin == conn.dest) {
      self_recurrent_nodes.insert(conn.origin);
    }
  }

  std::vector<unsigned int>& zero_out_indices = earliest_zero_out_indices;
  std::vector<unsigned int>& sigmoid_indices = earliest_sigmoid_indices;


  std::vector<std::vector<unsigned int> > zero_out_sets(num_connection_sets+1);
  std::vector<std::vector<unsigned int> > sigmoid_sets(num_connection_sets+1);

  for(unsigned int i=0; i<nodes.size(); i++) {
    bool is_self_recurrent = self_recurrent_nodes.count(i);
    if(!is_self_recurrent && i >= num_inputs) {
      zero_out_sets[zero_out_indices[i]].push_back(i);
    }
    if(i >= num_inputs) {
      sigmoid_sets[sigmoid_indices[i]].push_back(i);
    }
  }

  auto append_zero_out_set = [&](unsigned int i) {
    auto& zero_out_set = zero_out_sets[i];
    action_list.push_back(zero_out_set.size());
    for(unsigned int j : zero_out_set) {
      action_list.push_back(j);
    }
  };

  auto append_sigmoid_set = [&](unsigned int i) {
    auto& sigmoid_set = sigmoid_sets[i];
    action_list.push_back(sigmoid_set.size());
    for(unsigned int j : sigmoid_set) {
      action_list.push_back(j);
    }
  };



  action_list.clear();
  for(unsigned int i=0; i<num_connection_sets; i++) {
    append_zero_out_set(i);
    append_sigmoid_set(i);
    action_list.push_back(connection_set_sizes[i]);
  }

  append_zero_out_set(num_connection_sets);
  append_sigmoid_set(num_connection_sets);



  // print action list
  // for (auto& item : action_list) {
  //   std::cout << item << " ";
  // } std::cout << std::endl;
}


////////////////////////////////////////////////////////////////////////////

void ConcurrentNeuralNet::add_node(const NodeType& type) {
  switch (type) {
  case NodeType::Bias:
  case NodeType::Input:
    num_inputs++;
    break;
  case NodeType::Output:
    num_outputs++;
    break;
  default: // all hidden nodes
    break;
  };
  nodes.push_back(0.0);
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

  return std::vector<_float_> (nodes.begin()+num_inputs,nodes.begin()+num_inputs+num_outputs);
}
