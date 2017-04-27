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

void ConcurrentNeuralNet::sort_connections(unsigned int first, unsigned int num_connections) {
  if(connections_sorted) {
    return;
  }

  // if the first connection in the list to sort is not
  // the first connection, and num_connections is zero
  // this is an error
  assert(!(first!=0 && num_connections==0));
  // if num_connections is zero, then we will sort all connections
  num_connections = num_connections > 0 ? num_connections : connections.size();
  // the number of connections to sort cannot be
  // larger than the total number of connections
  assert(first+num_connections <= connections.size());

  // zero out connection set index for use in sorting
  for (auto i=first; i<first+num_connections; i++) {
    connections[i].set = 0;
  }
  for(auto i=first; i<first+num_connections; i++) {
    for(auto j=i+1; j<first+num_connections; j++) {
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

  auto split_iter = connections.begin()+first;
  auto last_iter = connections.begin()+first+num_connections;
  size_t current_set_num = 0;
  while(split_iter != last_iter) {
    auto next_split = std::partition(split_iter, last_iter,
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
      for(auto iter_not_done = next_split; iter_not_done<last_iter; iter_not_done++) {
        if (compare_connections(*iter_done,*iter_not_done) == EvaluationOrder::LessThan) {
          iter_not_done->set--;
        }
      }
    }

    split_iter = next_split;
  }

  // build the action list if num_connections was the total set
  // or if this is the last subset of connections (all others are sorted)
  if (first + num_connections == connections.size()) {
    // if first is nonzero then we have been sorting based on subsets and now all subset lock free buckets
    // need to be merged in a sort of the entire connections list where set now is the lock free set index
    // (before it was used as the subnet index)
    if (first != 0) {
      // sort connections based on evaluation set number if not already done
      std::sort(connections.begin(),connections.end(),[](Connection a, Connection b){ return a.set < b.set; });
    }


    build_action_list();
    connections_sorted = true; // we are done sorting
  }

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
    num_inputs++;
    nodes.push_back(1.0);
    break;
  case NodeType::Input:
    num_inputs++;
    nodes.push_back(0.0);
    break;
  case NodeType::Output:
    num_outputs++;
    nodes.push_back(0.0);
    break;
  case NodeType::Hidden:
    nodes.push_back(0.0);
    break;
  };
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
  assert(inputs.size() == num_inputs-1);
  sort_connections();

  // copy inputs in to network
  std::copy(inputs.begin(),inputs.end(),nodes.begin()+1);

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


void ConcurrentNeuralNet::print_network(std::ostream& os) const {
  std::stringstream ss; ss.str("");
  ss << "Action List: \n\n";

  auto i = 0u;
  int how_many_zero_out = action_list[i++];
  ss << "# Zero out: " << how_many_zero_out << "\n";
  i += how_many_zero_out;

  int how_many_sigmoid = action_list[i++];
  ss << "# Sigmoid: " << how_many_sigmoid << "\n";
  i += how_many_sigmoid;

  std::vector<unsigned int> num_conn_to_apply;
  int current_conn = 0;
  while(i<action_list.size()) {
    int how_many_conn = action_list[i++];
    ss << "# Connections: " << how_many_conn << "\n";
    current_conn += how_many_conn;
    num_conn_to_apply.push_back(how_many_conn);

    int how_many_zero_out = action_list[i++];
    ss << "# Zero out: " << how_many_zero_out << "\n";
    i += how_many_zero_out;

    int how_many_sigmoid = action_list[i++];
    ss << "# Sigmoid: " << how_many_sigmoid << "\n";
    i += how_many_sigmoid;
  }
  os << ss.str();

  ss.str("");
  ss << "\nConnection sets:\n";
  int counter = 0;
  unsigned int num = num_conn_to_apply[counter];
  for (auto i=0u; i<connections.size(); i++) {
    ss << connections[i].origin << " -> " << connections[i].dest << "\n";
    if (i == num-1) { num += num_conn_to_apply[++counter]; ss << "\n";}
  }

  os << ss.str();
}
