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
}

std::vector<_float_> ConcurrentNeuralNet::evaluate(std::vector<_float_> inputs) {
  return inputs;
}

void ConcurrentNeuralNet::clear_nodes(int* list, int n) {
  for(int i=0; i<n; i++) {
    nodes[list[i]] = 0;
  }
}

void ConcurrentNeuralNet::sigmoid_nodes(int* list, int n) {
  for(int i=0; i<n; i++) {
    nodes[list[i]] = sigmoid(nodes[list[i]]);
  }
}

void ConcurrentNeuralNet::apply_connections(Connection* list, int n) {
  for(int i=0; i<n; i++) {
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


