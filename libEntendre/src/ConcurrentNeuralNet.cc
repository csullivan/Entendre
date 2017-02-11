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


void ConcurrentNeuralNet::sort_connections() {

  unsigned int max_iterations =
    connections.size()*connections.size()*connections.size()+1;

  bool change_applied = false;
  for(unsigned int i_try=0; i_try < max_iterations; i_try++) {
    change_applied = false;

    for(int i=0; i<connections.size(); i++) {
      for(int j=i+1; j<connections.size(); j++) {
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

ConcurrentNeuralNet::ConcurrentNeuralNet(ConsecutiveNeuralNet&& net)
  : NeuralNet(std::move(net.nodes),std::move(net.connections)) {
  sigma = net.sigma;
  connections_sorted = net.connections_sorted;
}


std::vector<double> ConcurrentNeuralNet::evaluate(std::vector<double> inputs) {
  sort_connections();
  load_input_vals(inputs);

  for(auto& conn : connections) {
    double input_val = get_node_val(conn.origin);
    add_to_val(conn.dest, input_val * conn.weight);
  }

  return read_output_vals();
}
