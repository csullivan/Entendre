#include "ConsecutiveNeuralNet.hh"

#include <cassert>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>


std::vector<double> ConsecutiveNeuralNet::evaluate(std::vector<double> inputs) {
  sort_connections();
  load_input_vals(inputs);

  for(auto& conn : connections) {
    double input_val = get_node_val(conn.origin);
    add_to_val(conn.dest, input_val * conn.weight);
  }

  return read_output_vals();
}

void ConsecutiveNeuralNet::sort_connections() {
  if(connections_sorted) {
    return;
  }

  auto num_connections = connections.size();

  std::vector<Connection> sorted;
  sorted.reserve(num_connections);

  std::vector<bool> used(num_connections,false);

  for(size_t i = 0; i<num_connections; i++) {
    size_t possible;
    for(possible = 0; possible<num_connections; possible++) {

      if (used[possible]) { continue; }

      Connection& conn = connections[possible];
      bool disqualified = false;

      // Origin of normal connection has no unused input connections
      if(conn.type == ConnectionType::Normal) {
        for(size_t j=0; j<num_connections; j++) {
          Connection& other = connections[j];
          if(!used[j] &&
             conn.origin == other.dest) {
            disqualified = true;
            break;
          }
        }
      }
      if(disqualified) {
        continue;
      }

      // Destination of connection has no unused recurrent output connections
      // If the output recurrent connection is ourself, it is allowed.
      for(size_t j=0; j<num_connections; j++) {
        Connection& other = connections[j];
        if(!used[j] &&
           conn.dest == other.origin &&
           possible != j &&
           other.type == ConnectionType::Recurrent) {
          disqualified = true;
          break;
        }
      }
      if(disqualified) {
        continue;
      }

      // Hurray, we found one!
      break;
    }

    if(possible == num_connections) {
      throw std::runtime_error("Sorting failed. Replace this with a class specific Exception");
    }

    used[possible] = true;
    sorted.push_back(connections[possible]);
  }

  connections = sorted;
  connections_sorted = true;
}
