#include "NeuralNet.hh"

#include <cmath>
#include <iostream>
#include <vector>

NeuralNet::NeuralNet(std::vector<Node>&& Nodes, std::vector<Connection>&& Conn)
  : nodes(std::move(Nodes)), connections(std::move(Conn)) { ; }

NeuralNet::NeuralNet(const std::vector<Node>& Nodes) {
  nodes = Nodes;
}
std::vector<double> NeuralNet::evaluate(std::vector<double> inputs) {
  sort_connections();
  load_input_vals(inputs);

  for(auto& conn : connections) {
    double input_val = get_node_val(conn.origin);
    add_to_node(conn.dest, input_val * conn.weight);
  }

  return read_output_vals();
}

void NeuralNet::load_input_vals(const std::vector<double>& inputs) {
  size_t input_index = 0;

  for(auto& node : nodes) {
    switch(node.type) {
    case NodeType::Input:
      if(input_index < inputs.size()) {
        node.value = inputs[input_index];
      } else {
        node.value = 0;
      }
      node.is_sigmoid = true;
      input_index++;
      break;

    case NodeType::Bias:
      node.value = 1;
      node.is_sigmoid = true;
      break;

    default:
      break;
    }
  }
}

std::vector<double> NeuralNet::read_output_vals() {
  std::vector<double> output;
  for(size_t i=0; i<nodes.size(); i++) {
    if(nodes[i].type == NodeType::Output) {
      output.push_back(get_node_val(i));
    }
  }
  return output;
}

void NeuralNet::add_connection(int origin, int dest, double weight) {
  if(would_make_loop(dest,origin)) {
    connections.emplace_back(origin,dest,ConnectionType::Recurrent,weight);
  } else {
    connections.emplace_back(origin,dest,ConnectionType::Normal,weight);
  }
}

void NeuralNet::add_to_node(unsigned int dest, double val) {
  nodes[dest].value += val;
}

bool NeuralNet::would_make_loop(unsigned int i, unsigned int j) {

  std::vector<bool> reachable(nodes.size(), false);
  reachable[j] = true;

  while (true) {

    bool found_new_node = false;
    for (auto const& conn : connections) {
      // if the origin of this connection is reachable and its
      // desitination is not, then it should be made reachable
      // if it is a normal node. if it is the origin of the
      // proposed additional connection (i->j) then it would be
      // a loop
      if (reachable[conn.origin] &&
          !reachable[conn.dest] &&
          conn.type == ConnectionType::Normal) {
        if (conn.dest == i) {
          // the destination of this reachable connection is
          // the origin of the proposed connection. thus there
          // exists a path from j -> i. So this will be a loop.
          return true;
        }
        else {
          reachable[conn.dest] = true;
          found_new_node = true;
        }
      }
    }
    // no loop detected
    if (!found_new_node) {
      return false;
    }

  }
}

double NeuralNet::sigmoid(double val) const {
  return (sigma) ? sigma(val) :
  // Logistic curve
    1/(1 + std::exp(-val));
  // Other options for a sigmoid curve
  //val/std::sqrt(1+val*val);
  //std::tanh(val);
  //std::erf((std::sqrt(M_PI)/2)*val);
  //(2/M_PI) * std::atan((M_PI/2)*val);
  //val/(1+std::abs(val));
}

double NeuralNet::get_node_val(unsigned int i) {
  if(!nodes[i].is_sigmoid) {
    nodes[i].value = sigmoid(nodes[i].value);
    nodes[i].is_sigmoid = true;
  }
  return nodes[i].value;
}

void NeuralNet::add_to_val(unsigned int i, double val) {
  if(nodes[i].is_sigmoid) {
    nodes[i].value = 0;
    nodes[i].is_sigmoid = false;
  }
  nodes[i].value += val;
}

void NeuralNet::sort_connections() {
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
      // ERRRROOR!
      //throw SomethingBadHere();
      throw std::runtime_error("Sorting failed. Replace this with a class specific Exception");
    }

    used[possible] = true;
    sorted.push_back(connections[possible]);
  }

  connections = sorted;
  connections_sorted = true;
}
