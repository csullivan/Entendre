#include "NeuralNet.hh"

#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

#include "Genome.hh"

NeuralNet::NeuralNet(std::vector<Node>&& Nodes, std::vector<Connection>&& Conn)
  : nodes(std::move(Nodes)), connections(std::move(Conn)), connections_sorted(false) { ; }

NeuralNet::NeuralNet(const std::vector<NodeGene>& genes) : connections_sorted(false) {
  for (auto const& gene : genes) {
    nodes.emplace_back(gene.type);
  }
}

std::vector<double> NeuralNet::evaluate(std::vector<double> inputs) {
  sort_connections();
  load_input_vals(inputs);

  for(auto& conn : connections) {
    double input_val = get_node_val(conn.origin);
    add_to_val(conn.dest, input_val * conn.weight);
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
  if(would_make_loop(origin,dest)) {
    connections.emplace_back(origin,dest,ConnectionType::Recurrent,weight);
  } else {
    connections.emplace_back(origin,dest,ConnectionType::Normal,weight);
  }
}

bool NeuralNet::would_make_loop(unsigned int i, unsigned int j) const {
  // handle the case of a recurrent connection to itself up front
  if (i == j) { return true; }

  std::vector<bool> reachable(nodes.size(), false);
  reachable[j] = true;

  while (true) {

    bool found_new_node = false;
    for (auto const& conn : connections) {
      // if the origin of this connection is reachable and its
      // desitination is not, then it should be made reachable
      if (reachable[conn.origin] &&
          !reachable[conn.dest] &&
          conn.type == ConnectionType::Normal) {
        // if it is a normal node. if it is the origin of the
        // proposed additional connection (i->j) then it would be
        // a loop
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

bool IsSensor(const NodeType& type) {
  return type == NodeType::Input || type == NodeType::Bias;
}

std::ostream& operator<<(std::ostream& os, const NeuralNet& net) {
  std::map<unsigned int, std::string> names;
  unsigned int num_inputs = 0;
  unsigned int num_outputs = 0;
  unsigned int num_hidden = 0;
  for(unsigned int i=0; i<net.nodes.size(); i++) {
    std::stringstream ss;
    switch(net.nodes[i].type) {
      case NodeType::Input:
        ss << "I" << num_inputs++;
        break;
      case NodeType::Output:
        ss << "O" << num_outputs++;
        break;
      case NodeType::Hidden:
        ss << "H" << num_hidden++;
        break;
      case NodeType::Bias:
        ss << "B";
        break;

      default:
        std::cerr << "Type: " << int(net.nodes[i].type) << std::endl;
        assert(false);
        break;
    }
    names[i] = ss.str();
  }

  for(const auto& item : names) {
    std::cout << "Node " << item.first << " = " << item.second << std::endl;
  }

  for(auto& conn : net.connections) {
    os << names[conn.origin];
    //os << conn.origin;
    if(conn.type == ConnectionType::Normal) {
      os << " ---> ";
    } else {
      os << " -R-> ";
    }
    os << names[conn.dest];
    //os << conn.dest;

    if(&conn != &net.connections.back()) {
      os << "\n";
    }
  }
  return os;
}

std::vector<NodeType> NeuralNet::node_types() const {
  std::vector<NodeType> output;

  for(auto& node : nodes) {
    output.push_back(node.type);
  }

  return output;
}
