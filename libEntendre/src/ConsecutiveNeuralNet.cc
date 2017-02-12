#include "ConsecutiveNeuralNet.hh"

#include <cassert>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>


std::vector<_float_> ConsecutiveNeuralNet::evaluate(std::vector<_float_> inputs) {
  sort_connections();
  load_input_vals(inputs);

  for(auto& conn : connections) {
    _float_ input_val = get_node_val(conn.origin);
    add_to_val(conn.dest, input_val * conn.weight);
  }

  return read_output_vals();
}

void ConsecutiveNeuralNet::load_input_vals(const std::vector<_float_>& inputs) {
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

std::vector<_float_> ConsecutiveNeuralNet::read_output_vals() {
  std::vector<_float_> output;
  for(size_t i=0; i<nodes.size(); i++) {
    if(nodes[i].type == NodeType::Output) {
      output.push_back(get_node_val(i));
    }
  }
  return output;
}

_float_ ConsecutiveNeuralNet::get_node_val(unsigned int i) {
  if(!nodes[i].is_sigmoid) {
    nodes[i].value = sigmoid(nodes[i].value);
    nodes[i].is_sigmoid = true;
  }
  return nodes[i].value;
}

void ConsecutiveNeuralNet::add_to_val(unsigned int i, _float_ val) {
  if(nodes[i].is_sigmoid) {
    nodes[i].value = 0;
    nodes[i].is_sigmoid = false;
  }
  nodes[i].value += val;
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

std::vector<NodeType> ConsecutiveNeuralNet::node_types() const {
  std::vector<NodeType> output;

  for(auto& node : nodes) {
    output.push_back(node.type);
  }

  return output;
}

std::ostream& operator<<(std::ostream& os, const ConsecutiveNeuralNet& net) {
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
