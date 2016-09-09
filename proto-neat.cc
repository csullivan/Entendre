#include <cmath>

enum class NodeType { Input, Hidden, Output, Bias };

struct Node {
  unsigned int id;
  double value;
  bool is_sigmoid;
  NodeType type;
  Node(unsigned int _id, NodeType _type)
    : id(id), value(0.0),
      is_sigmoid(false), type(_type) {;}
};

enum class ConnectionType { Normal, Recurrent };

struct Connection {
  unsigned int to;
  unsigned int from;
  double weight;
  ConnectionType type;
  Connection(unsigned int _to,
             unsigned int _from,
             ConnectionType _type,
             double _weight=0.5)
    : to(_to), from(_from),
      weight(_weight), type(_type) {;}

};

class NeuralNet {
public:
  double evaluate(std::vector<double> inputs) {
    sort_connections();
    load_input_vals(inputs);

    for(auto& conn : connections) {
      double input_val = get_node_val(conn.from);
      add_to_node(conn.to, input_val * conn.weight);
    }

    return read_output_vals();
  }

  void load_input_vals(const std::vector<double>& inputs) {
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

  std::vector<double> read_output_vals() {
    std::vector<double> output;
    for(size_t i=0; i<nodes.size(); i++) {
      if(nodes[i].type == NodeType::Output) {
        output.push_back(get_node_val(i));
      }
    }
    return output;
  }

private:
  void add_connection(int i, int j) {
    if(would_make_loop(i,j)) {
      add_recurrent_connection(i,j);
    } else {
      add_normal_connection(i,j);
    }
  }

  double sigmoid(double val) const {
    // Logistic curve
    return 1/(1 + std::exp(-val));

    // Other options for a sigmoid curve
    //return val/std::sqrt(1+val*val);
    //return std::tanh(val);
    //return std::erf((std::sqrt(M_PI)/2)*val);
    //return (2/M_PI) * std::atan((M_PI/2)*val);
    //return val/(1+std::abs(val));
  }

  double get_node_val(unsigned int index) {
    if(!nodes[i].is_sigmoid) {
      nodes[i].value = sigmoid(nodes[i].value);
      nodes[i].is_sigmoid = true;
    }
    return nodes[i].value;
  }

  void add_to_val(unsigned int index, double val) {
    if(nodes[i].is_sigmoid) {
      nodes[i].value = 0;
      nodes[i].is_sigmoid = false;
    }
    nodes[i].value += val;
  }

  void sort_connections() {
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
               conn.from == other.to) {
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
             conn.to == other.from &&
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
        throw SomethingBadHere();
      }

      used[possible] = true;
      sorted.push_back(connections[possible]);
    }

    connections = sorted;
    connections_sorted = true;
  }

  std::vector<Node> nodes;
  std::vector<Connection> connections;
  bool connections_sorted;
};

struct Gene {
  bool enabled;
  long innovation_number;
  Connection link;
}

class Genome {
public:
  void BuildNetwork() {
    // add nodes to network
    for (auto const& node : nodes) {
      network.nodes.push_back(node);
    }
    // add network connections where appropriate
    for (auto const& gene : genes) {
      if (gene.enabled) {
        network.connections.push_back(gene.link);
      }
    }
  }
  auto AddNode(NodeType type) {
    nodes.emplace_back(nodes.back().id+1,type);
    return *this;
  }
  auto AddGene(unsigned int to, unsigned int from, ConnectionType type,
               bool status, long innovation_num, double weight = 0.5) {
    Gene g = {status,innovation_num,Connection(to,from,type,weight)};
    genes.push_back(g);
    return *this;
  }

private:
  std::vector<Node> nodes;
  std::vector<Gene> genes;
  NeuralNet network;
};
