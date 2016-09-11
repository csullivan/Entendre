#include <cmath>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <limits.h>
#include <random>
#include <chrono>
#include <memory>

namespace Constants {
  const float PerturbWeight = 0.9;
  const float StepSize = 0.1;
  const float ResetWeightScale = 4.0;
}

enum class NodeType { Input, Hidden, Output, Bias };

struct Node {
  double value;
  bool is_sigmoid;
  NodeType type;
  Node(NodeType _type)
    : value(0.0),
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
  NeuralNet(std::vector<Node>& Nodes, std::vector<Connection>& Conn) {
    nodes = std::move(Nodes);
    connections = std::move(Conn);
  }
  std::vector<double> evaluate(std::vector<double> inputs) {
    sort_connections();
    load_input_vals(inputs);

    for(auto& conn : connections) {
      double input_val = get_node_val(conn.from);
      add_to_node(conn.to, input_val * conn.weight);
    }

    return read_output_vals();
  }

  void add_to_node(unsigned int to, double val) {
    nodes[to].value += val;
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

  // void add_connection(int i, int j) {
  //   if(would_make_loop(i,j)) {
  //     add_recurrent_connection(i,j);
  //   } else {
  //     add_normal_connection(i,j);
  //   }
  // }

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

  double get_node_val(unsigned int i) {
    if(!nodes[i].is_sigmoid) {
      nodes[i].value = sigmoid(nodes[i].value);
      nodes[i].is_sigmoid = true;
    }
    return nodes[i].value;
  }

  void add_to_val(unsigned int i, double val) {
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
        //throw SomethingBadHere();
        throw std::runtime_error("Sorting failed. Replace this with a class specific Exception");
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

class RNG {
public:
  RNG() {
    std::random_device rd;
    if (rd.entropy() != 0) {
      mt = std::make_unique<std::mt19937>(rd());
    } else {
      mt = std::make_unique<std::mt19937>
        (std::chrono::system_clock::now().time_since_epoch().count());
    }
  }
  RNG(unsigned long seed) : mt(std::make_unique<std::mt19937>(seed)) {;}
  virtual ~RNG() { ; }
  virtual double operator()() = 0;

protected:
  std::unique_ptr<std::mt19937> mt;
};

class Gaussian : public RNG {
public:
  Gaussian(double mean, double sigma): RNG(), dist(mean,sigma) {;}
  double operator()() override { return dist(*mt); }
private:
  std::normal_distribution<double> dist;
};

class Uniform : public RNG {
public:
  Uniform(double min, double max): RNG(), dist(min,max) {;}
  double operator()() override { return dist(*mt); }
private:
  std::uniform_real_distribution<double> dist;
};

class uses_random_numbers {
public:
  auto get_generator() const { return generator; }
  void set_generator(const std::shared_ptr<RNG>& gen) { generator = gen; }
protected:
  double random() { return (*generator)(); }
  std::shared_ptr<RNG> generator;
};

unsigned long Hash(unsigned long to, unsigned long from, unsigned long previous_hash) {
  return ((to*746151647) xor (from*15141163) xor (previous_hash*94008721)); // % 10000000000u;
}
struct Gene {
  bool enabled;
  unsigned long innovation_number;
  Connection link;
};

class Genome : public uses_random_numbers {
public:
  operator NeuralNet() const {
    std::vector<Node> net_nodes;
    std::vector<Connection> net_conn;
    // add nodes to network
    for (auto const& node : nodes) {
      net_nodes.push_back(node);
    }
    // add network connections where appropriate
    for (auto const& gene : genes) {
      if (gene.enabled) {
        net_conn.push_back(gene.link);
      }
    }

    return NeuralNet(net_nodes,net_conn);
  }

  auto AddNode(NodeType type) {
    nodes.emplace_back(type);
    return *this;
  }
  auto AddGene(unsigned int to, unsigned int from, ConnectionType type,
               bool status, double weight) {

    unsigned long innovation = 0;
    if (genes.size()) {
      innovation = Hash(to,from,genes.back().innovation_number);
    }
    else {
      innovation = Hash(to,from,innovation);
    }


    genes.push_back({status,innovation,Connection(to,from,type,weight)});
    return *this;
  }

  void WeightMutate() {
    if (!generator) { throw std::runtime_error("No RNG engine set. Replace this with class specific exception."); }
    for (auto& gene : genes) {
      // perturb weight by a small amount
      if(random()<Constants::PerturbWeight) {
        gene.link.weight += Constants::StepSize*(2*random()-1);
      } else { // otherwise randomly set weight within reset range
        gene.link.weight = (random() - 0.5)*Constants::ResetWeightScale;
      }
    }
  }

  void LinkMutate() {

  }

  void NodeMutate() {

  }

  void Mutate() {

  }

  void PrintInnovations() {
    for (auto const& gene : genes) {
      std::cout << gene.innovation_number << std::endl;
    }
  }

private:
  std::vector<Node> nodes;
  std::vector<Gene> genes;
};


int main() {

  auto gnome2 = Genome()
    .AddNode(NodeType::Bias)
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Output)
    .AddGene(0,3,ConnectionType::Normal,true,0.5)
    .AddGene(1,3,ConnectionType::Normal,true,0.5)
    .AddGene(2,3,ConnectionType::Normal,true,0.5)
    .AddGene(2,3,ConnectionType::Normal,true,0.5)
    .AddNode(NodeType::Hidden)
    .AddGene(0,4,ConnectionType::Normal,true,0.5)
    .AddGene(4,3,ConnectionType::Normal,true,0.5);


  gnome2.set_generator(std::make_shared<Uniform>(0.,1.));
  gnome2.WeightMutate();
  //gnome2.PrintInnovations();

  return 0;
}
