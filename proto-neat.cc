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
  unsigned int origin;
  unsigned int dest;
  double weight;
  ConnectionType type;
  Connection(unsigned int _origin,
             unsigned int _dest,
             ConnectionType _type,
             double _weight=0.5)
    : origin(_origin), dest(_dest),
      weight(_weight), type(_type) {;}

};

class NeuralNet {
public:
  NeuralNet(std::vector<Node>&& Nodes, std::vector<Connection>&& Conn)
    : nodes(std::move(Nodes)), connections(std::move(Conn)) { ; }

  NeuralNet(const std::vector<Node>& Nodes) {
    nodes = Nodes;
  }
  std::vector<double> evaluate(std::vector<double> inputs) {
    sort_connections();
    load_input_vals(inputs);

    for(auto& conn : connections) {
      double input_val = get_node_val(conn.origin);
      add_to_node(conn.dest, input_val * conn.weight);
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

  void add_connection(int origin, int dest, double weight) {
    if(would_make_loop(dest,origin)) {
      connections.emplace_back(origin,dest,ConnectionType::Recurrent,weight);
    } else {
      connections.emplace_back(origin,dest,ConnectionType::Normal,weight);
    }
  }

private:

  void add_to_node(unsigned int dest, double val) {
    nodes[dest].value += val;
  }

  bool would_make_loop(unsigned int i, unsigned int j) {

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

unsigned long Hash(unsigned long origin, unsigned long dest, unsigned long previous_hash) {
  return ((origin*746151647) xor (dest*15141163) xor (previous_hash*94008721)); // % 10000000000u;
}
struct Gene {
  bool enabled;
  unsigned long innovation_number;
  Connection link;
};

class Genome : public uses_random_numbers {
public:
  operator NeuralNet() const {
    NeuralNet net(nodes);
    // add network connections where appropriate
    // std::vector<Connection> net_conn;
    for (auto const& gene : genes) {
      if (gene.enabled) {
        net.add_connection(gene.link.origin,gene.link.dest,gene.link.weight);
      }
    }

    return net;
  }

  auto AddNode(NodeType type) {
    nodes.emplace_back(type);
    return *this;
  }
  auto AddGene(unsigned int origin, unsigned int dest, ConnectionType type,
               bool status, double weight) {

    unsigned long innovation = 0;
    if (genes.size()) {
      innovation = Hash(dest,origin,genes.back().innovation_number);
    }
    else {
      innovation = Hash(dest,origin,innovation);
    }


    genes.push_back({status,innovation,Connection(origin,dest,type,weight)});
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
