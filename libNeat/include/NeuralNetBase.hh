#pragma once
#include <vector>
#include <stdexcept>
#include <functional>

// A network is the current real world view of an organisms
// internal network. It is what is used for performing calc-
// ulations and represents the state of all enabled links of
// a genome neuron set. Referred to as a pheonotype by NEAT.

struct NodeGene;

enum class NodeType { Input, Hidden, Output, Bias };
bool IsSensor(const NodeType& type);

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
  unsigned int set = 0;
  Connection(unsigned int _origin,
             unsigned int _dest,
             ConnectionType _type,
             double _weight=0.5)
    : origin(_origin), dest(_dest),
      weight(_weight), type(_type) {;}
};

class  NeuralNetBase {
public:
   NeuralNetBase(std::vector<Node>&& Nodes, std::vector<Connection>&& Conn);
   NeuralNetBase(const std::vector<NodeGene>& Nodes);

  void load_input_vals(const std::vector<double>& inputs);
  std::vector<double> read_output_vals();
  void add_connection(int origin, int dest, double weight);
  void register_sigmoid(std::function<double(double)> sig) {sigma = sig;}

  unsigned int num_nodes() const { return nodes.size(); }
  unsigned int num_connections() const { return connections.size(); }

  std::vector<NodeType> node_types() const;
  const std::vector<Connection>& get_connections() const { return connections; }

protected:
  double sigmoid(double val) const;
  double get_node_val(unsigned int i);
  void add_to_val(unsigned int i, double val);
  bool would_make_loop(unsigned int i, unsigned int j) const;


  std::vector<Node> nodes;
  std::vector<Connection> connections;
  bool connections_sorted;
  std::function<double(double val)> sigma;

private:

  virtual void sort_connections() = 0;

  friend std::ostream& operator<<(std::ostream& os, const  NeuralNetBase& net);
};


class NetworkException : public std::exception {
  using std::exception::exception;
};

class NetworkSensorSize : public NetworkException {
  using NetworkException::NetworkException;
};
