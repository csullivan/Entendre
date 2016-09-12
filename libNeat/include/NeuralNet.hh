#pragma once
#include <vector>
#include <stdexcept>
#include <functional>

// A network is the current real world view of an organisms
// internal network. It is what is used for performing calc-
// ulations and represents the state of all enabled links of
// a genome neuron set. Referred to as a pheonotype by NEAT.

struct Connection;
struct Node;

class NeuralNet {
public:
  NeuralNet(std::vector<Node>&& Nodes, std::vector<Connection>&& Conn);
  NeuralNet(const std::vector<Node>& Nodes);

  std::vector<double> evaluate(std::vector<double> inputs);
  void load_input_vals(const std::vector<double>& inputs);
  std::vector<double> read_output_vals();
  void add_connection(int origin, int dest, double weight);
  void register_sigmoid(std::function<double(double)> sig) {sigma = sig;}

private:

  bool would_make_loop(unsigned int i, unsigned int j);
  double sigmoid(double val) const;
  double get_node_val(unsigned int i);
  void add_to_val(unsigned int i, double val);
  void sort_connections();

  std::vector<Node> nodes;
  std::vector<Connection> connections;
  bool connections_sorted;
  std::function<double(double val)> sigma;

};

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


class NetworkException : public std::exception {
  using std::exception::exception;
};

class NetworkSensorSize : public NetworkException {
  using NetworkException::NetworkException;
};
