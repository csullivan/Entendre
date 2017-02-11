#pragma once
#include <vector>
#include <stdexcept>
#include <functional>
#include <cmath>


struct NodeGene;

enum class NodeType { Input, Hidden, Output, Bias };
struct Node {
  float value;
  bool is_sigmoid;
  NodeType type;
  Node(NodeType _type)
    : value(0.0),
      is_sigmoid(false), type(_type) {;}
  operator float() { return value; }
};

bool IsSensor(const NodeType& type);

enum class ConnectionType { Normal, Recurrent };
struct Connection {
  unsigned int origin;
  unsigned int dest;
  float weight;
  ConnectionType type;
  unsigned int set = 0;
  Connection(unsigned int _origin,
             unsigned int _dest,
             ConnectionType _type,
             float _weight=0.5)
    : origin(_origin), dest(_dest),
      weight(_weight), type(_type) {;}
};

// Abstract NeuralNet base class templated over node and connection types
// For subclass ConsecutiveNeuralNet, N=Node C=Connection
// For subclass ConcurrentNeuralNet, N=float C=Connection
template <typename N, typename C>
class  NeuralNet {
public:
  NeuralNet(std::vector<N>&& Nodes, std::vector<C>&& Conn);
  NeuralNet(const std::vector<N>& Nodes);

  void add_connection(int origin, int dest, float weight);
  void register_sigmoid(std::function<float(float)> sig) {sigma = sig;}

  unsigned int num_nodes() const { return nodes.size(); }
  unsigned int num_connections() const { return connections.size(); }

  const std::vector<C>& get_connections() const { return connections; }

protected:
  float sigmoid(float val) const;
  bool would_make_loop(unsigned int i, unsigned int j) const;


  std::vector<N> nodes;
  std::vector<C> connections;
  bool connections_sorted;
  std::function<float(float val)> sigma;

private:

  virtual void sort_connections() = 0;
  virtual std::vector<float> evaluate(std::vector<float> inputs) = 0;
};



template <typename N, typename C>
void NeuralNet<N,C>::add_connection(int origin, int dest, float weight) {
  if(would_make_loop(origin,dest)) {
    connections.emplace_back(origin,dest,ConnectionType::Recurrent,weight);
  } else {
    connections.emplace_back(origin,dest,ConnectionType::Normal,weight);
  }
}

template<typename N, typename C>
bool NeuralNet<N,C>::would_make_loop(unsigned int i, unsigned int j) const {
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

template<typename N, typename C>
float NeuralNet<N,C>::sigmoid(float val) const {
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
