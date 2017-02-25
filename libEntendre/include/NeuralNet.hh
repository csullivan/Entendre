#pragma once
#include <vector>
#include <stdexcept>
#include <functional>
#include <cmath>
#include <memory>
#include <algorithm>


typedef float _float_;


enum class NodeType { Input, Hidden, Output, Bias };

struct Node {
  _float_ value;
  bool is_sigmoid;
  NodeType type;
  Node(NodeType _type)
    : value(0.0),
      is_sigmoid(false), type(_type) {;}
  operator _float_() { return value; }
};

bool IsSensor(const NodeType& type);

enum class ConnectionType { Normal, Recurrent };
struct Connection {
  unsigned int origin;
  unsigned int dest;
  _float_ weight;
  ConnectionType type;
  unsigned int set = 0;
  Connection(unsigned int _origin,
             unsigned int _dest,
             ConnectionType _type,
             _float_ _weight=0.5)
    : origin(_origin), dest(_dest),
      weight(_weight), type(_type) {;}
};

class  NeuralNet {
public:
  virtual ~NeuralNet() { ; }


  virtual void add_node(const NodeType& type) = 0;
  virtual void add_connection(int origin, int dest, _float_ weight) = 0;
  virtual unsigned int num_nodes() = 0;
  virtual unsigned int num_connections() = 0;
  virtual Connection get_connection(unsigned int i) const = 0;
  virtual NodeType get_node_type(unsigned int i) const = 0;
  virtual std::vector<_float_> evaluate(std::vector<_float_> inputs) = 0;
  virtual std::unique_ptr<NeuralNet> clone() const = 0;

  virtual void print_network(std::ostream& os) const = 0;
  void register_sigmoid(std::function<_float_(_float_)> sig) {sigma = sig;}

protected:
  _float_ sigmoid(_float_ val) const;
  bool connections_sorted;
  std::function<_float_(_float_ val)> sigma;

private:
  virtual void sort_connections() = 0;

  friend std::ostream& operator<<(std::ostream& os, const NeuralNet& net);

};
