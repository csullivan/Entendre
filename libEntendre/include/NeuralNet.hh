#pragma once
#include <vector>
#include <stdexcept>
#include <functional>
#include <cmath>

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
  virtual NeuralNet* clone() const = 0;

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


template<typename T>
class NeuralNetRecursiveBase : public NeuralNet {
public:
  virtual ~NeuralNetRecursiveBase() { ; }
  virtual NeuralNet* clone() const {
    return new T(static_cast<T const&>(*this));
  }


  virtual void add_connection(int origin, int dest, _float_ weight);
  virtual unsigned int num_nodes() { return static_cast<T*>(this)->nodes.size(); }
  virtual unsigned int num_connections() { return static_cast<T*>(this)->connections.size(); }
  auto& get_connections() const { return static_cast<T*>(this)->connections; }

  virtual void print_network(std::ostream& os) const { std::string str = "Needs Impl."; os << str; }

protected:
  bool would_make_loop(unsigned int i, unsigned int j);

private:

};






template <typename T>
void NeuralNetRecursiveBase<T>::add_connection(int origin, int dest, _float_ weight) {
  if(would_make_loop(origin,dest)) {
    static_cast<T*>(this)->connections.emplace_back(origin,dest,ConnectionType::Recurrent,weight);
  } else {
    static_cast<T*>(this)->connections.emplace_back(origin,dest,ConnectionType::Normal,weight);
  }
}

template <typename T>
bool NeuralNetRecursiveBase<T>::would_make_loop(unsigned int i, unsigned int j) {
  // handle the case of a recurrent connection to itself up front
  if (i == j) { return true; }

  std::vector<bool> reachable(static_cast<T*>(this)->nodes.size(), false);
  reachable[j] = true;

  while (true) {

    bool found_new_node = false;
    for (auto const& conn : static_cast<T*>(this)->connections) {
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
