#pragma once
#include "NeuralNet.hh"

#include <vector>
#include <stdexcept>
#include <functional>
#include <cmath>
#include <memory>



template<typename T>
class NeuralNet_CRTP : public NeuralNet {
public:
  virtual ~NeuralNet_CRTP() { ; }
  virtual std::unique_ptr<NeuralNet> clone() const {
    return std::make_unique<T>(*static_cast<const T*>(this));
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
void NeuralNet_CRTP<T>::add_connection(int origin, int dest, _float_ weight) {
  if(would_make_loop(origin,dest)) {
    static_cast<T*>(this)->connections.emplace_back(origin,dest,ConnectionType::Recurrent,weight);
  } else {
    static_cast<T*>(this)->connections.emplace_back(origin,dest,ConnectionType::Normal,weight);
  }
}

template <typename T>
bool NeuralNet_CRTP<T>::would_make_loop(unsigned int i, unsigned int j) {
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
