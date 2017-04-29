#pragma once
#include "NeuralNet.hh"

#include <vector>
#include <stdexcept>
#include <functional>
#include <cmath>
#include <cassert>
#include <memory>
#include <map>



template<typename T>
class NeuralNet_CRTP : public NeuralNet {
public:
  virtual ~NeuralNet_CRTP() { ; }
  virtual std::unique_ptr<NeuralNet> clone() const {
    return std::make_unique<T>(*static_cast<const T*>(this));
  }


  virtual void add_connection(int origin, int dest, _float_ weight, unsigned int set=std::numeric_limits<unsigned int>::max());
  virtual unsigned int num_nodes() { return static_cast<T*>(this)->nodes.size(); }
  virtual unsigned int num_connections() { return static_cast<T*>(this)->connections.size(); }
  auto& get_connections() { return static_cast<T*>(this)->connections; }

  virtual void print_network(std::ostream& os) const { std::string str = "Needs Impl."; os << str; }

protected:
  bool would_make_loop(unsigned int i, unsigned int j, unsigned int set=std::numeric_limits<unsigned int>::max());

private:

};



#include <iostream>

template <typename T>
void NeuralNet_CRTP<T>::add_connection(int origin, int dest, _float_ weight, unsigned int set) {
  if(would_make_loop(origin,dest,set)) {
    static_cast<T*>(this)->connections.emplace_back(origin,dest,ConnectionType::Recurrent,weight,set);
  } else {
    static_cast<T*>(this)->connections.emplace_back(origin,dest,ConnectionType::Normal,weight,set);
  }
}

// TODO: This function needs some refactoring. Currently there is code repetition due to the fact that the
//       map required for the subnet sorting (which is only needed for composite nets) is very slow when
//       applied to an entire population. Thus, for now there are two different functionalities depending
//       on whether or not a finite set value is used
template <typename T>
bool NeuralNet_CRTP<T>::would_make_loop(unsigned int i, unsigned int j, unsigned int set) {
  // handle the case of a recurrent connection to itself up front
  if (i == j) { return true; }

  if (set == std::numeric_limits<unsigned int>::max()) {

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

  } else {
    // if set number is not zero, then it is assumed the added connection is
    // part of a subnet that is currently being added.

    std::map<unsigned int,unsigned int> subset_node_map;
    subset_node_map[i] = subset_node_map.size();
    subset_node_map[j] = subset_node_map.size();

    auto conn_iter = static_cast<T*>(this)->connections.end();
    while (conn_iter-- != static_cast<T*>(this)->connections.begin()) {
      auto conn_set = (*conn_iter).set;
      if (conn_set != set){
        break;
      } else {
        auto origin = (*conn_iter).origin;
        auto dest = (*conn_iter).dest;

        if (subset_node_map.count(origin)==0) {
          subset_node_map[origin] = subset_node_map.size();
        }
        if (subset_node_map.count(dest)==0) {
          subset_node_map[dest] = subset_node_map.size();
        }
      }

    }


    std::vector<bool> reachable(subset_node_map.size(), false);
    reachable[subset_node_map[j]] = true;
    while (true) {
      auto conn_start = conn_iter;

      bool found_new_node = false;
      while (++conn_start != static_cast<T*>(this)->connections.end()) {
        auto const& conn = *conn_start;
        assert(conn.set == set);

        // if the origin of this connection is reachable and its
        // desitination is not, then it should be made reachable
        if (reachable[subset_node_map[conn.origin]] &&
            !reachable[subset_node_map[conn.dest]] &&
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
            reachable[subset_node_map[conn.dest]] = true;
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
}
