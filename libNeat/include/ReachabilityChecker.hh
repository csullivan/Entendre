#pragma once

#include <cstring>
#include <utility>
#include <vector>

class RNG;

class ReachabilityChecker {
public:
  /// Constructs the reachability checker
  /*
    num_nodes is the total number of nodes.
    num_inputs is the number of input nodes.

    Here, an input node is defined as any node that cannot be the
    destination of a connection.
    The input nodes are at indices [0, num_inputs).
   */
  ReachabilityChecker(size_t num_nodes, size_t num_inputs = 0);

  /// Adds a connection
  /*
    origin is the index of the origin-node
    destination is the index of the destination-node.

    Assumes that destination >= num_inputs.

    The connection added is a normal connection,
      unless adding such a connection would cause the normal connections to contain a loop.
    In that case, the connection added is recurrent.
   */
  void AddConnection(size_t origin, size_t destination);

  /// Checks whether a given connection has been defined
  bool HasConnection(size_t origin, size_t destination) const {
    return at(origin,destination).has_any_connection;
  }

  /// Checks whether a given connection has been defined, and is normal
  bool HasNormalConnection(size_t origin, size_t destination) const {
    return at(origin,destination).has_normal_connection;
  }

  /// Checks whether a given connection has been defined, and is recurrent
  bool HasRecurrentConnection(size_t origin, size_t destination) const {
    return at(origin,destination).has_recurrent_connection;
  }

  /// Return true if a path from origin to destination exists using only normal connections.
  bool IsReachableNormal(size_t origin, size_t destination) const {
    return at(origin,destination).reachable_normal;
  }

  /// Return true if a path from origin to destination exists using any connections.
  bool IsReachableEither(size_t origin, size_t destination) const {
    return at(origin,destination).reachable_either;
  }

  /// Returns true if no additional connections can be added.
  bool IsFullyConnected() const {
    return (num_possible_normal_connections == 0 &&
            num_possible_recurrent_connections == 0);
  }

  /// Returns the number of different normal connections that could be added.
  size_t NumPossibleNormalConnections() const {
    return num_possible_normal_connections;
  }

  /// Returns the number of different recurrent connections that could be added.
  size_t NumPossibleRecurrentConnections() const {
    return num_possible_recurrent_connections;
  }

  /// Returns a randomly selected normal connection to add.
  /**
     If no such connection exists, returns (-1,-1).
   */
  std::pair<int,int> RandomNormalConnection(RNG& rng);

  /// Returns a randomly selected recurrent connection to add.
  /**
     If no such connection exists, returns (-1,-1).
   */
  std::pair<int,int> RandomRecurrentConnection(RNG& rng);




private:
  // Bit-packed, sizeof(MatrixElement)==1
  struct MatrixElement {
    MatrixElement() {
      memset(this, 0, sizeof(MatrixElement));
    }

    bool has_any_connection : 1;
    bool has_normal_connection : 1;
    bool has_recurrent_connection : 1;
    bool reachable_normal : 1;
    bool reachable_either : 1;
  };

  size_t num_nodes;
  size_t num_inputs;
  size_t num_possible_normal_connections;
  size_t num_possible_recurrent_connections;

  std::vector<MatrixElement> mat;

  MatrixElement& at(size_t i, size_t j) {
    return mat[i*num_nodes + j];
  }
  const MatrixElement& at(size_t i, size_t j) const {
    return mat[i*num_nodes + j];
  }

  void fill_normal_reachable(size_t origin, size_t destination);
  void fill_either_reachable(size_t origin, size_t destination);

  bool could_add_normal(size_t origin, size_t destination) {
    return (!at(origin,destination).has_any_connection &&
            !at(destination, origin).reachable_normal);
  }

  bool could_add_recurrent(size_t origin, size_t destination) {
    return (!at(origin,destination).has_any_connection &&
            at(destination, origin).reachable_normal);
  }
};
