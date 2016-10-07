#pragma once
#include <vector>
#include <cstring>

class ReachabilityChecker {
public:
  ReachabilityChecker(size_t num_nodes);

  void AddConnection(size_t origin, size_t destination);

  bool HasConnection(size_t origin, size_t destination) const {
    return at(origin,destination).has_any_connection;
  }
  bool HasNormalConnection(size_t origin, size_t destination) const {
    return at(origin,destination).has_normal_connection;
  }
  bool HasRecurrentConnection(size_t origin, size_t destination) const {
    return at(origin,destination).has_recurrent_connection;
  }

  bool IsReachableNormal(size_t origin, size_t destination) const {
    return at(origin,destination).reachable_normal;
  }
  bool IsReachableEither(size_t origin, size_t destination) const {
    return at(origin,destination).reachable_either;
  }

  // bool IsFullyConnected() const {
  // }
  // std::pair<size_t,size_t> AddRandomRecurrent() {
  // }
  // std::pair<size_t,size_t> AddRandomNormal() {
  // }



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
  std::vector<MatrixElement> mat;

  MatrixElement& at(size_t i, size_t j) {
    return mat[i*num_nodes + j];
  }
  const MatrixElement& at(size_t i, size_t j) const {
    return mat[i*num_nodes + j];
  }

  void fill_normal_reachable(size_t origin, size_t destination);
  void fill_either_reachable(size_t origin, size_t destination);
};
