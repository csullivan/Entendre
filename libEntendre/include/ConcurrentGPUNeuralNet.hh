#pragma once
#include "NeuralNet.hh"

#include <vector>
#include <stdexcept>
#include <functional>

class ConcurrentGPUNeuralNet : public NeuralNet {
public:

  virtual ~ConcurrentGPUNeuralNet();

  virtual void add_node(const NodeType& type);
  virtual void add_connection(int origin, int dest, _float_ weight);
  virtual unsigned int num_nodes() { return nodes.size(); }
  virtual unsigned int num_connections() { return connections.size(); }
  virtual std::vector<_float_> evaluate(std::vector<_float_> inputs);
  std::vector<_float_> device_evaluate(std::vector<_float_> inputs, unsigned int num_threads=16);

  virtual std::unique_ptr<NeuralNet> clone() const {
    return std::unique_ptr<ConcurrentGPUNeuralNet>(new ConcurrentGPUNeuralNet(*this));
  }
  virtual void print_network(std::ostream& os) const { std::string str = "Needs Impl."; os << str; }


  virtual Connection get_connection(unsigned int i) const {
    return connections[i];
  }
  virtual NodeType get_node_type(unsigned int i) const {
    return (i<num_inputs) ? NodeType::Input :
      (i >= nodes.size()-num_outputs) ? NodeType::Output : NodeType::Hidden;
  }

private:
  bool would_make_loop(unsigned int i, unsigned int j);
  virtual void sort_connections();
  void build_action_list();
  void synchronize();


  enum class EvaluationOrder { GreaterThan, LessThan, NotEqual, Unknown };
  EvaluationOrder compare_connections(const Connection& a, const Connection& b);

  size_t num_inputs = 0;
  size_t num_outputs = 0;


  struct Connections {
    Connections() { ; }
    std::vector<unsigned int> origin;
    std::vector<unsigned int> dest;
    std::vector<_float_> weight;
    Connections(size_t size) { origin.reserve(size); dest.reserve(size); weight.reserve(size); }
    inline unsigned int size() const { return origin.size(); }
    inline void add(int origin, int dest, _float_ weight) { this->origin.push_back(origin); this->dest.push_back(dest); this->weight.push_back(weight); }
  };

  std::vector<_float_> nodes;
  std::vector<Connection> connections;
  Connections connection_list;
  std::vector<unsigned int> action_list;

  // device pointers
  _float_* node_ = nullptr;
  unsigned int* origin_ = nullptr;
  unsigned int* dest_ = nullptr;
  _float_* weight_ = nullptr;
  unsigned int* action_list_ = nullptr;
};

template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(const std::vector<T>& vec, Compare& compare) {
  std::vector<std::size_t> p(vec.size());
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(),
      [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
  return p;
}

template <typename T>
void apply_permutation_in_place(std::vector<T>& vec, const std::vector<std::size_t>& p) {
  std::vector<bool> done(vec.size());
  for (std::size_t i = 0; i < vec.size(); ++i)
  {
    if (done[i])
    {
      continue;
    }
    done[i] = true;
    std::size_t prev_j = i;
    std::size_t j = p[i];
    while (i != j)
    {
      std::swap(vec[prev_j], vec[j]);
      done[j] = true;
      prev_j = j;
      j = p[j];
    }
  }
}
