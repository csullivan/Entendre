#pragma once
#include "NeuralNet.hh"

#include <vector>
#include <stdexcept>
#include <functional>

#ifdef CUDA_ENABLED
class ConcurrentGPUNeuralNet : public NeuralNet {
public:

  virtual ~ConcurrentGPUNeuralNet();

  virtual void add_node(NodeType type, ActivationFunction func);
  virtual void add_connection(int origin, int dest, _float_ weight, unsigned int set=std::numeric_limits<unsigned int>::max());

  virtual unsigned int num_nodes() { return nodes.size(); }
  virtual unsigned int num_connections() { return connections.size(); }
  virtual std::vector<_float_> evaluate(std::vector<_float_> inputs);
  std::vector<_float_> device_evaluate(std::vector<_float_> inputs, unsigned int num_threads=16);

  virtual std::unique_ptr<NeuralNet> clone() const {
    return std::unique_ptr<ConcurrentGPUNeuralNet>(new ConcurrentGPUNeuralNet(*this));
  }
  virtual void print_network(std::ostream& os) const;


  virtual Connection get_connection(unsigned int i) const {
    return connections[i];
  }
  virtual NodeType get_node_type(unsigned int i) const {
    // TODO: Update this for CPPN node types.
    // NodeType::Hidden was changed temporarily to NodeType::Sigmoid
    return (i<num_inputs) ? NodeType::Input :
      (i >= nodes.size()-num_outputs) ? NodeType::Output : NodeType::Hidden;
  }
  virtual void sort_connections(unsigned int first=0, unsigned int num_connections=0);
  virtual std::vector<Connection>& get_connections() { return connections; }

  virtual ActivationFunction get_activation_func(unsigned int i) const {
    return ActivationFunction::Sigmoid;
  }

private:
  bool would_make_loop(unsigned int i, unsigned int j, unsigned int set=std::numeric_limits<unsigned int>::max());
  void build_action_list();
  void synchronize();


  enum class EvaluationOrder { GreaterThan, LessThan, Unknown };
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

#else
typedef ConcurrentNeuralNet ConcurrentGPUNeuralNet;
#endif
