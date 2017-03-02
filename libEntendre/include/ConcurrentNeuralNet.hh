#pragma once
#include "NeuralNet_CRTP.hh"

#include <vector>
#include <stdexcept>
#include <functional>

class ConcurrentNeuralNet : public NeuralNet_CRTP<ConcurrentNeuralNet> {
  friend class NeuralNet_CRTP;
public:
  //using NeuralNet::NeuralNet;
  virtual ~ConcurrentNeuralNet() { ; }

  void sort_connections() override;
  std::vector<_float_> evaluate(std::vector<_float_> inputs);
  virtual void add_node(const NodeType& type);


  virtual Connection get_connection(unsigned int i) const {
    return connections[i];
  }
  virtual NodeType get_node_type(unsigned int i) const {
    return (i<num_inputs) ? NodeType::Input :
      (i >= nodes.size()-num_outputs) ? NodeType::Output : NodeType::Hidden;
  }


private:
  void clear_nodes(unsigned int* list, unsigned int n);
  void sigmoid_nodes(unsigned int* list, unsigned int n);
  void apply_connections(Connection* list, unsigned int n);
  void build_action_list();


  enum class EvaluationOrder { GreaterThan, LessThan, NotEqual, Unknown };
  EvaluationOrder compare_connections(const Connection& a, const Connection& b);

  size_t num_inputs = 0;
  size_t num_outputs = 0;

  std::vector<_float_> nodes;
  std::vector<Connection> connections;
  std::vector<unsigned int> action_list;
};
