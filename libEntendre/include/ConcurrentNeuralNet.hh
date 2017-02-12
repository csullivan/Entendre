#pragma once
#include <vector>
#include <stdexcept>
#include <functional>
#include "NeuralNet.hh"

class ConsecutiveNeuralNet;

class ConcurrentNeuralNet : public NeuralNet<_float_,Connection> {
public:
  using NeuralNet::NeuralNet;
  ConcurrentNeuralNet(ConsecutiveNeuralNet&& net);

  void sort_connections() override;
  std::vector<_float_> evaluate(std::vector<_float_> inputs);


private:
  void clear_nodes(unsigned int* list, unsigned int n);
  void sigmoid_nodes(unsigned int* list, unsigned int n);
  void apply_connections(Connection* list, unsigned int n);
  void build_action_list();


  enum class EvaluationOrder { GreaterThan, LessThan, NotEqual, Unknown };
  EvaluationOrder compare_connections(const Connection& a, const Connection& b);

  size_t num_inputs = 0;
  size_t num_outputs = 0;
  std::vector<unsigned int> action_list;
};
