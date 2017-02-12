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
  enum class EvaluationOrder { GreaterThan, LessThan, NotEqual, Unknown };
  EvaluationOrder compare_connections(const Connection& a, const Connection& b);


  std::vector<_float_> nodes_current;
  std::vector<_float_> nodes_prior;
};
