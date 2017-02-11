#pragma once
#include <vector>
#include <stdexcept>
#include <functional>
#include "NeuralNet.hh"

class ConsecutiveNeuralNet;

class ConcurrentNeuralNet : public NeuralNet<float,Connection> {
public:
  using NeuralNet::NeuralNet;
  ConcurrentNeuralNet(ConsecutiveNeuralNet&& net);

  void sort_connections() override;
  std::vector<float> evaluate(std::vector<float> inputs);


private:
  enum class EvaluationOrder { GreaterThan, LessThan, NotEqual, Unknown };
  EvaluationOrder compare_connections(const Connection& a, const Connection& b);


  std::vector<float> nodes_current;
  std::vector<float> nodes_prior;
};
