#pragma once
#include <vector>
#include <stdexcept>
#include <functional>
#include "NeuralNet.hh"

class ConsecutiveNeuralNet;

class ConcurrentNeuralNet : public NeuralNet<double,Connection> {
public:
  using NeuralNet::NeuralNet;
  ConcurrentNeuralNet(ConsecutiveNeuralNet&& net);

  void sort_connections() override;
  std::vector<double> evaluate(std::vector<double> inputs);


private:
  enum class EvaluationOrder { GreaterThan, LessThan, NotEqual, Unknown };
  EvaluationOrder compare_connections(const Connection& a, const Connection& b);


  std::vector<double> nodes_current;
  std::vector<double> nodes_prior;
};
