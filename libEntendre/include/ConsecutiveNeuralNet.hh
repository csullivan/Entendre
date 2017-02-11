#pragma once
#include <vector>
#include <stdexcept>
#include <functional>
#include "NeuralNet.hh"


class ConsecutiveNeuralNet : public NeuralNet {
  friend class ConcurrentNeuralNet;
public:
  using NeuralNet::NeuralNet;

  std::vector<double> evaluate(std::vector<double> inputs);

private:
  void sort_connections() override;
};
