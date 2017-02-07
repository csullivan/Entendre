#pragma once
#include <vector>
#include <stdexcept>
#include <functional>
#include "NeuralNetBase.hh"

// A network is the current real world view of an organisms
// internal network. It is what is used for performing calc-
// ulations and represents the state of all enabled links of
// a genome neuron set. Referred to as a pheonotype by NEAT.

class NeuralNet : public NeuralNetBase {
  friend class LFNetwork;
public:
  using NeuralNetBase::NeuralNetBase;

  std::vector<double> evaluate(std::vector<double> inputs);

private:
  void sort_connections() override;

};


class LFNetwork : public NeuralNetBase {
public:
  using NeuralNetBase::NeuralNetBase;
  LFNetwork(NeuralNet&& net);

  void sort_connections() override;


private:
  enum class EvaluationOrder { GreaterThan, LessThan, NotEqual, Unknown };
  EvaluationOrder compare_connections(const Connection& a, const Connection& b);
};
