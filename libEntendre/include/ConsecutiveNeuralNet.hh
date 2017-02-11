#pragma once
#include <vector>
#include <stdexcept>
#include <functional>
#include "NeuralNet.hh"


class ConsecutiveNeuralNet : public NeuralNet<Node,Connection> {
  friend class ConcurrentNeuralNet;
public:
  using NeuralNet::NeuralNet;

  void load_input_vals(const std::vector<double>& inputs);
  std::vector<double> read_output_vals();
  std::vector<double> evaluate(std::vector<double> inputs) override;

  std::vector<NodeType> node_types() const;
protected:
  double get_node_val(unsigned int i);
  void add_to_val(unsigned int i, double val);


private:
  void sort_connections() override;
  friend std::ostream& operator<<(std::ostream& os, const  ConsecutiveNeuralNet& net);

};
