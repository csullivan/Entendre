#pragma once
#include <vector>
#include <stdexcept>
#include <functional>
#include "NeuralNet.hh"


class ConsecutiveNeuralNet : public NeuralNet<Node,Connection> {
  friend class ConcurrentNeuralNet;
public:
  using NeuralNet::NeuralNet;

  void load_input_vals(const std::vector<_float_>& inputs);
  std::vector<_float_> read_output_vals();
  std::vector<_float_> evaluate(std::vector<_float_> inputs) override;

  std::vector<NodeType> node_types() const;
protected:
  _float_ get_node_val(unsigned int i);
  void add_to_val(unsigned int i, _float_ val);


private:
  void sort_connections() override;
  friend std::ostream& operator<<(std::ostream& os, const  ConsecutiveNeuralNet& net);

};
