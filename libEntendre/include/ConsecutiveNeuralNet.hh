#pragma once
#include <vector>
#include <stdexcept>
#include <functional>
#include "NeuralNet.hh"


class ConsecutiveNeuralNet : public NeuralNet<Node,Connection> {
  friend class ConcurrentNeuralNet;
public:
  using NeuralNet::NeuralNet;

  void load_input_vals(const std::vector<float>& inputs);
  std::vector<float> read_output_vals();
  std::vector<float> evaluate(std::vector<float> inputs) override;

  std::vector<NodeType> node_types() const;
protected:
  float get_node_val(unsigned int i);
  void add_to_val(unsigned int i, float val);


private:
  void sort_connections() override;
  friend std::ostream& operator<<(std::ostream& os, const  ConsecutiveNeuralNet& net);

};
