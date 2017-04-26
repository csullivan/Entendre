#pragma once
#include "NeuralNet_CRTP.hh"

#include <vector>
#include <stdexcept>
#include <functional>


class ConsecutiveNeuralNet : public NeuralNet_CRTP<ConsecutiveNeuralNet> {
  friend class NeuralNet_CRTP;
public:
  //using NeuralNet::NeuralNet;
  virtual ~ConsecutiveNeuralNet() { ; }

  void load_input_vals(const std::vector<_float_>& inputs);
  std::vector<_float_> read_output_vals();
  virtual std::vector<_float_> evaluate(std::vector<_float_> inputs);

  std::vector<NodeType> node_types() const;
  virtual void add_node(NodeType type, ActivationFunction func) {
    nodes.emplace_back(type,func);
  }

  virtual Connection get_connection(unsigned int i) const {
    return connections[i];
  }
  virtual NodeType get_node_type(unsigned int i) const {
    return nodes[i].type;
  }
  void sort_connections(unsigned int first=0, unsigned int num_connections=0) override;

  virtual ActivationFunction get_activation_func(unsigned int i) const {
    return nodes[i].func;
  }

  virtual void print_network(std::ostream& os) const;

protected:
  _float_ get_node_val(unsigned int i);
  void add_to_val(unsigned int i, _float_ val);
  void mult_into_val(unsigned int i, _float_ val);

private:


  std::vector<Node> nodes;
  std::vector<Connection> connections;
};
