#pragma once
#include <vector>
#include <stdexcept>
#include <functional>
#include "NeuralNet.hh"


class ConsecutiveNeuralNet : public NeuralNetRecursiveBase<ConsecutiveNeuralNet> {
  friend class NeuralNetRecursiveBase;
public:
  //using NeuralNet::NeuralNet;
  virtual ~ConsecutiveNeuralNet() { ; }

  void load_input_vals(const std::vector<_float_>& inputs);
  std::vector<_float_> read_output_vals();
  virtual std::vector<_float_> evaluate(std::vector<_float_> inputs);

  std::vector<NodeType> node_types() const;
  virtual void add_node(const NodeType& type) { nodes.emplace_back(type); }

  virtual Connection get_connection(unsigned int i) const {
    return connections[i];
  }
  virtual NodeType get_node_type(unsigned int i) const {
    return nodes[i].type;
  }

  virtual void print_network(std::ostream& os) const;

protected:
  _float_ get_node_val(unsigned int i);
  void add_to_val(unsigned int i, _float_ val);

private:
  void sort_connections() override;


  std::vector<Node> nodes;
  std::vector<Connection> connections;
};
