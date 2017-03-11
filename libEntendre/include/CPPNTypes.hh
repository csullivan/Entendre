#pragma once

typedef float _float_;

enum class ConnectionType { Normal, Recurrent };

// Possible TO DO:
/*
  We could consider creating an enum-factory for the NodeType activation function which
  would return a class functor wrapping the correct activation function. For now, I have
  decided that the default case for switch(NodeType) will be used for hidden nodes, which
  means our internal switches only need a minor change. If we want to move toward a factory
  implementation then we will need to consider how to handle CUDA kernel launches for the
  apply_activations (previously apply_sigmoid) function. My idea would be to register GPU
  specializations of the factory which call a wrapped c++ function that itself forwards the
  arugments to a kernel launch--for each of the different activation functions seen in the
  below enum. If the nodes had activation sets, then seperate activation kernel launches would
  be made for each set, and the factory could handle this all on the CPU side of things.
*/
enum class NodeType { Bias, Input, Output,
  Sigmoid, Tanh, Relu, Gaussian, Sin, Cos, Mult, Abs, Add, MultGaussian, Square };

inline bool IsSensor(const NodeType& type) {
  return type == NodeType::Input || type == NodeType::Bias;
}
