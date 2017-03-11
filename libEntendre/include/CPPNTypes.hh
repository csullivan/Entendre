#pragma once
#include <map>
#include <cmath>
#include <functional>

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
    Sigmoid, Tanh, Relu, Gaussian, Sin, Cos, Abs, Square, // simple activation functions
    Add, Mult, MultGaussian // custom activation functions
    };

inline bool IsSensor(const NodeType& type) {
  return type == NodeType::Input || type == NodeType::Bias;
}

namespace {
  inline _float_ Identity     (const _float_ val) { return val; }
  inline _float_ Sigmoid      (const _float_ val) { return 1/(1 + std::exp(-val)); }
  inline _float_ Tanh         (const _float_ val) { return std::tanh(val); }
  inline _float_ Relu         (const _float_ val) { return std::max(_float_(0.),val); }
  inline _float_ Gaussian     (const _float_ val) { return std::exp(-val*val/2.); }
  inline _float_ Sin          (const _float_ val) { return std::sin(val); }
  inline _float_ Cos          (const _float_ val) { return std::cos(val); }
  inline _float_ Abs          (const _float_ val) { return std::abs(val); }
  inline _float_ Square       (const _float_ val) { return val*val; }

  inline _float_ Mult         (const _float_ val) { return val; } // Mult implementation needed
  inline _float_ Add          (const _float_ val) { return val; } // Add implementation needed
  //inline _float_ MultGaussian (const _float_ val) { return val; } // MultGaussian implementation needed

  std::map<NodeType,std::function<_float_(const _float_&)>> activation_functions = {
    {NodeType::Bias, Identity},
    {NodeType::Input, Identity},
    {NodeType::Output, Sigmoid},
    {NodeType::Sigmoid, Sigmoid},
    {NodeType::Tanh, Tanh},
    {NodeType::Relu, Relu},
    {NodeType::Gaussian, Gaussian},
    {NodeType::Sin, Sin},
    {NodeType::Cos, Cos},
    {NodeType::Abs, Abs},
    {NodeType::Square, Square},
    {NodeType::Add, Add},
    {NodeType::Mult, Mult},
    {NodeType::MultGaussian, Gaussian}
  };
}
