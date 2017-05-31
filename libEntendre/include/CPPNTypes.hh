#pragma once
#include <map>
#include <cmath>
#include <functional>

typedef float _float_;

// TODO:
/*
  Consider creating an enum-factory for the NodeType activation function which
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
enum class NodeType {
  Bias,
    Input,
    Output,
    Hidden
    };

enum class ActivationFunction {
  Sigmoid,
    Identity,
    Tanh,
    Relu,
    Gaussian,
    Sin,
    Cos,
    Abs,
    Square,
  //Cube,
  //Log,
  //Exp,
};

namespace {
  static const _float_ max_exp_signal = std::log(std::numeric_limits<_float_>::max());

  inline _float_ clip(_float_ val, _float_ low, _float_ high) {
    return (val > high) ? high :
      (val < low ) ? low  : val;
  }
  inline _float_ clip_low(_float_ val, _float_ low) {
    return (val < low ) ? low  : val;
  }
  inline _float_ clip_high(_float_ val, _float_ high) {
    return (val > high) ? high : val;
  }

  inline _float_ identity     (_float_ val) { return val; }
  inline _float_ tanh         (_float_ val) { return std::tanh(val); }
  inline _float_ relu         (_float_ val) { return std::max(_float_(0.),val); }
  inline _float_ gaussian     (_float_ val) { return std::exp(-val*val/2.); }
  inline _float_ sin          (_float_ val) { return std::sin(val); }
  inline _float_ cos          (_float_ val) { return std::cos(val); }
  inline _float_ abs          (_float_ val) { return std::abs(val); }
  inline _float_ square       (_float_ val) { return val*val; }
  //inline _float_ cube         (_float_ val) { return std::pow(val,3); }
  //inline _float_ log          (_float_ val) { return std::log(std::abs(val)); }
  //inline _float_ exp          (_float_ val) { return std::exp(clip_high(val,50)); }
  inline _float_ logistic     (_float_ val) { return 1/(1 + std::exp(-val)); }


  inline _float_ activate     (ActivationFunction type, _float_ val) {
    switch(type) {
    case ActivationFunction::Sigmoid:
      return logistic(val);
    case ActivationFunction::Identity:
      return identity(val);
    case ActivationFunction::Tanh:
      return tanh(val);
    case ActivationFunction::Relu:
      return relu(val);
    case ActivationFunction::Gaussian:
      return gaussian(val);
    case ActivationFunction::Sin:
      return sin(val);
    case ActivationFunction::Cos:
      return cos(val);
    case ActivationFunction::Abs:
      return abs(val);
    case ActivationFunction::Square:
      return square(val);
    // case ActivationFunction::Cube:
    //   return cube(val);
    // case ActivationFunction::Log:
    //   return log(val);
      //case ActivationFunction::Exp:
      //return exp(val);
    default:
      throw std::runtime_error("Unimplemented ActivationFunction activation function.");
    };
  }
}
