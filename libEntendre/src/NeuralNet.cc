#include "NeuralNet.hh"

#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>



_float_ NeuralNet::sigmoid(_float_ val) const {
  return (sigma) ? sigma(val) :
    // Logistic curve
    1/(1 + std::exp(-val));
  // Other options for a sigmoid curve
  //val/std::sqrt(1+val*val);
  //std::tanh(val);
  //std::erf((std::sqrt(M_PI)/2)*val);
  //(2/M_PI) * std::atan((M_PI/2)*val);
  //val/(1+std::abs(val));
}

std::ostream& operator<<(std::ostream& os, const NeuralNet& net) {
  net.print_network(os);
  return os;
}
