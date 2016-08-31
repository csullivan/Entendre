#include <iostream>
#include "Neural.hh"

int main() {

  auto net =
    Entendre::MakeKernal<Entendre::NeuralNet>()
    .AddActivator([](){})
    .AddTransferFunction([](){})
    .SetTrainingRate(0.15)
    .SetMomentum(0.5);


  return 0;
}
