#include <iostream>

int main() {

  Entendre::Neural net;
  auto kernal =
    Entendre::MakeKernal<Entendre::Neural>()
    .AddActivator([](){})
    .AddTransferFunction([](){})
    .SetEta()
    .SetMomentum();


  return 0;
}
