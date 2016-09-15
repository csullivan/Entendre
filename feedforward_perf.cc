#include "Neural.hh"
#include <chrono>
#include <iostream>
#include "Timer.hh"

int main() {

  auto nTrials = 100u;
  double tperformance = 0.0;

  // Time the network evaluation
  for (auto i=0u; i < nTrials; i++)
  {
    Entendre::FeedForward network({10,20,10});

    Timer tbuild([&tperformance](int elapsed) {
        tperformance+=elapsed;
      });

    network.Feed({0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5});
  }
  std::cout << std:: endl
            << "Average time to evaluate network: "
            << tperformance/nTrials/1.0e6 << " ms\n"
            << std::endl;
  tperformance = 0.0;


  return 0;

}
