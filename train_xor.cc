#include "Neural.hh"
#include <random>
#include <chrono>
#include <iostream>
#include "Timer.hh"

int main() {

  std::mt19937 mt(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<double> dis(0.0, 2.0);
  Entendre::FeedForward network({2,4,4,3});

  {
    Timer wall([](int elapsed) { std::cout << "Elapsed: " << elapsed/1.e6 << std::endl;});
    for (int i = 0; i < 200000; i++) {

      auto i1 = int(dis(mt));
      auto i2 = int(dis(mt));

      auto result = i1 xor i2;

      network.Feed({i1,i2});


      if (i%100==0) {
        auto output = network.Results();
        std::cout << i1 << " xor " << i2 << " = " << result<< std::endl;
        std::cout << "net: ";
        for (auto& val : output) { std::cout << val << " ";}
        std::cout << std::endl;
      }

      network.BackPropogate({result});


    }
  }

  return 0;

}
