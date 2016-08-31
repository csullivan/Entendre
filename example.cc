#include "Neural.hh"
#include <random>
#include <chrono>
#include <iostream>


int main() {

  std::mt19937 mt(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<double> dis(0.0, 2.0);
  Entendre::FeedForward network({2,4,1});

  for (int i = 0; i < 5000; i++) {

    auto i1 = int(dis(mt));
    auto i2 = int(dis(mt));

    auto result = i1 xor i2;

    network.Feed({i1,i2});


    if (i%100==0) {
      auto output = network.Results();
      std::cout << i1 << " xor " << i2 << " = " << result<< std::endl;
      std::cout << "net: " << output[0] << std::endl;
    }

    network.BackPropogate({result});


  }



  return 0;

}
