#pragma once
#include <memory>
#include <random>
#include <chrono>
#include <fstream>
#include <iostream>
#include <queue>


class RNG {
public:
  virtual ~RNG() { }

  double operator()() { return uniform(0, 1); }
  virtual double uniform(double min=0, double max=1) = 0;
  virtual double gaussian(double mean=0, double sigma=1) = 0;
};

class RNG_MersenneTwister : public RNG {
public:
  RNG_MersenneTwister() {
    std::random_device rd;
    if (rd.entropy() != 0) {
      mt = std::make_unique<std::mt19937>(rd());
    } else {
      mt = std::make_unique<std::mt19937>
        (std::chrono::system_clock::now().time_since_epoch().count());
    }
  }

  RNG_MersenneTwister(unsigned long seed)
    : mt(std::make_unique<std::mt19937>(seed)) { }

  virtual ~RNG_MersenneTwister() { }

  virtual double uniform(double min=0, double max=1) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(*mt);
  }

  virtual double gaussian(double mean=0, double sigma=1) {
    std::normal_distribution<double> dist(mean, sigma);
    return dist(*mt);
  }

protected:
  std::unique_ptr<std::mt19937> mt;
};

// class UniformLogger : public RNG {
// public:
//   UniformLogger(double min, double max): RNG(), dist(min,max) {;}
//   double operator()() override {
//     auto rand = dist(*mt);
//     numbers.push_back(rand);
//     return rand;
//   }
//   void save_all(std::string filename="./random_numbers.dat") {
//     std::cout << "Saving numbers to: " << filename << std::endl;
//     std::ofstream ofs;
//     ofs.open(filename, std::ofstream::out);
//     ofs.precision(17);
//     for (auto& num : numbers) {
//       ofs << num;
//       ofs << std::endl;
//     }
//     ofs.close();
//   }
// private:
//   std::vector<double> numbers;
//   std::uniform_real_distribution<double> dist;
// };

// class Cached : public RNG {
// public:
//   Cached(std::string filename="./random_numbers.dat"): RNG() {
//     std::ifstream ifs;
//     ifs.open(filename);
//     std::string line;

//     while (std::getline(ifs, line)) {
//       numbers.push(std::stod(line));
//     }

//   }
//   double operator()() override {
//     auto rand = numbers.front();
//     numbers.pop();
//     return rand;
//   }
// private:
//   std::queue<double> numbers;
// };

class uses_random_numbers {
public:
  auto get_generator() const { return generator; }
  void set_generator(const std::shared_ptr<RNG>& gen) { generator = gen; }
protected:
  double random() { return (*generator)(); }
  std::shared_ptr<RNG> generator;
};
