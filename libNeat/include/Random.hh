#pragma once
#include <memory>
#include <random>
#include <chrono>

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

class uses_random_numbers {
public:
  auto get_generator() const { return generator; }
  void set_generator(const std::shared_ptr<RNG>& gen) { generator = gen; }
protected:
  double random() { return (*generator)(); }
  std::shared_ptr<RNG> generator;
};
