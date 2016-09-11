#pragma once
#include <memory>
#include <random>
#include <chrono>

class RNG {
public:
  RNG() {
    std::random_device rd;
    if (rd.entropy() != 0) {
      mt = std::make_unique<std::mt19937>(rd());
    } else {
      mt = std::make_unique<std::mt19937>
        (std::chrono::system_clock::now().time_since_epoch().count());
    }
  }
  RNG(unsigned long seed) : mt(std::make_unique<std::mt19937>(seed)) {;}
  virtual ~RNG() { ; }
  virtual double operator()() = 0;

protected:
  std::unique_ptr<std::mt19937> mt;
};


class Gaussian : public RNG {
public:
  Gaussian(double mean, double sigma): RNG(), dist(mean,sigma) {;}
  double operator()() override { return dist(*mt); }
private:
  std::normal_distribution<double> dist;
};


class Uniform : public RNG {
public:
  Uniform(double min, double max): RNG(), dist(min,max) {;}
  double operator()() override { return dist(*mt); }
private:
  std::uniform_real_distribution<double> dist;
};


class uses_random_numbers {
public:
  auto get_generator() const { return generator; }
  void set_generator(const std::shared_ptr<RNG>& gen) { generator = gen; }
protected:
  double random() { return (*generator)(); }
  std::shared_ptr<RNG> generator;
};
