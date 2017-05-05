#pragma once
class NetProxy;

class FitnessEvaluator {
public:
  virtual ~FitnessEvaluator() {;}
  virtual void step(NetProxy&) = 0;
};


class NetProxy {
public:
  NetProxy(Organism* org) : organism(org) { ; }
  auto num_nodes() { return organism->network()->num_nodes(); }
  auto num_connections() { return organism->network()->num_connections(); }

  void request_calc(std::vector<_float_> inputs,
                std::function<void(const std::vector<_float_>&)> callback) {
    this->inputs = std::move(inputs);
    this->callback = callback;
  }
  void set_fitness_value(double fitness) { organism->fitness = fitness; }
  bool has_inputs() { return inputs.size() ? true : false; }
  void clear() { inputs.clear(); }
  std::vector<_float_> evaluate() {
    assert(has_inputs());
    return organism->network()->evaluate(inputs);
  }

  Organism* organism; // non-owning
  std::function<void(const std::vector<_float_>&)> callback;
  std::vector<_float_> inputs;
};
