#pragma once

struct GenomeConverter {
  virtual std::unique_ptr<NeuralNet> convert(const Genome&) = 0;
};
template<typename NetType>
struct GenomeConverter_Impl : GenomeConverter {
  virtual std::unique_ptr<NeuralNet> convert(const Genome& genome) {
    return genome.MakeNet<NetType>();
  }
};

struct Organism {
  Organism(const Genome& gen, std::shared_ptr<GenomeConverter> converter)
    : fitness(std::numeric_limits<double>::quiet_NaN()),
      adj_fitness(std::numeric_limits<double>::quiet_NaN()),
      genome(gen), net(nullptr), converter(converter) { ; }

  Organism(const Genome& gen, std::unique_ptr<NeuralNet>&& net)
    : fitness(std::numeric_limits<double>::quiet_NaN()),
      adj_fitness(std::numeric_limits<double>::quiet_NaN()),
      genome(gen) , net(std::move(net)), converter(nullptr) { ; }
  Organism(const Organism& org)
    : fitness(org.fitness), adj_fitness(org.adj_fitness),
      genome(org.genome), net(org.net ? org.net->clone() : nullptr),
      converter(org.converter) { ; }
  Organism& operator=(const Organism& rhs) {
    fitness = rhs.fitness;
    adj_fitness = rhs.adj_fitness;
    genome = rhs.genome;
    net = rhs.net ? rhs.net->clone() : nullptr;
    converter = rhs.converter;
    return *this;
  }
  NeuralNet* network() {
    if (net) { return net.get(); }
    net = converter->convert(genome);
    return net.get();
  }
  double fitness;
  double adj_fitness;
  Genome genome;
private:
  std::unique_ptr<NeuralNet> net;
  std::shared_ptr<GenomeConverter> converter;
};


struct Species {
  std::vector<Organism> organisms;

  unsigned int id;
  Genome representative;
  unsigned int age;
  //unsigned int age_since_last_improvement;
  double best_fitness;
};
