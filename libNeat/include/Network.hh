#pragma once
#include <memory>
#include <vector>
#include <stdexcept>

class Genome;
class Node;

// A network is the current real world view of an organisms
// internal network. It is what is used for performing calc-
// ulations and represents the state of all enabled links of
// a genome neuron set. Otherwise known as a Pheonotype.

class Network { friend class Genome;
  int id;
  std::vector<std::shared_ptr<Node>> nodes;
  std::shared_ptr<Genome> genotype;
  std::vector<std::shared_ptr<Node>> inputs;
  std::vector<std::shared_ptr<Node>> outputs;
public:
  Network(const std::shared_ptr<Genome>& genome, int id_=-1);
  void LoadInputs(const std::vector<double>&);
  double Activate();
};


class NetworkException : public std::exception {
  using std::exception::exception;
};

class NetworkSensorSize : public NetworkException {
  using NetworkException::NetworkException;
};
