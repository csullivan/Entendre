#pragma once
#include <memory>
#include <vector>
#include <iostream>


class Link;
class Node {
public:

  enum class Type {Sensor, Neuron};
  enum class Function {Bias, Input, Hidden, Output};

  Node();
  Node(Type,Function);
  Node(const std::shared_ptr<Node>&);

  void operator<<(double val) { activation = val; }
  operator Type() { return type; }
  operator Function() { return functype; }

  std::vector<std::shared_ptr<Link> > input_links;
  std::vector<std::shared_ptr<Link> > output_links;
  double activation;
private:
  Type type;
  Function functype;
};
