#pragma once
#include <memory>

class Link;
class Node;

class Gene {
public:
  enum class Status {Enabled, Disabled};
  Gene(std::shared_ptr<Node>& in,
       std::shared_ptr<Node>& out,
       double weight,
       Gene::Status s,
       int ninnovation);

  std::shared_ptr<Link> connection;
  int innovation;
  Status status;
};
