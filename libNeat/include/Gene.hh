#pragma once
#include <memory>

class Link;

class Gene {
public:
  std::shared_ptr<Link> connection;
  int innovation;
  enum class Status {
    Enabled, Disabled
  };
  Status status;
};
