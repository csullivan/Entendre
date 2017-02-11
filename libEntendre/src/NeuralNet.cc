#include "NeuralNet.hh"

#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>


template<>
NeuralNet<Node,Connection>::NeuralNet(std::vector<Node>&& Nodes, std::vector<Connection>&& Conn)
  : nodes(std::move(Nodes)), connections(std::move(Conn)), connections_sorted(false) { ; }

template<>
NeuralNet<Node,Connection>::NeuralNet(const std::vector<Node>& nodes) : nodes(nodes), connections_sorted(false) { ; }

template<>
NeuralNet<float,Connection>::NeuralNet(std::vector<float>&& Nodes, std::vector<Connection>&& Conn)
  : nodes(std::move(Nodes)), connections(std::move(Conn)), connections_sorted(false) { ; }

template<>
NeuralNet<float,Connection>::NeuralNet(const std::vector<float>& nodes) : nodes(nodes), connections_sorted(false) { ; }


bool IsSensor(const NodeType& type) {
  return type == NodeType::Input || type == NodeType::Bias;
}

