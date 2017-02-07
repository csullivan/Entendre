#include "Genome.hh"


int main() {

  auto sigmoid = [](double val) {return 1/(1 + std::exp(-val));};
  auto genome = Genome()
    .AddNode(NodeType::Bias)
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Hidden)
    .AddNode(NodeType::Output)
    .AddConnection(0,3,true,1.)
    .AddConnection(1,3,true,1.)
    .AddConnection(1,2,true,1.)
    .AddConnection(2,3,true,1.)
    .AddConnection(3,2,true,1.); // recurrent
  auto net = NeuralNet(genome);
  net.register_sigmoid(sigmoid);
  //auto result = net.evaluate({0.5});

  LFNetwork lfnet(std::move(net));
  lfnet.sort_connections();
  for (auto const& conn : lfnet.get_connections()) {
    std::cout << "origin: " << conn.origin << " dest: " << conn.dest <<
      " evaluation set: " << conn.set << std::endl;
  }
  std::cout << std::endl;
  lfnet.sort_connections();
  for (auto const& conn : lfnet.get_connections()) {
    std::cout << "origin: " << conn.origin << " dest: " << conn.dest <<
      " evaluation set: " << conn.set << std::endl;
  }



}
