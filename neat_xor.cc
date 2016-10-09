#include <random>
#include <chrono>
#include <cmath>
#include <iostream>

#include "Population.hh"
#include "Timer.hh"

int main() {

  auto seed = Genome()
    .AddNode(NodeType::Bias)
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Input)
    .AddNode(NodeType::Output)
    .AddConnection(0,3,true,1.)
    .AddConnection(1,3,true,1.)
    .AddConnection(2,3,true,1.);

  Population pop(seed,
                 std::make_shared<Uniform>(0,1),
                 std::make_shared<Probabilities>());

  auto max_generations = 200u;

  std::vector<vector<double>> possible_inputs = {
    {0,0},
    {0,1},
    {1,0},
    {1,1}
  };

  NeuralNet* winner;
  unsigned int generation;
  for (generation = 0u; generation < max_generations; generation++) {


    pop = pop.Reproduce(
      // fitness lambda
      [&](NeuralNet& net) {
        std::vector<double> output;
        for (auto& inputs : possible_inputs) {
          output.push_back(net.evaluate(inputs)[0]);
        }
        double error = (
          std::abs(output[0])+
          std::abs(1-output[1])+
          std::abs(1-output[2])+
          std::abs(output[3]) );

        // look for a winner
        if ( output[0]<0.5 && output[1]>=0.5 && output[2]>=0.5 && output[3]<0.5 )
        {
          winner = &net;
        }

        return std::pow(4.-error,2);
      });

    if (winner) { break; }

  }

  std::cout << "Winner found in generation " << generation  << ".\n";

  return 0;

}
