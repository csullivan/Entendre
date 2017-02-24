#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>

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

  // no recurrent
  Probabilities params;
  params.new_connection_is_recurrent = 0;

  Population pop(seed,
                 std::make_shared<RNG_MersenneTwister>(),
                 std::make_shared<Probabilities>(params));

  pop.SetNetType<ConsecutiveNeuralNet>();

  auto max_generations = 2000u;

  std::vector<vector<_float_>> possible_inputs = {
    {0,0},
    {0,1},
    {1,0},
    {1,1}
  };

  for (auto& input : possible_inputs) {
    if (input[0] == input[1]) {
      input.push_back(0);
    } else {
      input.push_back(1);
    }
  }

  std::unique_ptr<NeuralNet> winner = nullptr;
  unsigned int generation;

  auto show = [&](){
    auto best = pop.BestNet();
    if(!best) { return; }
    std::cout << " ----------- Gen " << generation << " ----------------" << std::endl;
    std::cout << pop.NumSpecies() << " species total" << std::endl;
    std::cout << "Best (nodes, conn) = (" << best->num_nodes() << ", " << best->num_connections()
    << ")" << std::endl;
    _float_ error = 0;
    for(auto& input : possible_inputs) {
      _float_ val = best->evaluate(input)[0];
      std::cout << input[0] << " ^ " << input[1] << " = " << val << std::endl;;
      error += std::abs(val - input[2]);
    }
    std::cout << "Error: " << error << std::endl;
  };

  for (generation = 0u; generation < max_generations; generation++) {

    auto next_gen = pop.Reproduce(
      // fitness lambda
      [&](NeuralNet& net) {

        // randomize input order and then create the solution set
        std::random_shuffle(possible_inputs.begin(),possible_inputs.end());

        std::vector<_float_> output;
        for (auto& inputs : possible_inputs) {
          output.push_back(net.evaluate(inputs)[0]);
        }

        _float_ error = 0;
        for (auto i=0u; i< possible_inputs.size(); i++) {
          error += (possible_inputs[i][2] == 0) ? std::pow(output[i],2) : std::pow(1-output[i],2);
          //error += (possible_inputs[i][2] == 0) ? std::abs(output[i]) : std::abs(1-output[i]);
        }

        // look for a winner
        int truth_count = 0;
        for (auto i=0u; i< possible_inputs.size(); i++) {
          truth_count += (possible_inputs[i][2] == 0)  ?  ((output[i]<0.5) ? 1:0) : ((output[i]>=0.5) ? 1:0);
        }
        if (truth_count == 4) {
          winner = net.clone();
        }

        return std::pow(4.-error,2);
      });


    if(generation%10 == 0) {
      show();
    }

    // stop the simulation if a winner is found
    if (winner) { break; }

    pop = std::move(next_gen);
  }

  show();

  if (winner) {
    auto nodesize = winner->num_nodes();
    auto connsize = winner->num_connections();
    std::cout << "Winner found in generation " << generation  << ". Winning network size (nodes,connections): (" << nodesize << "," << connsize << ")"<< std::endl; ;
    std:: cout << *winner << std::endl;
  } else {
    std::cout << "No winner found after " << max_generations << " generations." << std::endl;
  }

  // for (auto& input : possible_inputs) {
  //   auto results = winner->evaluate(input);
  //   std::cout << input[0] << " xor " << input[1] << " = ";
  //   std::cout << std::round(results[0]) << std::endl;
  // }



  return 0;

}
